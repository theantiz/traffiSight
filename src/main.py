import asyncio
import base64
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse

from fastapi.middleware.cors import CORSMiddleware

from src.detector import TrafficTracker

app = FastAPI(title="TraffiSight API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tracker = TrafficTracker()

# ── logging ──
LOG_DIR = "data/logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"traffic_{datetime.now():%Y%m%d_%H%M%S}.csv")

stats_history: list[dict] = []
MAX_HISTORY = 2000
session_start = time.time()
peak_count = 0


# ── video state manager ──
class VideoState:
    def __init__(self):
        self.path = None
        os.makedirs("data", exist_ok=True)
        if os.path.exists("data/traffic.mp4"):
            self.path = "data/traffic.mp4"
        else:
            # Check for any existing mp4 in data/
            for f in os.listdir("data"):
                if f.lower().endswith(".mp4"):
                    self.path = os.path.join("data", f)
                    break
        self.version = 1 if self.path else 0


video_state = VideoState()


def _init_csv():
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow([
            "timestamp", "total_active", "cars", "trucks", "buses",
            "motorcycles", "avg_speed", "crossings_in", "crossings_out",
            "congestion",
        ])


def _log_row(s: dict):
    cc = s.get("class_counts", {})
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(),
            s.get("total_active", 0),
            cc.get("car", 0), cc.get("truck", 0),
            cc.get("bus", 0), cc.get("motorcycle", 0),
            s.get("avg_speed", 0),
            s["line_crossings"].get("in", 0),
            s["line_crossings"].get("out", 0),
            s.get("congestion", "free"),
        ])


_init_csv()


def _congestion(n: int) -> str:
    if n > 15:
        return "heavy"
    if n > 8:
        return "moderate"
    return "free"


# ── routes ──
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/api/stats")
async def api_stats():
    if stats_history:
        return stats_history[-1]
    return {"total_active": 0, "class_counts": {}, "line_crossings": {"in": 0, "out": 0}}


@app.get("/api/history")
async def api_history(limit: int = 200):
    return stats_history[-limit:]


@app.get("/api/download-log")
async def api_download():
    return FileResponse(LOG_FILE, filename="traffic_log.csv", media_type="text/csv")


@app.post("/api/upload-video")
@app.post("/api/upload-video/")
async def api_upload(file: UploadFile = File(...)):
    global peak_count, stats_history
    filename = file.filename or "uploaded_video.mp4"
    path = os.path.join("data", filename)

    # Stream write in 1MB chunks for large files
    with open(path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    video_state.path = path
    video_state.version += 1

    tracker.reset()
    peak_count = 0
    stats_history.clear()

    return {
        "status": "ok",
        "path": path,
        "filename": filename,
        "size_bytes": os.path.getsize(path),
    }


# ── websocket ──
@app.websocket("/ws/traffic")
async def ws_traffic(websocket: WebSocket):
    global peak_count
    await websocket.accept()

    cap = None
    show_heatmap = False

    async def receive_controls():
        nonlocal show_heatmap
        try:
            while True:
                msg = await websocket.receive_text()
                if msg == "toggle_heatmap":
                    show_heatmap = not show_heatmap
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception:
            pass

    recv_task = asyncio.create_task(receive_controls())

    cap = None
    loaded_version = -1
    last_time = time.time()
    fps = 0.0

    try:
        while True:
            # Check if video changed or needs initialization
            if loaded_version != video_state.version:
                if cap is not None:
                    cap.release()
                    cap = None

                if video_state.path and os.path.exists(video_state.path):
                    new_cap = cv2.VideoCapture(video_state.path)
                    if new_cap.isOpened():
                        cap = new_cap
                        loaded_version = video_state.version
                        tracker.reset()
                    else:
                        new_cap.release()
                else:
                    loaded_version = video_state.version

            # If no video is available yet, send a standby placeholder frame
            if cap is None or not cap.isOpened():
                standby_frame = np.zeros((360, 640, 3), dtype=np.uint8)
                standby_frame[:] = (18, 18, 18)
                cv2.putText(
                    standby_frame,
                    "TRAFFISIGHT // NO VIDEO LOADED",
                    (60, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (110, 110, 110),
                    2,
                )
                cv2.putText(
                    standby_frame,
                    "Upload an MP4 video in the sidebar to begin analysis",
                    (60, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (70, 70, 70),
                    1,
                )

                _, buf = cv2.imencode(".jpg", standby_frame)
                b64 = base64.b64encode(buf).decode("utf-8")

                await websocket.send_json({
                    "status": "waiting_for_video",
                    "frame": f"data:image/jpeg;base64,{b64}",
                    "total_active": 0,
                    "class_counts": {},
                    "line_crossings": {"in": 0, "out": 0},
                    "avg_speed": 0,
                    "max_speed_session": 0,
                    "peak_active": 0,
                    "total_unique": 0,
                    "congestion": "free",
                    "fps": 0,
                    "proc_ms": 0,
                    "uptime_s": round(time.time() - session_start),
                    "frame_num": 0,
                    "recent_events": [],
                    "speed_distribution": [0, 0, 0, 0, 0],
                })
                await asyncio.sleep(0.5)
                continue

            # Read frame
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.05)
                    continue

            t0 = time.time()
            annotated, stats = tracker.process_frame(frame)
            proc_ms = (time.time() - t0) * 1000

            now = time.time()
            dt = now - last_time
            fps = 1.0 / dt if dt > 0 else 0
            last_time = now

            if show_heatmap:
                annotated = tracker.get_heatmap_overlay(frame)

            n = stats["total_active"]
            stats["congestion"] = _congestion(n)
            stats["timestamp"] = datetime.now().isoformat()
            stats["fps"] = round(fps, 1)
            stats["proc_ms"] = round(proc_ms, 1)
            stats["uptime_s"] = round(now - session_start)

            if n > peak_count:
                peak_count = n

            _log_row(stats)
            stats_history.append(stats)
            if len(stats_history) > MAX_HISTORY:
                stats_history.pop(0)

            _, buf = cv2.imencode(".jpg", annotated)
            b64 = base64.b64encode(buf).decode("utf-8")

            payload = {k: v for k, v in stats.items() if k != "detections"}
            payload["frame"] = f"data:image/jpeg;base64,{b64}"

            await websocket.send_json(payload)
            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket session ended: {e}")
    finally:
        recv_task.cancel()
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=["src"])
