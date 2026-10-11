import asyncio
import base64
import csv
import os
import time
from datetime import datetime

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse

from src.detector import TrafficTracker

app = FastAPI(title="TraffiSight API")
tracker = TrafficTracker()

# logging
LOG_DIR = "data/logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"traffic_{datetime.now():%Y%m%d_%H%M%S}.csv")

stats_history: list[dict] = []
MAX_HISTORY = 2000
session_start = time.time()
peak_count = 0


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


# routes
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
async def api_upload(file: UploadFile = File(...)):
    path = f"data/{file.filename}"
    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)
    return {"status": "ok", "path": path, "size_bytes": len(content)}


# websocket
@app.websocket("/ws/traffic")
async def ws_traffic(websocket: WebSocket):
    global peak_count
    await websocket.accept()
    cap = cv2.VideoCapture("data/traffic.mp4")
    show_heatmap = False
    last_time = time.time()
    fps = 0.0

    try:
        while cap.isOpened():
            # non-blocking control messages
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                if msg == "toggle_heatmap":
                    show_heatmap = not show_heatmap
            except asyncio.TimeoutError:
                pass

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            t0 = time.time()
            annotated, stats = tracker.process_frame(frame)
            proc_ms = (time.time() - t0) * 1000

            # fps
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
    finally:
        cap.release()
