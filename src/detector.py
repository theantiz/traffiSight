import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time


class TrafficTracker:
    """Traffic detection, tracking, and analytics engine."""

    VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

        # Per-track state
        self.track_history: dict[int, list] = defaultdict(list)
        self.speeds: dict[int, float] = {}

        # Line crossing
        self.crossed_ids: set[int] = set()
        self.cross_count = {"in": 0, "out": 0}
        self.line_y_ratio = 0.6
        self.line_y: int | None = None

        # Heatmap
        self.heatmap: np.ndarray | None = None

        # Session-level aggregates
        self.all_seen_ids: set[int] = set()
        self.max_speed: float = 0.0
        self.peak_active: int = 0
        self.crossing_events: list[dict] = []  # recent crossing log
        self.frame_count: int = 0

    # speed
    def _estimate_speed(self, tid: int, cx: float, cy: float, ts: float) -> float:
        h = self.track_history[tid]
        h.append((cx, cy, ts))
        if len(h) > 10:
            h.pop(0)
        if len(h) >= 2:
            x1, y1, t1 = h[0]
            x2, y2, t2 = h[-1]
            dt = t2 - t1
            if dt > 0:
                dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                spd = (dist / dt) * 0.05 * 3.6
                spd = round(spd, 1)
                self.speeds[tid] = spd
                if spd > self.max_speed:
                    self.max_speed = spd
                return spd
        return 0.0

    # line crossing
    def _check_line_crossing(self, tid: int, cy: float, cls_name: str, ts: float) -> None:
        h = self.track_history[tid]
        if len(h) >= 2 and tid not in self.crossed_ids:
            prev_cy = h[-2][1]
            direction = None
            if prev_cy < self.line_y <= cy:
                direction = "out"
            elif prev_cy > self.line_y >= cy:
                direction = "in"
            if direction:
                self.cross_count[direction] += 1
                self.crossed_ids.add(tid)
                evt = {"id": tid, "dir": direction, "cls": cls_name, "t": ts}
                self.crossing_events.append(evt)
                if len(self.crossing_events) > 50:
                    self.crossing_events.pop(0)

    # heatmap
    def _update_heatmap(self, cx: float, cy: float) -> None:
        if self.heatmap is not None:
            cv2.circle(self.heatmap, (int(cx), int(cy)), 20, 1, -1)

    def get_heatmap_overlay(self, frame: np.ndarray) -> np.ndarray:
        if self.heatmap is None or self.heatmap.max() == 0:
            return frame
        n = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        c = cv2.applyColorMap(n.astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.6, c, 0.4, 0)

    # main loop
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        h, w = frame.shape[:2]
        if self.heatmap is None:
            self.heatmap = np.zeros((h, w), dtype=np.float32)
        if self.line_y is None:
            self.line_y = int(h * self.line_y_ratio)

        self.frame_count += 1
        ts = time.time()

        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        annotated = results[0].plot()

        # counting line
        cv2.line(annotated, (0, self.line_y), (w, self.line_y), (0, 255, 255), 2)

        detections = []
        class_counts: dict[str, int] = defaultdict(int)
        frame_speeds: list[float] = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            tids = results[0].boxes.id.int().cpu().numpy()
            cids = results[0].boxes.cls.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, tid, cid, conf in zip(boxes, tids, cids, confs):
                cx, cy, bw, bh = box
                cid, tid = int(cid), int(tid)
                if cid not in self.VEHICLE_CLASSES:
                    continue

                cls_name = self.model.names.get(cid, "unknown")
                class_counts[cls_name] += 1
                self.all_seen_ids.add(tid)

                speed = self._estimate_speed(tid, float(cx), float(cy), ts)
                self._check_line_crossing(tid, float(cy), cls_name, ts)
                self._update_heatmap(float(cx), float(cy))

                if speed > 0:
                    frame_speeds.append(speed)
                    cv2.putText(annotated, f"{speed:.0f} km/h",
                                (int(cx - bw / 2), int(cy - bh / 2 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                detections.append({
                    "track_id": tid,
                    "class": cls_name,
                    "confidence": round(float(conf), 2),
                    "speed_kmh": speed,
                })

        active = len(detections)
        if active > self.peak_active:
            self.peak_active = active

        cv2.putText(annotated,
                    f"IN: {self.cross_count['in']}  OUT: {self.cross_count['out']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        avg_spd = round(float(np.mean(frame_speeds)), 1) if frame_speeds else 0.0
        max_spd_frame = round(float(max(frame_speeds)), 1) if frame_speeds else 0.0

        # speed distribution buckets: 0-20, 20-40, 40-60, 60-80, 80+
        all_speeds = [s for s in self.speeds.values() if s > 0]
        speed_dist = [0, 0, 0, 0, 0]
        for s in all_speeds:
            idx = min(int(s // 20), 4)
            speed_dist[idx] += 1

        stats = {
            "detections": detections,
            "class_counts": dict(class_counts),
            "total_active": active,
            "line_crossings": dict(self.cross_count),
            "avg_speed": avg_spd,
            "max_speed_frame": max_spd_frame,
            "max_speed_session": round(self.max_speed, 1),
            "peak_active": self.peak_active,
            "total_unique": len(self.all_seen_ids),
            "speed_distribution": speed_dist,
            "recent_events": self.crossing_events[-10:],
            "frame_num": self.frame_count,
        }
        return annotated, stats
