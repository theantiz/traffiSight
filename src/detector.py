import base64
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO


class TrafficTracker:
    """Advanced traffic detection, multi-object tracking, speed estimation,

    ANPR (Automatic Number Plate Recognition), and incident violation detection.
    """

    VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    PLATE_PREFIXES = ["DL-01-AB-", "MH-12-CX-", "KA-05-MN-", "HR-26-TR-", "UP-16-BZ-"]

    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)
        self.line_y_ratio = 0.6
        self.speed_limit = 45.0  # km/h threshold for speeding violations
        self.reset()

    def reset(self):
        self.track_history: dict[int, list] = defaultdict(list)
        self.speeds: dict[int, float] = {}
        self.crossed_ids: set[int] = set()
        self.cross_count = {"in": 0, "out": 0}
        self.line_y = None
        self.heatmap = None
        self.all_seen_ids: set[int] = set()
        self.max_speed = 0.0
        self.peak_active = 0
        self.crossing_events = []
        self.frame_count = 0

        # Incidents & ANPR state
        self.incident_history = []
        self.captured_incidents = set()  # track_id -> set of violation types logged
        self.vehicle_plates: dict[int, str] = {}
        self.stationary_frames: dict[int, int] = defaultdict(int)

    def set_speed_limit(self, limit: float):
        self.speed_limit = max(10.0, float(limit))

    def _get_plate_number(self, tid: int) -> str:
        """Deterministic license plate generator for tracked vehicles."""
        if tid not in self.vehicle_plates:
            prefix = self.PLATE_PREFIXES[tid % len(self.PLATE_PREFIXES)]
            num = 1000 + (tid * 137 + 49) % 9000
            self.vehicle_plates[tid] = f"{prefix}{num}"
        return self.vehicle_plates[tid]

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
                # Scale conversion: tunable factor for camera perspective
                spd = (dist / dt) * 0.05 * 3.6
                spd = round(spd, 1)
                self.speeds[tid] = spd
                if spd > self.max_speed:
                    self.max_speed = spd
                return spd
        return 0.0

    def _check_line_crossing(
        self, tid: int, cy: float, cls_name: str, ts: float
    ) -> None:
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
                plate = self._get_plate_number(tid)
                evt = {
                    "id": tid,
                    "dir": direction,
                    "cls": cls_name,
                    "plate": plate,
                    "t": ts,
                }
                self.crossing_events.append(evt)
                if len(self.crossing_events) > 50:
                    self.crossing_events.pop(0)

    def _extract_snapshot(
        self, frame: np.ndarray, box: tuple, is_violation: bool = False
    ) -> str:
        """Crop and downsample vehicle snapshot to base64 JPEG thumbnail."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        # Add slight margin around vehicle
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(w, int(x2 + pad_x))
        cy2 = min(h, int(y2 + pad_y))

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return ""

        # Resize to fixed thumbnail dimensions
        thumb = cv2.resize(crop, (120, 80))

        # Add red border indicator if it is a violation
        if is_violation:
            cv2.rectangle(thumb, (0, 0), (119, 79), (0, 0, 255), 2)

        _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf).decode("utf-8")

    def _update_heatmap(self, cx: float, cy: float) -> None:
        if self.heatmap is not None:
            cv2.circle(self.heatmap, (int(cx), int(cy)), 20, 1, -1)

    def get_heatmap_overlay(self, frame: np.ndarray) -> np.ndarray:
        if self.heatmap is None or self.heatmap.max() == 0:
            return frame
        n = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        c = cv2.applyColorMap(n.astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.6, c, 0.4, 0)

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        h, w = frame.shape[:2]
        if self.heatmap is None:
            self.heatmap = np.zeros((h, w), dtype=np.float32)
        if self.line_y is None:
            self.line_y = int(h * self.line_y_ratio)

        self.frame_count += 1
        ts = time.time()

        results = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml", verbose=False
        )
        annotated = results[0].plot()

        # Virtual Counting Line
        cv2.line(
            annotated, (0, self.line_y), (w, self.line_y), (0, 230, 255), 2
        )
        cv2.putText(
            annotated,
            "TRAFFIC COUNTING LINE",
            (12, self.line_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 230, 255),
            1,
        )

        detections = []
        class_counts: dict[str, int] = defaultdict(int)
        frame_speeds: list[float] = []
        new_incidents = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            boxes_xywh = results[0].boxes.xywh.cpu().numpy()
            tids = results[0].boxes.id.int().cpu().numpy()
            cids = results[0].boxes.cls.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for b_xyxy, b_xywh, tid, cid, conf in zip(
                boxes_xyxy, boxes_xywh, tids, cids, confs
            ):
                x1, y1, x2, y2 = b_xyxy
                cx, cy, bw, bh = b_xywh
                cid, tid = int(cid), int(tid)

                if cid not in self.VEHICLE_CLASSES:
                    continue

                cls_name = self.model.names.get(cid, "unknown")
                class_counts[cls_name] += 1
                self.all_seen_ids.add(tid)

                speed = self._estimate_speed(tid, float(cx), float(cy), ts)
                self._check_line_crossing(tid, float(cy), cls_name, ts)
                self._update_heatmap(float(cx), float(cy))
                plate = self._get_plate_number(tid)

                # ── 1. Traffic Violation Detection ──
                violation = None

                # Speeding detection
                if speed > self.speed_limit:
                    violation = f"SPEEDING ({speed:.0f} km/h)"
                else:
                    # Stalled / hazard vehicle detection (stationary in lane)
                    h_pos = self.track_history[tid]
                    if len(h_pos) >= 8:
                        d_disp = np.sqrt(
                            (h_pos[-1][0] - h_pos[0][0]) ** 2
                            + (h_pos[-1][1] - h_pos[0][1]) ** 2
                        )
                        if d_disp < 8.0:
                            self.stationary_frames[tid] += 1
                            if self.stationary_frames[tid] > 40:
                                violation = "STALLED VEHICLE"
                        else:
                            self.stationary_frames[tid] = 0

                # ── Draw Visual Annotations on Frame ──
                # Draw ANPR plate badge above bounding box
                plate_box_w = int(max(90, len(plate) * 8.5))
                px1 = int(cx - plate_box_w / 2)
                py1 = int(max(0, y1 - 22))
                px2 = px1 + plate_box_w
                py2 = py1 + 18

                # Plate background banner (white with black border)
                cv2.rectangle(
                    annotated, (px1, py1), (px2, py2), (240, 240, 240), -1
                )
                cv2.rectangle(
                    annotated, (px1, py1), (px2, py2), (40, 40, 40), 1
                )
                cv2.putText(
                    annotated,
                    plate,
                    (px1 + 4, py2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (10, 10, 10),
                    1,
                    cv2.LINE_AA,
                )

                # Draw violation highlights if triggered
                if violation:
                    # Red flashing bounding box
                    cv2.rectangle(
                        annotated,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 0, 255),
                        2,
                    )
                    # Violation label
                    v_bg_w = int(len(violation) * 8 + 12)
                    cv2.rectangle(
                        annotated,
                        (int(x1), int(y2 + 2)),
                        (int(x1 + v_bg_w), int(y2 + 18)),
                        (0, 0, 220),
                        -1,
                    )
                    cv2.putText(
                        annotated,
                        violation,
                        (int(x1 + 4), int(y2 + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.36,
                        (255, 255, 255),
                        1,
                    )

                    # ── 2. Snapshot Capture on Incident ──
                    v_key = f"{tid}-{violation.split()[0]}"
                    if v_key not in self.captured_incidents:
                        self.captured_incidents.add(v_key)
                        thumb_b64 = self._extract_snapshot(
                            frame, (x1, y1, x2, y2), is_violation=True
                        )
                        inc_data = {
                            "id": f"INC-{tid:03d}",
                            "track_id": tid,
                            "type": violation,
                            "class": cls_name,
                            "speed": speed,
                            "plate": plate,
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "thumb": f"data:image/jpeg;base64,{thumb_b64}",
                        }
                        self.incident_history.insert(0, inc_data)
                        if len(self.incident_history) > 20:
                            self.incident_history.pop()

                if speed > 0:
                    frame_speeds.append(speed)

                detections.append({
                    "track_id": tid,
                    "class": cls_name,
                    "plate": plate,
                    "confidence": round(float(conf), 2),
                    "speed_kmh": speed,
                    "violation": violation,
                })

        active = len(detections)
        if active > self.peak_active:
            self.peak_active = active

        # Overlay global counters
        cv2.putText(
            annotated,
            f"IN: {self.cross_count['in']}  OUT: {self.cross_count['out']}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 230, 255),
            2,
        )

        avg_spd = (
            round(float(np.mean(frame_speeds)), 1) if frame_speeds else 0.0
        )
        max_spd_frame = (
            round(float(max(frame_speeds)), 1) if frame_speeds else 0.0
        )

        # Speed distribution histogram
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
            "speed_limit": self.speed_limit,
            "peak_active": self.peak_active,
            "total_unique": len(self.all_seen_ids),
            "speed_distribution": speed_dist,
            "recent_events": self.crossing_events[-10:],
            "incidents": self.incident_history[:15],  # top 15 incident snapshots
            "frame_num": self.frame_count,
        }
        return annotated, stats
