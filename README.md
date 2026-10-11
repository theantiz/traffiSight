# TraffiSight

Real-time traffic analysis system using YOLOv8 and ByteTrack. Processes video feeds to detect, classify, and track vehicles — then streams analytics to a web dashboard over WebSocket.

---

## What It Does

- **Detects and tracks vehicles** across frames with persistent IDs (handles occlusions)
- **Classifies** into car, truck, bus, motorcycle with per-class counts
- **Estimates speed** from centroid displacement across a sliding window
- **Counts directional flow** — vehicles crossing a virtual line are tallied as IN or OUT
- **Flags congestion** based on active vehicle density (free / moderate / heavy)
- **Generates heatmaps** showing where vehicles accumulate over time
- **Logs everything** to CSV for post-analysis

---

## Architecture

```
Video Source (.mp4 / upload)
        │
        ▼
┌──────────────────────┐
│  YOLOv8 + ByteTrack  │──── detect, classify, assign track IDs
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Analytics Pipeline  │──── speed estimation, line crossing,
│                      │     congestion level, heatmap accumulation
└──────────┬───────────┘
           │
     ┌─────┼──────────┐
     ▼     ▼          ▼
 WebSocket  REST API   CSV Log
 (frames +  /api/*     data/logs/
  stats)
     │
     ▼
 Browser Dashboard
 (Tailwind + Chart.js)
```

---

## Features

### Detection & Tracking

| | |
|---|---|
| Model | YOLOv8n (COCO pre-trained, single-stage anchor-free) |
| Tracker | ByteTrack — Kalman filter + IoU matching on both high and low confidence detections |
| Filtered classes | car, truck, bus, motorcycle (COCO IDs 2, 3, 5, 7) |

### Speed Estimation

Centroid positions are buffered over the last 10 frames. Speed is computed as:

$$v = \frac{\sqrt{(\Delta x)^2 + (\Delta y)^2}}{\Delta t} \times k$$

where $k$ is a pixel-to-meter conversion factor (default `0.05 m/px`, tunable per camera).

### Line Crossing

A horizontal counting line sits at 60% of frame height. When a track's centroid crosses it:
- downward → OUT
- upward → IN

Each track ID fires at most once.

### Congestion

| Active vehicles | Level |
|---|---|
| 0–8 | Free |
| 9–15 | Moderate |
| 16+ | Heavy |

### Heatmap

An accumulator the size of the frame is incremented at each detection centroid (circular kernel, radius 20px). Normalized and rendered as a JET colormap overlay, togglable from the dashboard.

---

## Dashboard

The frontend connects via WebSocket and renders:

- Annotated video feed (bounding boxes, track IDs, speed labels, counting line)
- Vehicle count over time (line chart, rolling 60 points)
- Vehicle type distribution (doughnut chart)
- Stats cards — active count, per-class breakdown, IN/OUT crossings, average speed
- Congestion indicator
- Heatmap toggle
- Video upload
- CSV download

Built with Tailwind CSS (CDN) and Chart.js. No build step.

---

## API

| Method | Endpoint | Returns |
|---|---|---|
| `GET` | `/api/stats` | Latest stats snapshot |
| `GET` | `/api/history?limit=100` | Last N snapshots |
| `GET` | `/api/download-log` | CSV file download |
| `POST` | `/api/upload-video` | Upload a new video source |

---

## CSV Log Schema

Logs are written per-frame to `data/logs/traffic_log_<timestamp>.csv`.

| Column | Type | Description |
|---|---|---|
| timestamp | ISO 8601 | Processing time |
| total_active | int | Vehicles in frame |
| cars | int | Car count |
| trucks | int | Truck count |
| buses | int | Bus count |
| motorcycles | int | Motorcycle count |
| avg_speed | float | Mean speed (km/h) |
| crossings_in | int | Cumulative IN |
| crossings_out | int | Cumulative OUT |
| congestion | string | free / moderate / heavy |

---

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── detector.py      # YOLOv8 + ByteTrack engine, speed/counting/heatmap logic
│   └── main.py           # FastAPI server, WebSocket stream, REST API, CSV logging
├── data/
│   ├── traffic.mp4        # Input video
│   └── logs/              # Auto-generated CSV logs
├── models/                # Optional custom YOLO weights
├── index.html             # Dashboard (Tailwind + Chart.js)
└── requirements.txt
```

---

## Tech Stack

| | |
|---|---|
| Detection | YOLOv8 (Ultralytics) |
| Tracking | ByteTrack |
| Vision | OpenCV |
| Backend | FastAPI, Uvicorn |
| Frontend | Tailwind CSS, Chart.js |
| Data | NumPy, CSV |

---

## Future Work

- License plate recognition (OCR/ANPR)
- Multi-camera feed stitching
- Incident detection (stopped vehicles, wrong-way driving)
- Time-series database (InfluxDB / PostgreSQL) instead of flat CSV
- Cloud deployment with RTSP ingestion
