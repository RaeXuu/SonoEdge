# SonoEdge App Redesign

## Goal

Redesign the iOS app from a developer-oriented debug UI to a polished patient-facing personal heart sound monitor. Core DSP/inference pipeline stays intact. App shell gets rebuilt.

## Primary User

Patients self-monitoring at home with an ESP32 wearable patch (microphone + BLE).

## Core Experience

- Continuous background monitoring via BLE streaming
- Anomaly detection with local push notifications
- Patient reviews alerts and can share with doctor
- Dashboard shows today's status, recent trend, alert log

## Architecture

**Core module (unchanged — validated against Pi reference):**
- `Audio/AudioProcessor.swift` — Butterworth bandpass, segmentation, log-mel spectrogram
- `Inference/InferencePipeline.swift` — dual-stage SQA + Diagnosis streaming inference
- `Inference/TFLiteEngine.swift` — INT8 TFLite model wrapper
- `Storage/RecordStore.swift` — JSONL persistence (extend for trend queries)
- `Debug/DebugPreprocess.swift` — Pi comparison tool

**New app shell:**

```
SonoEdge/
├── App/
│   ├── SonoEdgeApp.swift          # unchanged
│   └── MainTabView.swift          # refactored tab structure
├── UI/
│   ├── Dashboard/
│   │   ├── DashboardView.swift    # main screen
│   │   ├── StatusCard.swift       # today's heart status indicator
│   │   ├── TrendChart.swift       # 24h sparkline
│   │   └── AlertsList.swift       # recent anomaly alerts
│   ├── History/
│   │   └── HistoryView.swift      # readings timeline with filters
│   └── Settings/
│       └── SettingsView.swift     # device, alerts, sharing, about
├── Services/
│   ├── MonitoringService.swift    # continuous BLE + inference orchestrator
│   ├── NotificationService.swift  # UNUserNotificationCenter wrapper
│   └── BackgroundTaskService.swift # BGTaskScheduler support
├── Audio/                          # [UNCHANGED]
├── Inference/                      # [UNCHANGED]
├── Storage/                        # [UNCHANGED — minor additions]
└── Debug/                          # [UNCHANGED]
```

**Files deleted:**
- `SonoEdge/App/ContentView.swift` — replaced by DashboardView + extracted AppViewModel logic goes into MonitoringService

## Data Flow

```
ESP32 Patch → BLE notify → BLEService (inside MonitoringService)
                                │
                                ▼ (int16 bytes)
                        MonitoringService
                        ┌─────────────────┐
                        │ Chunk assembler │
                        │ (20s = 80000B)  │
                        └───────┬─────────┘
                                ▼
                        InferencePipeline.run()
                                │
                                ▼
                        ChunkResult
                           │           │
                           ▼           ▼
                     RecordStore   anomaly check
                     (persist)     (label == "Abnormal")
                                       │
                                       ▼ yes
                               NotificationService
                               (local push notification)
                           │
                           ▼
                   @Published state → SwiftUI views
```

## UI Design

### Tab 1 — Dashboard
- StatusCard: large visual indicator (Normal/Attention/Recording), last check timestamp
- TrendChart: 24h sparkline of P(Normal) values
- AlertsList: last 5 anomaly alerts with timestamps
- Start/Stop monitoring button

### Tab 2 — History
- Timeline grouped by date
- Filter: All / Normal / Abnormal / Noise
- Tap entry for detail

### Tab 3 — Settings
- Device connection status
- Alert preferences (on/off)
- Share with doctor (share sheet)
- App version

## Implementation Steps

1. Delete `ContentView.swift`, remove `AppViewModel`
2. Create `Services/MonitoringService.swift` — wraps BLERecorder + InferencePipeline into continuous loop, exposes @Published state
3. Create `Services/NotificationService.swift` — local notification for anomalies
4. Build `UI/Dashboard/DashboardView.swift` + StatusCard + TrendChart + AlertsList
5. Refactor `UI/History/HistoryView.swift` — timeline with filter chips
6. Build `UI/Settings/SettingsView.swift`
7. Update `MainTabView.swift` to wire new views
8. Add background task support via `BackgroundTaskService`
