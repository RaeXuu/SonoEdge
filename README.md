# 🫀 SonoEdge

> On-device heart sound AI — real-time inference on iPhone via BLE stethoscope.

SonoEdge runs a two-stage INT8 TFLite pipeline on iPhone to classify heart sounds (Normal / Abnormal) streamed from an ESP32 BLE stethoscope. No cloud, no latency, fully on-device.

## 🧠 How It Works

```
ESP32 ──BLE──▶ iPhone ──DSP──▶ SQA Model ──▶ Diagnosis Model ──▶ UI / Notification
```

1. **BLE stream** — int16 PCM @ 2000 Hz from ESP32 stethoscope
2. **DSP** — bandpass filter (25–400 Hz) → 2s sliding windows → 64×64 Mel spectrograms
3. **SQA model** — gates signal quality (P ≥ 0.05 passes)
4. **Diagnosis model** — classifies Normal / Abnormal
5. **Fusion** — SQA-weighted average → final confidence

## ⚡ Performance (iPhone 16 Pro vs Pi 4B)

| Platform | Per Window | Speedup |
|----------|-----------|---------|
| Raspberry Pi 4B (Cortex-A72) | 17.4 ms | 1× |
| iPhone 16 Pro (A18 Pro) | **1.36 ms** | **~13×** |

Models: 144 KB each (INT8), load in <18 ms. Energy impact negligible (nominal thermal state).

## 📦 Requirements

- iOS 16.0+ (physical device — BLE not available in simulator)
- Xcode 15+, CocoaPods
- ESP32 BLE stethoscope (advertises as `ESP32_Steth`)

## 🚀 Quick Start

```bash
cd SonoEdge
pod install
open SonoEdge.xcworkspace   # ⌘R on a real iPhone
```

## 📱 App Overview

| Tab | What It Does |
|-----|-------------|
| 🏠 **Monitor** | Real-time status card, confidence, 24h trend chart, recent alerts |
| 📋 **History** | All past records, filterable by Normal / Abnormal / Noise |
| ⚙️ **Settings** | Push notification toggle, share report with doctor |

Abnormal detections trigger **local push notifications** immediately.

## 🗂️ Project Layout

```
SonoEdge/
├── Audio/         # BLE recorder + Butterworth DSP (Accelerate/vDSP)
├── Inference/     # TFLite engine + two-stage pipeline
├── Services/      # Monitoring orchestrator, notifications, background tasks
├── UI/            # SwiftUI: Dashboard, History, Settings
├── Storage/       # JSONL record persistence
├── Models/        # INT8 + FP32 TFLite models
└── Debug/         # WAV preprocessing debug tool
```
