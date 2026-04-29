import Foundation
import Combine
import SwiftUI

/// Central orchestrator: owns BLE → Inference pipeline, exposes @Published state for UI
@MainActor
final class MonitoringService: ObservableObject {

    // MARK: - Published state

    @Published var connectionStatus = "未连接"
    @Published var isConnected = false
    @Published var isMonitoring = false
    @Published var isProcessingChunk = false

    @Published var currentStatusText = "就绪"
    @Published var lastLabel: String? = nil
    @Published var lastProbNormal: Float? = nil
    @Published var lastInferenceMs: Double? = nil

    @Published var totalNormal = 0
    @Published var totalAbnormal = 0
    @Published var totalNoise = 0

    @Published var currentWindow: Int = 0
    @Published var totalWindows: Int = 19
    @Published var runningAvgNormal: Float? = nil
    @Published var currentWindowDetails: [WindowResult] = []

    // Trend data for 24h sparkline
    @Published var recentReadings: [ReadingPoint] = []

    // MARK: - Internal

    private var bleRecorder: BLERecorder?
    private var pipeline: InferencePipeline?
    private let modelDir = Bundle.main.resourcePath!
    private var alerts: [AlertItem] = []

    // MARK: - Setup

    func setup() throws {
        let sqaPath  = modelDir + "/heart_quality_int8full.tflite"
        let diagPath = modelDir + "/heart_model_int8full.tflite"

        let sqa  = try TFLiteEngine(modelPath: sqaPath)
        let diag = try TFLiteEngine(modelPath: diagPath)

        try sqa.warmup(count: 5)
        try diag.warmup(count: 5)

        pipeline = InferencePipeline(sqaEngine: sqa, diagEngine: diag)

        bleRecorder = BLERecorder()
        bleRecorder?.onChunkReady = { [weak self] rawChunk in
            Task { [weak self] in
                await self?.processChunk(rawChunk)
            }
        }

        bleRecorder?.$connectionStatus.assign(to: &$connectionStatus)
        bleRecorder?.$isRunning.assign(to: &$isConnected)
    }

    // MARK: - Connect / Disconnect

    func connect() {
        bleRecorder?.startScan()
    }

    func disconnect() {
        bleRecorder?.disconnect()
        isConnected = false
        isMonitoring = false
        currentStatusText = "已断开"
        isProcessingChunk = false
    }

    // MARK: - Chunk processing

    private func processChunk(_ raw: Data) async {
        guard let pipeline = pipeline else { return }

        isProcessingChunk = true
        currentWindowDetails = []
        currentWindow = 0
        totalWindows = 19

        do {
            let result = try await pipeline.run(on: raw) {
                [weak self] runningAvg, win, total, latest in
                Task { @MainActor [weak self] in
                    guard let self = self else { return }
                    self.runningAvgNormal = runningAvg
                    self.currentWindow = win
                    self.totalWindows = total
                    self.currentWindowDetails.append(latest)

                    if let avg = runningAvg {
                        let label = avg > 0.5 ? "Normal" : "Abnormal"
                        self.currentStatusText = "W\(win)/\(total) | \(label) \(String(format: "%.1f", avg * 100))%"
                    } else {
                        self.currentStatusText = "W\(win)/\(total) | 等待有效窗口..."
                    }
                }
            }

            self.lastInferenceMs = result.inferenceMs

            if let label = result.label {
                self.lastLabel = label
                self.lastProbNormal = result.avgProbNormal
                if label == "Normal" { self.totalNormal += 1 }
                else { self.totalAbnormal += 1 }
                self.currentStatusText = "\(label) (\(String(format: "%.1f", (result.avgProbNormal ?? 0) * 100))%)"

                // Add to recent readings for trend chart
                let point = ReadingPoint(
                    timestamp: Date(),
                    probNormal: result.avgProbNormal ?? 0,
                    label: label
                )
                self.recentReadings.append(point)
                // Keep last 24h
                let cutoff = Date().addingTimeInterval(-86400)
                self.recentReadings = self.recentReadings.filter { $0.timestamp > cutoff }

                // Anomaly alert
                if label == "Abnormal" {
                    let alert = AlertItem(
                        timestamp: Date(),
                        label: label,
                        probNormal: result.avgProbNormal ?? 0
                    )
                    self.alerts.insert(alert, at: 0)
                    await NotificationService.sendAnomalyAlert(probNormal: result.avgProbNormal ?? 0)
                }
            } else {
                self.lastLabel = nil
                self.lastProbNormal = nil
                self.totalNoise += 1
                self.currentStatusText = "低质量 (\(result.validWindows)/\(result.totalWindows) 窗口)"
            }

            RecordStore.appendSummary(label: result.label,
                                      probNormal: result.avgProbNormal,
                                      validSegs: result.validWindows,
                                      totalSegs: result.totalWindows)

            self.isProcessingChunk = false
        } catch {
            self.currentStatusText = "推理错误: \(error.localizedDescription)"
            self.isProcessingChunk = false
        }

        bleRecorder?.markChunkConsumed()
    }

    // MARK: - Alerts access

    func recentAlerts(limit: Int = 5) -> [AlertItem] {
        Array(alerts.prefix(limit))
    }

    // MARK: - Reset

    func resetTally() {
        totalNormal   = 0
        totalAbnormal = 0
        totalNoise    = 0
        lastLabel     = nil
        lastProbNormal = nil
        lastInferenceMs = nil
        currentWindowDetails = []
        currentWindow = 0
        runningAvgNormal = nil
    }
}

// MARK: - Supporting types

struct ReadingPoint: Identifiable {
    let id = UUID()
    let timestamp: Date
    let probNormal: Float
    let label: String
}

struct AlertItem: Identifiable {
    let id = UUID()
    let timestamp: Date
    let label: String
    let probNormal: Float
}
