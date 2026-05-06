import Foundation
import Combine
import SwiftUI

/// Central orchestrator: owns BLE → Inference pipeline, exposes @Published state for UI
@MainActor
final class MonitoringService: ObservableObject {

    // MARK: - Published state

    @Published var connectionStatus = "Disconnected"
    @Published var isConnected = false
    @Published var isMonitoring = false
    @Published var isProcessingChunk = false

    @Published var currentStatusText = "Ready"
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
    private var isSetup = false

    // MARK: - Setup

    func setup() throws {
        guard !isSetup else { return }
        isSetup = true
        let sqaPath  = modelDir + "/heart_quality_int8full.tflite"
        let diagPath = modelDir + "/heart_model_int8full.tflite"

        let sqa  = try TFLiteEngine(modelPath: sqaPath)
        let diag = try TFLiteEngine(modelPath: diagPath)

        try sqa.warmup(count: 5)
        try diag.warmup(count: 5)

        #if DEBUG
        try sqa.benchmark(iterations: 100)
        try diag.benchmark(iterations: 100)
        try sqa.benchmarkSingleThread(iterations: 100)
        try diag.benchmarkSingleThread(iterations: 100)
        #endif

        pipeline = InferencePipeline(sqaEngine: sqa, diagEngine: diag)

        // #if DEBUG — Energy Log commented out: was blocking main actor for 60s on every launch.
        // Uncomment only when profiling inference performance without a BLE device.
        // Task { [weak self] in ... }

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
        currentStatusText = "Disconnected"
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
                        self.currentStatusText = "W\(win)/\(total) | Waiting for valid window..."
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
                self.currentStatusText = "Low quality (\(result.validWindows)/\(result.totalWindows) windows)"
            }

            RecordStore.appendSummary(label: result.label,
                                      probNormal: result.avgProbNormal,
                                      validSegs: result.validWindows,
                                      totalSegs: result.totalWindows)

            self.isProcessingChunk = false
        } catch {
            self.currentStatusText = "Inference error: \(error.localizedDescription)"
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
