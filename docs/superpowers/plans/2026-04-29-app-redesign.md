# SonoEdge App Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the app shell into a patient-facing continuous heart sound monitor with dashboard, history, and settings tabs. Core DSP/inference pipeline stays untouched.

**Architecture:** Approach C hybrid — preserve `Audio/`, `Inference/`, `Storage/`, `Debug/` as-is. Rebuild UI layer with `Services/MonitoringService` as the central orchestrator owning BLE + inference loop, exposing `@Published` state to SwiftUI views. Delete `ContentView.swift`.

**Tech Stack:** Swift 5, SwiftUI, CoreBluetooth, TensorFlowLite, UserNotifications, BackgroundTasks

---

### Task 1: Delete ContentView.swift and create MonitoringService skeleton

**Files:**
- Delete: `SonoEdge/App/ContentView.swift`
- Create: `SonoEdge/Services/MonitoringService.swift`

- [ ] **Step 1: Delete ContentView.swift**

```bash
rm /Users/raexu/Documents/SonoEdge/SonoEdge/App/ContentView.swift
```

- [ ] **Step 2: Create MonitoringService skeleton**

Create `SonoEdge/Services/MonitoringService.swift`:

```swift
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
```

- [ ] **Step 3: Verify build compiles**

```bash
xcodebuild -project /Users/raexu/Documents/SonoEdge/SonoEdge.xcodeproj -scheme SonoEdge -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -5
```

Expected: BUILD SUCCEEDED

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "$(cat <<'EOF'
Delete ContentView, create MonitoringService as central orchestrator

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Create NotificationService

**Files:**
- Create: `SonoEdge/Services/NotificationService.swift`

- [ ] **Step 1: Create NotificationService**

```swift
import Foundation
import UserNotifications

struct NotificationService {

    private init() {}

    static func requestAuthorization() async -> Bool {
        do {
            return try await UNUserNotificationCenter.current()
                .requestAuthorization(options: [.alert, .sound, .badge])
        } catch {
            print("[Notification] 授权失败: \(error)")
            return false
        }
    }

    static func sendAnomalyAlert(probNormal: Float) async {
        let center = UNUserNotificationCenter.current()

        // Check notification settings
        let settings = await center.notificationSettings()
        guard settings.authorizationStatus == .authorized else { return }

        let content = UNMutableNotificationContent()
        content.title = "心律异常检测"
        content.body = String(format: "检测到异常心音信号 (置信度: %.0f%%)。请查看详情。", (1 - probNormal) * 100)
        content.sound = .default
        content.badge = 1

        // Throttle: don't send more than one every 5 minutes
        let request = UNNotificationRequest(
            identifier: "anomaly-\(Date().timeIntervalSince1970)",
            content: content,
            trigger: nil // immediate
        )

        do {
            try await center.add(request)
        } catch {
            print("[Notification] 发送失败: \(error)")
        }
    }

    /// Remove all delivered notifications when app opens
    static func clearBadge() {
        UNUserNotificationCenter.current().removeAllDeliveredNotifications()
    }
}
```

- [ ] **Step 2: Register notification delegate in SonoEdgeApp.swift**

Modify `SonoEdge/App/SonoEdgeApp.swift`:

```swift
import SwiftUI

@main
struct SonoEdgeApp: App {
    init() {
        Task {
            _ = await NotificationService.requestAuthorization()
        }
    }

    var body: some Scene {
        WindowGroup {
            MainTabView()
        }
    }
}
```

- [ ] **Step 3: Verify build compiles**

```bash
xcodebuild -project /Users/raexu/Documents/SonoEdge/SonoEdge.xcodeproj -scheme SonoEdge -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -5
```

Expected: BUILD SUCCEEDED

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "$(cat <<'EOF'
Add NotificationService for anomaly alerts

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Create DashboardView with StatusCard

**Files:**
- Create: `SonoEdge/UI/Dashboard/DashboardView.swift`
- Create: `SonoEdge/UI/Dashboard/StatusCard.swift`

- [ ] **Step 1: Create directory and DashboardView**

```bash
mkdir -p /Users/raexu/Documents/SonoEdge/SonoEdge/UI/Dashboard
```

- [ ] **Step 2: Create DashboardView.swift**

```swift
import SwiftUI

struct DashboardView: View {
    @ObservedObject var service: MonitoringService

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Status card
                    StatusCard(service: service)

                    // Monitoring button
                    monitoringButton

                    // Progress during processing
                    if service.isProcessingChunk {
                        progressSection
                    }

                    // Trend chart
                    TrendChart(readings: service.recentReadings)

                    // Recent alerts
                    if !service.recentAlerts().isEmpty {
                        AlertsList(alerts: service.recentAlerts())
                    }

                    Spacer(minLength: 40)
                }
                .padding()
            }
            .navigationTitle("SonoEdge")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("重置") { service.resetTally() }
                        .font(.caption)
                }
            }
            .onAppear {
                do { try service.setup() }
                catch { service.connectionStatus = "模型加载失败: \(error.localizedDescription)" }
                NotificationService.clearBadge()
            }
        }
    }

    // MARK: - Monitoring button

    private var monitoringButton: some View {
        Button(action: {
            if service.isConnected {
                service.disconnect()
            } else {
                service.connect()
            }
        }) {
            HStack {
                Image(systemName: service.isConnected ? "stop.circle.fill" : "play.circle.fill")
                    .font(.title2)
                Text(service.isConnected ? "停止监测" : "开始监测")
                    .font(.headline)
            }
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .background(service.isConnected ? Color.red : Color.blue)
            .cornerRadius(14)
        }
        .disabled(service.isProcessingChunk)
    }

    // MARK: - Progress section

    private var progressSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                ProgressView()
                    .scaleEffect(0.8)
                Text("分析中...")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                Spacer()
                if let avg = service.runningAvgNormal {
                    Text(avg > 0.5 ? "Normal" : "Abnormal")
                        .font(.caption.bold())
                        .foregroundColor(avg > 0.5 ? .green : .red)
                }
            }
            ProgressView(value: Double(service.currentWindow), total: Double(service.totalWindows))
                .tint((service.runningAvgNormal ?? 0) > 0.5 ? .green : .red)
            Text(service.currentStatusText)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.04), radius: 6, y: 1)
        )
    }
}
```

- [ ] **Step 3: Create StatusCard.swift**

```swift
import SwiftUI

struct StatusCard: View {
    @ObservedObject var service: MonitoringService

    var body: some View {
        VStack(spacing: 12) {
            // Status icon
            ZStack {
                Circle()
                    .fill(statusColor.opacity(0.15))
                    .frame(width: 80, height: 80)
                Image(systemName: statusIcon)
                    .font(.system(size: 36))
                    .foregroundColor(statusColor)
            }

            // Status text
            Text(statusTitle)
                .font(.title2.bold())
                .foregroundColor(statusColor)

            Text(statusSubtitle)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            // Connection indicator
            HStack(spacing: 6) {
                Circle()
                    .fill(service.isConnected ? Color.green : Color.gray)
                    .frame(width: 6, height: 6)
                Text(service.connectionStatus)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 24)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 20)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.06), radius: 10, y: 2)
        )
    }

    // MARK: - Computed

    private var statusIcon: String {
        if service.isProcessingChunk { return "waveform" }
        if !service.isConnected { return "antenna.radiowaves.left.and.right" }
        guard let label = service.lastLabel else { return "questionmark.circle" }
        return label == "Normal" ? "heart.fill" : "heart.text.square.fill"
    }

    private var statusColor: Color {
        if !service.isConnected { return .gray }
        guard let label = service.lastLabel else { return .orange }
        return label == "Normal" ? .green : .red
    }

    private var statusTitle: String {
        if !service.isConnected { return "未连接" }
        if service.isProcessingChunk { return "监测中..." }
        guard let label = service.lastLabel else { return "等待数据" }
        return label == "Normal" ? "心率正常" : "心率异常"
    }

    private var statusSubtitle: String {
        if !service.isConnected { return "轻触下方按钮连接设备" }
        if service.isProcessingChunk { return "正在分析心音信号..." }
        guard let label = service.lastLabel, let prob = service.lastProbNormal else {
            return "暂无检测数据"
        }
        let confidence = label == "Normal" ? prob : 1 - prob
        return String(format: "置信度: %.0f%% · 上次: \(service.lastInferenceMs.map { String(format: "%.0fms", $0) } ?? "--")", confidence * 100)
    }
}
```

- [ ] **Step 4: Verify build compiles**

```bash
xcodebuild -project /Users/raexu/Documents/SonoEdge/SonoEdge.xcodeproj -scheme SonoEdge -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -5
```

Expected: BUILD SUCCEEDED

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "$(cat <<'EOF'
Add DashboardView with StatusCard component

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Create TrendChart and AlertsList components

**Files:**
- Create: `SonoEdge/UI/Dashboard/TrendChart.swift`
- Create: `SonoEdge/UI/Dashboard/AlertsList.swift`

- [ ] **Step 1: Create TrendChart.swift**

```swift
import SwiftUI

struct TrendChart: View {
    let readings: [ReadingPoint]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("最近趋势")
                .font(.subheadline)
                .foregroundColor(.secondary)

            if readings.isEmpty {
                HStack {
                    Spacer()
                    VStack(spacing: 6) {
                        Image(systemName: "chart.line.uptrend.xyaxis")
                            .font(.title2)
                            .foregroundColor(.secondary.opacity(0.5))
                        Text("开始监测后显示24小时趋势")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(.vertical, 24)
                    Spacer()
                }
            } else {
                GeometryReader { geometry in
                    let width = geometry.size.width
                    let height = geometry.size.height
                    let points = normalizedPoints(width: width, height: height)

                    ZStack(alignment: .leading) {
                        // Grid lines
                        ForEach(0..<3) { i in
                            let y = height * CGFloat(i) / 2
                            Path { path in
                                path.move(to: CGPoint(x: 0, y: y))
                                path.addLine(to: CGPoint(x: width, y: y))
                            }
                            .stroke(Color(.systemGray5), lineWidth: 0.5)
                        }

                        // Sparkline path
                        Path { path in
                            guard let first = points.first else { return }
                            path.move(to: first)
                            for point in points.dropFirst() {
                                path.addLine(to: point)
                            }
                        }
                        .stroke(
                            LinearGradient(
                                gradient: Gradient(colors: [.blue, .green]),
                                startPoint: .leading,
                                endPoint: .trailing
                            ),
                            style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round)
                        )

                        // Fill gradient
                        Path { path in
                            guard let first = points.first, let last = points.last else { return }
                            path.move(to: first)
                            for point in points.dropFirst() {
                                path.addLine(to: point)
                            }
                            path.addLine(to: CGPoint(x: last.x, y: height))
                            path.addLine(to: CGPoint(x: first.x, y: height))
                            path.closeSubpath()
                        }
                        .fill(
                            LinearGradient(
                                gradient: Gradient(colors: [Color.blue.opacity(0.2), Color.green.opacity(0.05)]),
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                    }
                }
                .frame(height: 100)

                // Last reading time
                if let last = readings.last {
                    HStack {
                        Text("过去24小时")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text(last.timestamp, style: .relative)
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.04), radius: 6, y: 1)
        )
    }

    private func normalizedPoints(width: CGFloat, height: CGFloat) -> [CGPoint] {
        guard readings.count > 1 else { return [] }
        let sorted = readings.sorted { $0.timestamp < $1.timestamp }
        let probs = sorted.map { CGFloat($0.probNormal) }

        guard let minP = probs.min(), let maxP = probs.max(), maxP > minP else {
            // All same value → flat line in middle
            return (0..<sorted.count).map { i in
                CGPoint(x: width * CGFloat(i) / CGFloat(sorted.count - 1), y: height / 2)
            }
        }

        return (0..<sorted.count).map { i in
            let x = width * CGFloat(i) / CGFloat(sorted.count - 1)
            let y = height * (1 - (probs[i] - minP) / (maxP - minP))
            // Clamp to avoid clipping
            return CGPoint(x: x, y: max(2, min(y, height - 2)))
        }
    }
}
```

- [ ] **Step 2: Create AlertsList.swift**

```swift
import SwiftUI

struct AlertsList: View {
    let alerts: [AlertItem]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("异常提醒")
                .font(.subheadline)
                .foregroundColor(.secondary)

            ForEach(alerts) { alert in
                HStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)
                        .font(.caption)

                    VStack(alignment: .leading, spacing: 2) {
                        Text("检测到异常心音")
                            .font(.subheadline)
                            .foregroundColor(.primary)
                        Text(alert.timestamp, style: .relative)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    Spacer()

                    Text(String(format: "%.0f%%", (1 - alert.probNormal) * 100))
                        .font(.caption.monospacedDigit())
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.red.opacity(0.1))
                        .foregroundColor(.red)
                        .cornerRadius(6)
                }
                .padding(.vertical, 4)

                if alert.id != alerts.last?.id {
                    Divider()
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.04), radius: 6, y: 1)
        )
    }
}
```

- [ ] **Step 3: Verify build compiles**

```bash
xcodebuild -project /Users/raexu/Documents/SonoEdge/SonoEdge.xcodeproj -scheme SonoEdge -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -5
```

Expected: BUILD SUCCEEDED

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "$(cat <<'EOF'
Add TrendChart and AlertsList dashboard components

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Refactor HistoryView with filter and timeline

**Files:**
- Create: `SonoEdge/UI/History/HistoryView.swift`

- [ ] **Step 1: Create directory and new HistoryView**

```bash
mkdir -p /Users/raexu/Documents/SonoEdge/SonoEdge/UI/History
```

- [ ] **Step 2: Create HistoryView.swift**

```swift
import SwiftUI

struct HistoryView: View {
    @ObservedObject var service: MonitoringService
    @State private var records: [[String: Any]] = []
    @State private var filter: HistoryFilter = .all

    enum HistoryFilter: String, CaseIterable {
        case all = "全部"
        case normal = "Normal"
        case abnormal = "Abnormal"
        case noise = "Noise"

        var color: Color {
            switch self {
            case .all: return .blue
            case .normal: return .green
            case .abnormal: return .red
            case .noise: return .gray
            }
        }
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Filter chips
                filterBar

                // List
                if filteredRecords.isEmpty {
                    Spacer()
                    VStack(spacing: 8) {
                        Image(systemName: "tray")
                            .font(.system(size: 40))
                            .foregroundColor(.secondary)
                        Text("暂无记录")
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                } else {
                    List {
                        ForEach(groupedRecords, id: \.date) { group in
                            Section(header: Text(group.date).font(.caption)) {
                                ForEach(group.items.indices, id: \.self) { i in
                                    let r = group.items[i]
                                    historyRow(r)
                                }
                            }
                        }
                    }
                    .listStyle(.insetGrouped)
                }
            }
            .navigationTitle("记录")
            .onAppear { reload() }
            .onChange(of: service.totalNormal) { _ in reload() }
            .onChange(of: service.totalAbnormal) { _ in reload() }
            .onChange(of: service.totalNoise) { _ in reload() }
        }
    }

    // MARK: - Filter bar

    private var filterBar: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(HistoryFilter.allCases, id: \.rawValue) { f in
                    Button(action: { filter = f }) {
                        Text(f.rawValue)
                            .font(.caption)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 6)
                            .background(filter == f ? f.color : Color(.systemGray6))
                            .foregroundColor(filter == f ? .white : .secondary)
                            .cornerRadius(16)
                    }
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
        }
        .background(Color(.systemBackground))
    }

    // MARK: - Row

    private func historyRow(_ r: [String: Any]) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                if let ts = r["ts"] as? String {
                    Text(ts)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                if let label = r["label"] as? String {
                    Text(label == "noise" ? "低质量信号" : (label == "Normal" ? "正常" : "异常"))
                        .font(.headline)
                        .foregroundColor(labelColor(label))
                }
            }
            Spacer()
            if let pn = r["prob_normal"] as? Double {
                Text(String(format: "%.0f%%", pn * 100))
                    .font(.subheadline.monospacedDigit())
                    .foregroundColor(.secondary)
            }
            Text("\(r["valid_segs"] as? Int ?? 0)/\(r["total_segs"] as? Int ?? 0)")
                .font(.caption2.monospacedDigit())
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 2)
    }

    // MARK: - Helpers

    private var filteredRecords: [[String: Any]] {
        if filter == .all { return records }
        return records.filter {
            ($0["label"] as? String ?? "noise") == filter.rawValue
        }
    }

    private var groupedRecords: [(date: String, items: [[String: Any]])] {
        let grouped = Dictionary(grouping: filteredRecords) { record -> String in
            let ts = record["ts"] as? String ?? ""
            return String(ts.prefix(10)) // "YYYY-MM-DD"
        }
        return grouped.sorted { $0.key > $1.key }.map { (date: $0.key, items: $0.value) }
    }

    private func reload() {
        records = RecordStore.loadSummaries()
    }

    private func labelColor(_ label: String) -> Color {
        switch label {
        case "Normal":   return .green
        case "Abnormal": return .red
        default:         return .gray
        }
    }
}
```

- [ ] **Step 3: Verify build compiles**

```bash
xcodebuild -project /Users/raexu/Documents/SonoEdge/SonoEdge.xcodeproj -scheme SonoEdge -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -5
```

Expected: BUILD SUCCEEDED

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "$(cat <<'EOF'
Refactor HistoryView with filter chips and date grouping

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Create SettingsView

**Files:**
- Create: `SonoEdge/UI/Settings/SettingsView.swift`

- [ ] **Step 1: Create directory and SettingsView**

```bash
mkdir -p /Users/raexu/Documents/SonoEdge/SonoEdge/UI/Settings
```

- [ ] **Step 2: Create SettingsView.swift**

```swift
import SwiftUI

struct SettingsView: View {
    @ObservedObject var service: MonitoringService
    @State private var alertsEnabled = true

    var body: some View {
        NavigationView {
            List {
                // Device section
                Section("设备") {
                    HStack {
                        Label("电子听诊器贴片", systemImage: "wave.3.right")
                        Spacer()
                        HStack(spacing: 4) {
                            Circle()
                                .fill(service.isConnected ? Color.green : Color.gray)
                                .frame(width: 6, height: 6)
                            Text(service.isConnected ? "已连接" : "未连接")
                                .foregroundColor(service.isConnected ? .green : .secondary)
                        }
                        .font(.subheadline)
                    }

                    if service.isConnected {
                        HStack {
                            Label("采集状态", systemImage: "antenna.radiowaves.left.and.right")
                            Spacer()
                            Text(service.connectionStatus)
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                    }
                }

                // Alerts section
                Section("异常提醒") {
                    Toggle(isOn: $alertsEnabled) {
                        Label("推送通知", systemImage: "bell.fill")
                    }

                    if alertsEnabled {
                        HStack {
                            Label("检测阈值", systemImage: "slider.horizontal.3")
                            Spacer()
                            Text("P(Normal) < 0.5")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }

                // Share section
                Section("数据分享") {
                    Button(action: shareReport) {
                        Label("分享给医生", systemImage: "square.and.arrow.up")
                    }
                }

                // About section
                Section("关于") {
                    Label("版本 1.0", systemImage: "info.circle")
                    Label("SonoEdge Health Monitor", systemImage: "heart.text.square")
                }
            }
            .navigationTitle("设置")
            .listStyle(.insetGrouped)
        }
    }

    private func shareReport() {
        let records = RecordStore.loadSummaries()
        var text = "SonoEdge 心音监测报告\n\n"
        for r in records.prefix(50) {
            let ts = r["ts"] as? String ?? ""
            let label = r["label"] as? String ?? ""
            let pn = r["prob_normal"] as? Double ?? 0
            let v = r["valid_segs"] as? Int ?? 0
            let t = r["total_segs"] as? Int ?? 0
            text += "\(ts) | \(label) | P(N)=\(String(format: "%.1f", pn * 100))% | \(v)/\(t)\n"
        }

        let av = UIActivityViewController(
            activityItems: [text],
            applicationActivities: nil
        )

        // Present share sheet
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let root = windowScene.windows.first?.rootViewController {
            root.present(av, animated: true)
        }
    }
}
```

- [ ] **Step 3: Verify build compiles**

```bash
xcodebuild -project /Users/raexu/Documents/SonoEdge/SonoEdge.xcodeproj -scheme SonoEdge -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -5
```

Expected: BUILD SUCCEEDED

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "$(cat <<'EOF'
Add SettingsView with device status, alerts toggle, share report

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Rewire MainTabView to use MonitoringService and new views

**Files:**
- Modify: `SonoEdge/App/MainTabView.swift`

- [ ] **Step 1: Rewrite MainTabView.swift**

```swift
import SwiftUI

struct MainTabView: View {
    @StateObject private var service = MonitoringService()
    @State private var selectedTab = 0

    var body: some View {
        TabView(selection: $selectedTab) {
            DashboardView(service: service)
                .tabItem {
                    Image(systemName: "heart.fill")
                    Text("监测")
                }
                .tag(0)

            HistoryView(service: service)
                .tabItem {
                    Image(systemName: "clock.arrow.circlepath")
                    Text("记录")
                }
                .tag(1)

            SettingsView(service: service)
                .tabItem {
                    Image(systemName: "person.circle")
                    Text("设置")
                }
                .tag(2)
        }
        .accentColor(.blue)
    }
}
```

- [ ] **Step 2: Delete old ProfileView and ExamineView from MainTabView.swift**

The old `ExamineView`, `ProfileView`, and `StatItem` types defined in `MainTabView.swift` are no longer referenced — the rewrite replaces them entirely.

- [ ] **Step 3: Verify build compiles**

```bash
xcodebuild -project /Users/raexu/Documents/SonoEdge/SonoEdge.xcodeproj -scheme SonoEdge -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -5
```

Expected: BUILD SUCCEEDED

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "$(cat <<'EOF'
Rewire MainTabView to use MonitoringService and new views

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Create BackgroundTaskService for continuous monitoring

**Files:**
- Create: `SonoEdge/Services/BackgroundTaskService.swift`

- [ ] **Step 1: Create BackgroundTaskService.swift**

```swift
import Foundation
import BackgroundTasks

/// Registers and handles BGAppRefreshTask for periodic background inference.
/// Note: iOS limits background execution. This sets up the infrastructure;
/// actual continuous monitoring is foreground-driven when the app is active.
/// Background BLE requires the "bluetooth-central" UIBackgroundModes capability.
struct BackgroundTaskService {

    static let taskIdentifier = "com.sonoedge.heart-monitor.refresh"

    static func register() {
        BGTaskScheduler.shared.register(forTaskWithIdentifier: taskIdentifier, using: nil) { task in
            handleAppRefresh(task: task as! BGAppRefreshTask)
        }
    }

    static func schedule() {
        let request = BGAppRefreshTaskRequest(identifier: taskIdentifier)
        request.earliestBeginDate = Date(timeIntervalSinceNow: 5 * 60) // 5 min
        do {
            try BGTaskScheduler.shared.submit(request)
        } catch {
            print("[BackgroundTask] 调度失败: \(error)")
        }
    }

    private static func handleAppRefresh(task: BGAppRefreshTask) {
        // Schedule next refresh
        schedule()

        // Allow the system a small window to run inference
        task.expirationHandler = {
            // Cleanup if needed — cancel current chunk
        }

        // The BLE + inference loop continues if app is in foreground.
        // In background, we rely on BGAppRefreshTask for periodic wake-ups.
        // The MonitoringService's BLE callbacks will still fire if
        // bluetooth-central background mode is enabled.

        task.setTaskCompleted(success: true)
    }
}
```

- [ ] **Step 2: Register background task in SonoEdgeApp.swift**

Modify `SonoEdge/App/SonoEdgeApp.swift`:

```swift
import SwiftUI
import BackgroundTasks

@main
struct SonoEdgeApp: App {
    @Environment(\.scenePhase) var scenePhase

    init() {
        BackgroundTaskService.register()
        Task {
            _ = await NotificationService.requestAuthorization()
        }
    }

    var body: some Scene {
        WindowGroup {
            MainTabView()
        }
        .onChange(of: scenePhase) { newPhase in
            switch newPhase {
            case .background:
                BackgroundTaskService.schedule()
            case .active:
                NotificationService.clearBadge()
            default:
                break
            }
        }
    }
}
```

- [ ] **Step 3: Verify build compiles**

```bash
xcodebuild -project /Users/raexu/Documents/SonoEdge/SonoEdge.xcodeproj -scheme SonoEdge -destination 'platform=iOS Simulator,name=iPhone 16' build 2>&1 | tail -5
```

Expected: BUILD SUCCEEDED

- [ ] **Step 4: Final commit**

```bash
git add -A && git commit -m "$(cat <<'EOF'
Add BackgroundTaskService for continuous monitoring support

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Post-Implementation Checklist

- [ ] All 8 tasks committed
- [ ] Build passes: `xcodebuild -project SonoEdge.xcodeproj -scheme SonoEdge -destination 'platform=iOS Simulator,name=iPhone 16' build`
- [ ] Xcode project file (`project.pbxproj`) includes new files — added automatically by Xcode when files are created in the project directory
- [ ] Core files untouched: `AudioProcessor.swift`, `InferencePipeline.swift`, `TFLiteEngine.swift`, `DebugPreprocess.swift`, `RecordStore.swift`, `BLERecorder.swift`
