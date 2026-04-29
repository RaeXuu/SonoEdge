import SwiftUI
import Combine

// MARK: - ViewModel

@MainActor
final class AppViewModel: ObservableObject {

    @Published var isConnected = false
    @Published var isRunning = false
    @Published var connectionStatus = "未连接"
    @Published var statusText = "就绪"
    @Published var lastLabel: String? = nil
    @Published var lastProbNormal: Float? = nil
    @Published var lastInferenceMs: Double? = nil
    @Published var totalNormal = 0
    @Published var totalAbnormal = 0
    @Published var totalNoise = 0

    /// 当前块的逐窗结果 (实时增长)
    @Published var currentWindowDetails: [WindowResult] = []
    @Published var currentWindow: Int = 0
    @Published var totalWindows: Int = 19
    @Published var runningAvgNormal: Float? = nil
    @Published var isProcessingChunk = false

    private var bleRecorder: BLERecorder?
    private var pipeline: InferencePipeline?

    private let modelDir: String = {
        Bundle.main.resourcePath!
    }()

    // MARK: - Init engines

    func setup() throws {
        let sqaPath  = modelDir + "/heart_quality_int8full.tflite"
        let diagPath = modelDir + "/heart_model_int8full.tflite"

        let sqa  = try TFLiteEngine(modelPath: sqaPath)
        let diag = try TFLiteEngine(modelPath: diagPath)

        try sqa.warmup(count: 5)
        try diag.warmup(count: 5)

        pipeline = InferencePipeline(sqaEngine: sqa, diagEngine: diag)

#if DEBUG
        // 比对预处理：把测试 WAV 加入项目，名字改成实际文件名即可
        DebugPreprocess.run(wavName: "test_comparison.wav", pipeline: pipeline!)
#endif

        bleRecorder = BLERecorder()
        bleRecorder?.onChunkReady = { [weak self] rawChunk in
            Task { [weak self] in
                await self?.processChunk(rawChunk)
            }
        }

        // 同步 BLE 状态
        bleRecorder?.$connectionStatus.assign(to: &$connectionStatus)
        bleRecorder?.$isRunning.assign(to: &$isRunning)
    }

    // MARK: - Connect / Disconnect

    func connect() {
        bleRecorder?.startScan()
    }

    func disconnect() {
        bleRecorder?.disconnect()
        isConnected = false
        statusText = "已断开"
        isProcessingChunk = false
    }

    /// 监听 BLE 状态变化
    func syncBLEState() {
        guard let ble = bleRecorder else { return }
        isConnected = ble.isRunning
        connectionStatus = ble.connectionStatus
    }

    // MARK: - Process chunk (流式)

    private func processChunk(_ raw: Data) async {
        guard let pipeline = pipeline else { return }

        await MainActor.run {
            self.isProcessingChunk = true
            self.currentWindowDetails = []
            self.currentWindow = 0
            self.totalWindows = 19
        }

        do {
            let result = try await pipeline.run(on: raw) {
                [weak self] runningAvg, win, total, latest in
                await MainActor.run { [weak self] in
                    guard let self = self else { return }
                    self.runningAvgNormal = runningAvg
                    self.currentWindow = win
                    self.totalWindows = total
                    self.currentWindowDetails.append(latest)

                    if let avg = runningAvg {
                        let label = avg > 0.5 ? "Normal" : "Abnormal"
                        self.statusText = "W\(win)/\(total) | \(label) \(String(format: "%.1f", avg * 100))%"
                    } else {
                        self.statusText = "W\(win)/\(total) | 等待有效窗口..."
                    }
                }
            }

            await MainActor.run {
                self.lastInferenceMs = result.inferenceMs

                if let label = result.label {
                    self.lastLabel = label
                    self.lastProbNormal = result.avgProbNormal
                    if label == "Normal" { self.totalNormal += 1 }
                    else { self.totalAbnormal += 1 }
                    self.statusText = "\(label) (\(String(format: "%.1f", (result.avgProbNormal ?? 0) * 100))%)"
                } else {
                    self.lastLabel = nil
                    self.lastProbNormal = nil
                    self.totalNoise += 1
                    self.statusText = "低质量 (\(result.validWindows)/\(result.totalWindows) 窗口)"
                }

                RecordStore.appendSummary(label: result.label,
                                          probNormal: result.avgProbNormal,
                                          validSegs: result.validWindows,
                                          totalSegs: result.totalWindows)

                self.isProcessingChunk = false
            }
        } catch {
            await MainActor.run {
                self.statusText = "推理错误: \(error.localizedDescription)"
                self.isProcessingChunk = false
            }
        }

        bleRecorder?.markChunkConsumed()
    }

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

// MARK: - Content View

struct ContentView: View {
    @StateObject private var vm = AppViewModel()

    var body: some View {
        VStack(spacing: 16) {
            Spacer().frame(height: 8)

            // ── 标题 ──
            Text("SonoEdge")
                .font(.largeTitle.bold())

            // ── 连接状态 ──
            connectionStatusBar

            // ── 推理状态 ──
            if vm.isConnected {
               HStack {
                   Circle()
                       .fill(vm.isProcessingChunk ? Color.orange : Color.blue)
                       .frame(width: 10, height: 10)
                   Text(vm.statusText)
                       .font(.subheadline)
                       .lineLimit(2)
               }
            }

            // ── 连接/断开按钮 ──
            connectButton

            // ── 流式进度条 ──
            if vm.isProcessingChunk {
                progressBar
                    .padding(.horizontal, 30)
            }

            // ── 统计 ──
            HStack(spacing: 30) {
                tallyItem("Normal", vm.totalNormal, .green)
                tallyItem("Abnormal", vm.totalAbnormal, .red)
                tallyItem("Noise", vm.totalNoise, .gray)
            }

            // ── 耗时 ──
            if let ms = vm.lastInferenceMs {
                Text("上一块耗时: \(String(format: "%.0f", ms)) ms")
                    .font(.caption).foregroundColor(.secondary)
            }

            // ── 逐窗详情 ──
            if !vm.currentWindowDetails.isEmpty {
                windowDetailList
                    .frame(maxHeight: 220)
            }

            // ── 重置 ──
            Button("重置计数") { vm.resetTally() }
                .font(.caption)
                .padding(.bottom, 16)
        }
        .padding(.horizontal)
        .onAppear {
            do { try vm.setup() }
            catch { vm.connectionStatus = "模型加载失败: \(error.localizedDescription)" }
        }
    }

    // MARK: - Subviews

    private var connectionStatusBar: some View {
        HStack {
            Circle()
                .fill(vm.isConnected
                      ? (vm.isRunning ? Color.green : Color.orange)
                      : Color.gray)
                .frame(width: 10, height: 10)
            Text(vm.connectionStatus)
                .font(.headline)
        }
    }

    private var connectButton: some View {
        Button(action: {
            if vm.isConnected {
                vm.disconnect()
            } else {
                vm.connect()
            }
        }) {
            ZStack {
                Circle()
                    .fill(vm.isConnected ? Color.red : Color.blue)
                    .frame(width: 88, height: 88)
                Image(systemName: vm.isConnected ? "stop.fill" : "bolt.fill")
                    .font(.system(size: 36))
                    .foregroundColor(.white)
            }
        }
    }

    private var progressBar: some View {
        ProgressView(value: Double(vm.currentWindow), total: Double(vm.totalWindows)) {
            HStack {
                Text("窗口 \(vm.currentWindow) / \(vm.totalWindows)")
                    .font(.caption)
                Spacer()
                if let avg = vm.runningAvgNormal {
                    Text("\(avg > 0.5 ? "Normal" : "Abnormal") \(String(format: "%.1f", avg * 100))%")
                        .font(.caption.bold())
                        .foregroundColor(avg > 0.5 ? .green : .red)
                }
            }
        }
        .tint(vm.runningAvgNormal.map { $0 > 0.5 ? Color.green : Color.red } ?? .blue)
    }

    private func tallyItem(_ label: String, _ count: Int, _ color: Color) -> some View {
        VStack {
            Text("\(count)").font(.title2.bold()).foregroundColor(color)
            Text(label).font(.caption)
        }
    }

    private var windowDetailList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(vm.currentWindowDetails, id: \.windowIndex) { w in
                        windowRow(w).id(w.windowIndex)
                    }
                }
            }
            .onChange(of: vm.currentWindowDetails.count) { _ in
                if let last = vm.currentWindowDetails.last {
                    withAnimation { proxy.scrollTo(last.windowIndex, anchor: .bottom) }
                }
            }
            .padding(6)
            .background(Color(.systemGray6))
            .cornerRadius(8)
        }
    }

    private func windowRow(_ w: WindowResult) -> some View {
        HStack {
            Text(String(format: "W%02d", w.windowIndex))
                .font(.system(.caption, design: .monospaced))
            Text(w.passedSQA ? "✓" : "✗")
                .font(.caption)
                .foregroundColor(w.passedSQA ? .green : .gray)
            if w.passedSQA, let pn = w.probNormal {
                Text(String(format: "P(N):%.2f", pn))
                    .font(.system(.caption, design: .monospaced))
                    .foregroundColor(pn > 0.5 ? .green : .red)
            } else {
                Text("✗").font(.caption).foregroundColor(.gray)
            }
        }
    }
}
