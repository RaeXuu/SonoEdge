import SwiftUI

// MARK: - Main Tab View

struct MainTabView: View {
    @StateObject private var vm = AppViewModel()
    @State private var selectedTab = 0

    var body: some View {
        TabView(selection: $selectedTab) {
            ExamineView(vm: vm)
                .tabItem {
                    Image(systemName: "stethoscope")
                    Text("检测")
                }
                .tag(0)

            HistoryView()
                .tabItem {
                    Image(systemName: "clock.arrow.circlepath")
                    Text("记录")
                }
                .tag(1)

            ProfileView()
                .tabItem {
                    Image(systemName: "person.circle")
                    Text("我的")
                }
                .tag(2)
        }
        .accentColor(.blue)
        .onAppear {
            do { try vm.setup() }
            catch { vm.connectionStatus = "模型加载失败: \(error.localizedDescription)" }
        }
    }
}

// MARK: - 检测 Tab

struct ExamineView: View {
    @ObservedObject var vm: AppViewModel

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {

                    // ── 连接卡片 ──
                    connectionCard

                    // ── 实时状态 ──
                    if vm.isConnected {
                        statusCard
                    }

                    // ── 进度条 ──
                    if vm.isProcessingChunk {
                        progressCard
                    }

                    // ── 统计 ──
                    statsCard

                    // ── 逐窗详情 ──
                    if !vm.currentWindowDetails.isEmpty {
                        detailCard
                    }

                    Spacer(minLength: 40)
                }
                .padding()
            }
            .navigationTitle("SonoEdge")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("重置") { vm.resetTally() }
                        .font(.caption)
                }
            }
        }
    }

    // MARK: - 连接卡片

    private var connectionCard: some View {
        VStack(spacing: 12) {
            // 连接按钮
            Button(action: {
                if vm.isConnected { vm.disconnect() }
                else { vm.connect() }
            }) {
                ZStack {
                    Circle()
                        .fill(vm.isConnected ? Color.red : Color.blue)
                        .frame(width: 100, height: 100)
                        .shadow(color: (vm.isConnected ? Color.red : Color.blue).opacity(0.4),
                                radius: 12)

                    Image(systemName: vm.isConnected ? "stop.fill" : "bolt.fill")
                        .font(.system(size: 40))
                        .foregroundColor(.white)
                }
            }

            // 状态文字
            HStack(spacing: 6) {
                Circle()
                    .fill(vm.isConnected
                          ? (vm.isRunning ? Color.green : Color.orange)
                          : Color.gray)
                    .frame(width: 8, height: 8)
                Text(vm.connectionStatus)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 20)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.06), radius: 8, y: 2)
        )
    }

    // MARK: - 状态卡片

    private var statusCard: some View {
        VStack(spacing: 10) {
            if vm.isProcessingChunk {
                HStack {
                    ProgressView()
                        .scaleEffect(0.8)
                    Text("分析中...")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
            }

            Text(vm.statusText)
                .font(.body)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            if let ms = vm.lastInferenceMs {
                Text("耗时 \(String(format: "%.0f", ms)) ms")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.04), radius: 6, y: 1)
        )
    }

    // MARK: - 进度卡片

    private var progressCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("窗口 \(vm.currentWindow) / \(vm.totalWindows)")
                    .font(.caption)
                Spacer()
                if let avg = vm.runningAvgNormal {
                    Text(avg > 0.5 ? "Normal" : "Abnormal")
                        .font(.caption.bold())
                        .foregroundColor(avg > 0.5 ? .green : .red)
                }
            }
            ProgressView(value: Double(vm.currentWindow), total: Double(vm.totalWindows))
                .tint((vm.runningAvgNormal ?? 0) > 0.5 ? .green : .red)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.04), radius: 6, y: 1)
        )
    }

    // MARK: - 统计卡片

    private var statsCard: some View {
        HStack(spacing: 0) {
            StatItem(count: vm.totalNormal, label: "Normal", color: .green)
            Divider().frame(height: 40)
            StatItem(count: vm.totalAbnormal, label: "Abnormal", color: .red)
            Divider().frame(height: 40)
            StatItem(count: vm.totalNoise, label: "Noise", color: .gray)
        }
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.04), radius: 6, y: 1)
        )
    }

    // MARK: - 详情卡片

    private var detailCard: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("逐窗详情")
                .font(.caption)
                .foregroundColor(.secondary)

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(vm.currentWindowDetails, id: \.windowIndex) { w in
                            windowRow(w).id(w.windowIndex)
                        }
                    }
                }
                .frame(maxHeight: 180)
                .onChange(of: vm.currentWindowDetails.count) { _ in
                    if let last = vm.currentWindowDetails.last {
                        withAnimation { proxy.scrollTo(last.windowIndex, anchor: .bottom) }
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

    private func windowRow(_ w: WindowResult) -> some View {
        HStack {
            Text(String(format: "W%02d", w.windowIndex))
                .font(.system(.caption, design: .monospaced))
            Text(String(format: "SQA:%.2f", w.sqaScore))
                .font(.system(.caption, design: .monospaced))
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

// MARK: - 统计项

struct StatItem: View {
    let count: Int
    let label: String
    let color: Color

    var body: some View {
        VStack(spacing: 2) {
            Text("\(count)")
                .font(.title2.bold())
                .foregroundColor(color)
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - 记录 Tab（占位）

struct HistoryView: View {
    var body: some View {
        NavigationView {
            VStack {
                Image(systemName: "tray")
                    .font(.system(size: 40))
                    .foregroundColor(.secondary)
                Text("暂无记录")
                    .foregroundColor(.secondary)
            }
            .navigationTitle("记录")
        }
    }
}

// MARK: - 我的 Tab（占位）

struct ProfileView: View {
    var body: some View {
        NavigationView {
            List {
                Section("设备") {
                    HStack {
                        Label("电子听诊器", systemImage: "wave.3.right")
                        Spacer()
                        Text("未连接")
                            .foregroundColor(.secondary)
                    }
                }
                Section("关于") {
                    Label("版本 1.0", systemImage: "info.circle")
                }
            }
            .navigationTitle("我的")
        }
    }
}
