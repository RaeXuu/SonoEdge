import SwiftUI

struct SettingsView: View {
    @ObservedObject var service: MonitoringService
    @State private var alertsEnabled = true

    var body: some View {
        NavigationView {
            List {
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

                Section("数据分享") {
                    Button(action: shareReport) {
                        Label("分享给医生", systemImage: "square.and.arrow.up")
                    }
                }

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

        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let root = windowScene.windows.first?.rootViewController {
            root.present(av, animated: true)
        }
    }
}
