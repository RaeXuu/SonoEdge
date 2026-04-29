import SwiftUI

struct DashboardView: View {
    @ObservedObject var service: MonitoringService

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    StatusCard(service: service)

                    monitoringButton

                    if service.isProcessingChunk {
                        progressSection
                    }

                    TrendChart(readings: service.recentReadings)

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
