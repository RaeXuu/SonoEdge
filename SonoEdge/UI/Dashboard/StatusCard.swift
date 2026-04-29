import SwiftUI

struct StatusCard: View {
    @ObservedObject var service: MonitoringService

    var body: some View {
        VStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(statusColor.opacity(0.15))
                    .frame(width: 80, height: 80)
                Image(systemName: statusIcon)
                    .font(.system(size: 36))
                    .foregroundColor(statusColor)
            }

            Text(statusTitle)
                .font(.title2.bold())
                .foregroundColor(statusColor)

            Text(statusSubtitle)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

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
