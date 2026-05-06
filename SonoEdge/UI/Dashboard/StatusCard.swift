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
        if !service.isConnected { return "Disconnected" }
        if service.isProcessingChunk { return "Monitoring..." }
        guard let label = service.lastLabel else { return "Waiting for Data" }
        return label == "Normal" ? "Heart Rate Normal" : "Heart Rate Abnormal"
    }

    private var statusSubtitle: String {
        if !service.isConnected { return "Tap the button below to connect" }
        if service.isProcessingChunk { return "Analyzing heart sound signals..." }
        guard let label = service.lastLabel, let prob = service.lastProbNormal else {
            return "No detection data available"
        }
        let confidence = label == "Normal" ? prob : 1 - prob
        return String(format: "Confidence: %.0f%% · Last: \(service.lastInferenceMs.map { String(format: "%.0fms", $0) } ?? "--")", confidence * 100)
    }
}
