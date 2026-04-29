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
