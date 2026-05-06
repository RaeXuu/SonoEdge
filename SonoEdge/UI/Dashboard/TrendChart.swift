import SwiftUI

struct TrendChart: View {
    let readings: [ReadingPoint]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Recent Trends")
                .font(.subheadline)
                .foregroundColor(.secondary)

            if readings.isEmpty {
                HStack {
                    Spacer()
                    VStack(spacing: 6) {
                        Image(systemName: "chart.line.uptrend.xyaxis")
                            .font(.title2)
                            .foregroundColor(.secondary.opacity(0.5))
                        Text("Start monitoring to see 24-hour trends")
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
                        ForEach(0..<3) { i in
                            let y = height * CGFloat(i) / 2
                            Path { path in
                                path.move(to: CGPoint(x: 0, y: y))
                                path.addLine(to: CGPoint(x: width, y: y))
                            }
                            .stroke(Color(.systemGray5), lineWidth: 0.5)
                        }

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

                if let last = readings.last {
                    HStack {
                        Text("Past 24 Hours")
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
            return (0..<sorted.count).map { i in
                CGPoint(x: width * CGFloat(i) / CGFloat(sorted.count - 1), y: height / 2)
            }
        }

        return (0..<sorted.count).map { i in
            let x = width * CGFloat(i) / CGFloat(sorted.count - 1)
            let y = height * (1 - (probs[i] - minP) / (maxP - minP))
            return CGPoint(x: x, y: max(2, min(y, height - 2)))
        }
    }
}
