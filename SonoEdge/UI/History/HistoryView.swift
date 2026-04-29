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
                filterBar

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

    private var filteredRecords: [[String: Any]] {
        if filter == .all { return records }
        return records.filter {
            ($0["label"] as? String ?? "noise") == filter.rawValue
        }
    }

    private var groupedRecords: [(date: String, items: [[String: Any]])] {
        let grouped = Dictionary(grouping: filteredRecords) { record -> String in
            let ts = record["ts"] as? String ?? ""
            return String(ts.prefix(10))
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
