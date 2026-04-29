import Foundation

/// 对齐 Pi 端 src/storage/summary.py 的 append_summary
/// JSONL 格式，每条记录一行
struct RecordStore {

    private static let fileManager = FileManager.default
    private static let documentsDir: URL = {
        fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
    }()

    private static var recordsDir: URL {
        documentsDir.appendingPathComponent("records")
    }

    private static var summaryPath: URL {
        recordsDir.appendingPathComponent("summary.jsonl")
    }

    // MARK: - Summary (JSONL)

    /// 每块推理完成后追加一行到 records/summary.jsonl
    static func appendSummary(label: String?,
                              probNormal: Float?,
                              validSegs: Int,
                              totalSegs: Int) {
        ensureDir()

        let record: [String: Any] = [
            "ts": ISO8601DateFormatter().string(from: Date()),
            "label": label ?? "noise",
            "prob_normal": probNormal.map { round($0 * 10000) / 10000 } as Any,
            "valid_segs": validSegs,
            "total_segs": totalSegs,
        ]

        guard let data = try? JSONSerialization.data(withJSONObject: record, options: []),
              let line = String(data: data, encoding: .utf8) else { return }

        if fileManager.fileExists(atPath: summaryPath.path) {
            guard let handle = try? FileHandle(forWritingTo: summaryPath) else { return }
            handle.seekToEndOfFile()
            handle.write((line + "\n").data(using: .utf8)!)
            handle.closeFile()
        } else {
            try? (line + "\n").write(to: summaryPath, atomically: true, encoding: .utf8)
        }
    }

    /// 加载所有记录，用于"记录"tab 展示
    static func loadSummaries() -> [[String: Any]] {
        guard let content = try? String(contentsOf: summaryPath, encoding: .utf8) else { return [] }
        return content
            .split(separator: "\n", omittingEmptySubsequences: true)
            .compactMap { line in
                guard let data = line.data(using: .utf8),
                      let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
                else { return nil }
                return obj
            }
    }

    // MARK: - Internal

    private static func ensureDir() {
        if !fileManager.fileExists(atPath: recordsDir.path) {
            try? fileManager.createDirectory(at: recordsDir,
                                             withIntermediateDirectories: true)
            // 标记不备份到 iCloud
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            var url = recordsDir
            try? url.setResourceValues(values)
        }
    }
}
