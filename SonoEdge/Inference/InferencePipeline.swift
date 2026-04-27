import Foundation
import Accelerate

// ================================================================
// 对齐 Pi 端 main_pi.py 的 run_inference() 双阶段流式推理流水线：
//   SQA 门控 (阈值 0.65) → 诊断推理 → 加权平均
//   逐窗口驱动进度回调，实现流式 UI 更新
//
// 索引约定:
//   SQA 模型 : index 0 = Good, index 1 = Bad
//   诊断模型: index 0 = Normal, index 1 = Abnormal
// ================================================================

// MARK: - 类型定义

struct WindowResult: Equatable {
    let windowIndex: Int
    let sqaScore: Float
    let passedSQA: Bool
    let probNormal: Float?
}

struct ChunkResult {
    let label: String?
    let avgProbNormal: Float?
    let validWindows: Int
    let totalWindows: Int
    let windowDetails: [WindowResult]
    let inferenceMs: Double
}

/// 每处理完一个窗口时回调
/// - runningAvgNormal: 截止当前窗口的加权平均 P(Normal), nil 表示尚无有效窗口
/// - currentWindow: 当前窗口编号 (1-based)
/// - totalWindows: 总窗口数
typealias WindowProgressCallback = (
    _ runningAvgNormal: Float?,
    _ currentWindow: Int,
    _ totalWindows: Int,
    _ latestResult: WindowResult
) async -> Void

// MARK: - 流水线

final class InferencePipeline {

    private let sqaEngine: TFLiteEngine
    private let diagEngine: TFLiteEngine

    private let sqaThreshold: Float  = 0.65
    private let diagThreshold: Float = 0.5

    private let chunkSamples: Int  = 2000 * 20   // 40000
    private let segSamples: Int    = 4000
    private let hopSamples: Int    = 2000

    init(sqaEngine: TFLiteEngine, diagEngine: TFLiteEngine) {
        self.sqaEngine = sqaEngine
        self.diagEngine = diagEngine
    }

    // MARK: - 入口

    /// 对一块 int16 raw bytes 做双阶段流式推理
    /// - Parameters:
    ///   - rawBytes: int16 PCM @ 2000Hz, 20s (80000 bytes)
    ///   - onWindow: 每处理完一个窗口时回调 (non-nil即可启用逐窗进度)
    func run(on rawBytes: Data,
             onWindow: WindowProgressCallback? = nil) async throws -> ChunkResult {
        let t0 = CFAbsoluteTimeGetCurrent()

        // 1. int16 → float32 / 32768.0 (对齐 main_pi.py:129)
        var audio = rawBytes.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) -> [Float] in
            let intPtr = ptr.bindMemory(to: Int16.self)
            let count = min(rawBytes.count / 2, chunkSamples)
            return (0..<count).map { Float(intPtr[$0]) / 32768.0 }
        }

        if audio.count < chunkSamples {
            audio.append(contentsOf: [Float](repeating: 0,
                                             count: chunkSamples - audio.count))
        }

        // 2. 带通滤波 (整块一次)
        let filtered = ButterworthBandpass.apply(to: audio)

        // 3. 滑动窗口 + 流式推理
        var validResults = [(sqa: Float, normal: Float)]()
        var windowDetails = [WindowResult]()
        let totalWindows = (chunkSamples - segSamples) / hopSamples + 1  // 19

        for winIdx in 0..<totalWindows {
            let start = winIdx * hopSamples
            let window = Array(filtered[start..<start + segSamples])

            // Mel 频谱 (64×64 → 展平 4096, 内含峰值归一化)
            let mel = MelSpectrogram.compute(from: window)

            // SQA 推理
            let sqaRaw = try sqaEngine.infer(floatInput: mel)
            let sqaProbs = softmax(sqaRaw)
            let sqaScore = sqaProbs[0]

            let passed = sqaScore >= sqaThreshold

            if !passed {
                let r = WindowResult(windowIndex: winIdx + 1,
                                     sqaScore: sqaScore,
                                     passedSQA: false, probNormal: nil)
                windowDetails.append(r)

                // 进度回调: 加权平均保持不变
                let runningAvg: Float? = validResults.isEmpty ? nil : {
                    let w = validResults.map { $0.sqa }
                    let p = validResults.map { $0.normal }
                    return zip(w, p).reduce(0) { $0 + $1.0 * $1.1 } / w.reduce(0, +)
                }()
                await onWindow?(runningAvg, winIdx + 1, totalWindows, r)
                continue
            }

            // 诊断推理
            let diagRaw = try diagEngine.infer(floatInput: mel)
            let diagProbs = softmax(diagRaw)
            let probNormal = diagProbs[0]

            validResults.append((sqa: sqaScore, normal: probNormal))
            let r = WindowResult(windowIndex: winIdx + 1,
                                 sqaScore: sqaScore,
                                 passedSQA: true, probNormal: probNormal)
            windowDetails.append(r)

            // 计算当前累计加权平均
            let w = validResults.map { $0.sqa }
            let p = validResults.map { $0.normal }
            let runningAvg = zip(w, p).reduce(0) { $0 + $1.0 * $1.1 } / w.reduce(0, +)

            await onWindow?(runningAvg, winIdx + 1, totalWindows, r)
        }

        let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        guard !validResults.isEmpty else {
            return ChunkResult(label: nil, avgProbNormal: nil,
                               validWindows: 0, totalWindows: totalWindows,
                               windowDetails: windowDetails,
                               inferenceMs: elapsed)
        }

        let weights = validResults.map { $0.sqa }
        let normals = validResults.map { $0.normal }
        let avgNormal = zip(weights, normals).reduce(0) { $0 + $1.0 * $1.1 }
                       / weights.reduce(0, +)
        let label = avgNormal > diagThreshold ? "Normal" : "Abnormal"

        return ChunkResult(label: label,
                           avgProbNormal: avgNormal,
                           validWindows: validResults.count,
                           totalWindows: totalWindows,
                           windowDetails: windowDetails,
                           inferenceMs: elapsed)
    }

    // MARK: - Softmax

    private func softmax(_ x: [Float]) -> [Float] {
        let maxX = x.max() ?? 0
        let expX = x.map { exp($0 - maxX) }
        let sum = expX.reduce(0, +)
        return expX.map { $0 / sum }
    }
}
