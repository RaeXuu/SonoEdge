import Foundation
import Accelerate

// ================================================================
// Aligns with Pi main_pi.py run_inference() two-stage streaming pipeline:
//   SQA gate (threshold 0.65) → diagnosis inference → weighted average
//   Per-window progress callbacks for streaming UI updates
//
// Index convention:
//   SQA model:  index 0 = Good, index 1 = Poor
//   Diag model:  index 0 = Normal, index 1 = Abnormal
// ================================================================

// MARK: - Type definitions

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

/// Callback after each window is processed
/// - runningAvgNormal: Weighted average P(Normal) up to current window, nil means no valid windows yet
/// - currentWindow: Current window index (1-based)
/// - totalWindows: Total number of windows
typealias WindowProgressCallback = (
    _ runningAvgNormal: Float?,
    _ currentWindow: Int,
    _ totalWindows: Int,
    _ latestResult: WindowResult
) async -> Void

// MARK: - Pipeline

final class InferencePipeline {

    private let sqaEngine: TFLiteEngine
    private let diagEngine: TFLiteEngine

    private let sqaThreshold: Float  = 0.05
    private let diagThreshold: Float = 0.5

    private let chunkSamples: Int  = 2000 * 20   // 40000
    private let segSamples: Int    = 4000
    private let hopSamples: Int    = 2000

    init(sqaEngine: TFLiteEngine, diagEngine: TFLiteEngine) {
        self.sqaEngine = sqaEngine
        self.diagEngine = diagEngine
    }

    // MARK: - Entry

    /// Two-stage streaming inference on a chunk of int16 raw bytes
    /// - Parameters:
    ///   - rawBytes: int16 PCM @ 2000Hz, 20s (80000 bytes)
    ///   - onWindow: Callback after each window (non-nil enables per-window progress)
    func run(on rawBytes: Data,
             onWindow: WindowProgressCallback? = nil) async throws -> ChunkResult {
        let t0 = CFAbsoluteTimeGetCurrent()

        // 1. int16 → float32 / 32768.0 (aligns with main_pi.py:129)
        var audio = rawBytes.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) -> [Float] in
            let intPtr = ptr.bindMemory(to: Int16.self)
            let count = min(rawBytes.count / 2, chunkSamples)
            return (0..<count).map { Float(intPtr[$0]) / 32768.0 }
        }

        if audio.count < chunkSamples {
            audio.append(contentsOf: [Float](repeating: 0,
                                             count: chunkSamples - audio.count))
        }

        // 2. Bandpass filter (whole chunk at once)
        let filtered = ButterworthBandpass.apply(to: audio)

        // 3. Sliding windows + streaming inference
        var validResults = [(sqa: Float, normal: Float)]()
        var windowDetails = [WindowResult]()
        let totalWindows = (chunkSamples - segSamples) / hopSamples + 1  // 19

        for winIdx in 0..<totalWindows {
            let start = winIdx * hopSamples
            let window = Array(filtered[start..<start + segSamples])

            // Mel spectrogram (64×64 → flattened to 4096, includes peak normalization)
            let mel = MelSpectrogram.compute(from: window)

            // SQA inference
            let sqaRaw = try sqaEngine.infer(floatInput: mel)
            let sqaProbs = softmax(sqaRaw)
            let sqaScore = sqaProbs[0]

            let passed = sqaScore >= sqaThreshold

            if !passed {
                let r = WindowResult(windowIndex: winIdx + 1,
                                     sqaScore: sqaScore,
                                     passedSQA: false, probNormal: nil)
                windowDetails.append(r)

                // Progress callback: weighted average unchanged
                let runningAvg: Float? = validResults.isEmpty ? nil : {
                    let w = validResults.map { $0.sqa }
                    let p = validResults.map { $0.normal }
                    return zip(w, p).reduce(0) { $0 + $1.0 * $1.1 } / w.reduce(0, +)
                }()
                await onWindow?(runningAvg, winIdx + 1, totalWindows, r)
                continue
            }

            // Diagnosis inference
            let diagRaw = try diagEngine.infer(floatInput: mel)
            let diagProbs = softmax(diagRaw)
            let probNormal = diagProbs[0]

            validResults.append((sqa: sqaScore, normal: probNormal))
            let r = WindowResult(windowIndex: winIdx + 1,
                                 sqaScore: sqaScore,
                                 passedSQA: true, probNormal: probNormal)
            windowDetails.append(r)

            // Compute current cumulative weighted average
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
