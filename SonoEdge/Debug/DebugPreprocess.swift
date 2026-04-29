import Foundation
import Accelerate


/// 对齐 Pi 端 main_pi_debug.py：加载 WAV → 预处理 → 推理 → 打印中间值
/// 用于对比 Pi 和 iOS 的预处理差异
struct DebugPreprocess {

    /// 处理 app bundle 中的 WAV 文件，打印 Mel 值、SQA 分数、诊断结果
    static func run(wavName: String, pipeline: InferencePipeline) {
        guard let url = Bundle.main.url(forResource: wavName, withExtension: nil) else {
            print("[Debug] 找不到文件: \(wavName)")
            return
        }

        guard let wav = readWAV(url: url) else {
            print("[Debug] WAV 解析失败: \(wavName)")
            return
        }

        let sr = wav.sampleRate
        var audio = wav.samples

        // 如果采样率不是 2000，简单降采样（取平均）
        if sr != 2000 {
            let ratio = sr / 2000
            var downsampled = [Float]()
            for i in stride(from: 0, to: audio.count - ratio, by: ratio) {
                let chunk = audio[i..<i + ratio]
                downsampled.append(chunk.reduce(0, +) / Float(ratio))
            }
            audio = downsampled
            print("[Debug] 降采样: \(sr) → 2000Hz, 样本数: \(audio.count)")
        }

        // 全局峰值归一化，对齐 Pi 端 load_wav 的行为
        let globalMax = audio.map(abs).max() ?? 1.0
        if globalMax > 0 { audio = audio.map { $0 / globalMax } }

        let chunkSamples = 40000
        if audio.count < chunkSamples {
            audio.append(contentsOf: [Float](repeating: 0, count: chunkSamples - audio.count))
        }
        let chunk = Array(audio.prefix(chunkSamples))

        print("[Debug] === 预处理对比 ===")
        print("[Debug] 原始: first10=\(chunk[0..<10].map { String(format: "%.6f", $0) })")

        let filtered = ButterworthBandpass.apply(to: chunk)
        print("[Debug] filtfilt chunk.count=\(chunk.count) filtered.count=\(filtered.count)")
        print("[Debug] 滤波后: first10=\(filtered[0..<10].map { String(format: "%.6f", $0) })")
        print("[Debug] 滤波后[100..<110]=\(filtered[100..<110].map { String(format: "%.6f", $0) })")

        let window = Array(filtered[0..<4000])
        let mx = window.map(abs).max() ?? 1.0
        let normalized = mx > 0 ? window.map { $0 / mx } : window
        print("[Debug] 归一化后: first10=\(normalized[0..<10].map { String(format: "%.6f", $0) }) max=\(mx)")

        // STFT 第一帧
        let stftFirst = DebugPreprocess.stftFirstFrame(signal: normalized)
        print("[Debug] STFT[0]: first10=\(stftFirst[0..<10].map { String(format: "%.2f", $0) })")

        let mel = MelSpectrogram.compute(from: normalized)
        print("[Debug] Mel[0..<10] = \(mel[0..<10].map { String(format: "%.6f", $0) })")
        print("[Debug] Mel[64..<74] = \(mel[64..<74].map { String(format: "%.6f", $0) })")
        print("[Debug] Mel min=\(mel.min()!) max=\(mel.max()!) mean=\(mel.reduce(0,+)/Float(mel.count))")

        // 跑一遍推理
        print("[Debug] === 推理结果 ===")
        let int16Samples = chunk.map { Int16(max(-32768, min(32767, ($0 * 32767).rounded()))) }
        let rawBytes = int16Samples.withUnsafeBytes { Data($0) }

        Task {
            do {
                let result = try await pipeline.run(on: rawBytes)
                print("[Debug] label=\(result.label ?? "noise") avgP(N)=\(result.avgProbNormal.map { String(format: "%.6f", $0) } ?? "nil") valid=\(result.validWindows)/\(result.totalWindows) ms=\(String(format: "%.0f", result.inferenceMs))")
                for w in result.windowDetails {
                    let pn = w.probNormal.map { String(format: "%.4f", $0) } ?? "-"
                    print("[Debug]   W\(String(format: "%02d", w.windowIndex)) SQA=\(String(format: "%.4f", w.sqaScore)) passed=\(w.passedSQA) P(N)=\(pn)")
                }
            } catch {
                print("[Debug] 推理失败: \(error)")
            }
        }
    }

    // MARK: - WAV reader

    // MARK: - STFT first frame (mirrors MelSpectrogram.stftLibrosaStyle)

    static func stftFirstFrame(signal: [Float]) -> [Float] {
        let nFFT = 256, winLen = 256, pad = 128
        let padded = [Float](repeating: 0, count: pad) + signal + [Float](repeating: 0, count: pad)

        var window = [Float](repeating: 0, count: winLen)
        vDSP_hann_window(&window, vDSP_Length(winLen), Int32(vDSP_HANN_DENORM))

        let half = nFFT / 2
        var realPart = [Float](repeating: 0, count: half)
        var imagPart = [Float](repeating: 0, count: half)
        for j in 0..<half {
            realPart[j] = padded[2 * j]       * window[2 * j]
            imagPart[j] = padded[2 * j + 1]   * window[2 * j + 1]
        }

        realPart.withUnsafeMutableBufferPointer { rp in
            imagPart.withUnsafeMutableBufferPointer { ip in
                var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                vDSP.FFT(log2n: vDSP_Length(log2(Float(nFFT))), radix: .radix2, ofType: DSPSplitComplex.self)!.forward(input: split, output: &split)
            }
        }

        var mag = [Float](repeating: 0, count: nFFT / 2 + 1)
        mag[0] = pow(abs(realPart[0] / 2.0), 2.0)
        for k in 1..<(nFFT / 2) {
            let m = sqrt(realPart[k] * realPart[k] + imagPart[k] * imagPart[k]) / 2.0
            mag[k] = m * m
        }
        mag[nFFT / 2] = pow(abs(imagPart[0] / 2.0), 2.0)
        return mag
    }

    private struct WAVFile {
        let sampleRate: Int
        let samples: [Float]
    }

    private static func readWAV(url: URL) -> WAVFile? {
        guard let data = try? Data(contentsOf: url) else { return nil }

        // WAV header: "RIFF" + fileSize + "WAVE" + "fmt " + fmtSize
        guard data.count > 44,
              String(data: data.subdata(in: 0..<4), encoding: .ascii) == "RIFF",
              String(data: data.subdata(in: 8..<12), encoding: .ascii) == "WAVE" else {
            print("[Debug] 不是有效的 WAV 文件")
            return nil
        }

        // 解析 fmt chunk
        let sampleRate = Int(data[24..<28].withUnsafeBytes { $0.load(as: UInt32.self) })
        let bitsPerSample = Int(data[34..<36].withUnsafeBytes { $0.load(as: UInt16.self) })
        let numChannels = Int(data[22..<24].withUnsafeBytes { $0.load(as: UInt16.self) })

        // 找 "data" chunk
        var offset = 36
        while offset + 8 <= data.count {
            let chunkID = String(data: data.subdata(in: offset..<offset+4), encoding: .ascii)
            let chunkSize = Int(data[offset+4..<offset+8].withUnsafeBytes { $0.load(as: UInt32.self) })
            if chunkID == "data" {
                var samples = [Float]()
                let rawStart = offset + 8
                let rawEnd = min(rawStart + chunkSize, data.count)

                if bitsPerSample == 16 {
                    for i in stride(from: rawStart, to: rawEnd, by: 2) {
                        let val = Int16(data[i..<i+2].withUnsafeBytes { $0.load(as: Int16.self) })
                        samples.append(Float(val) / 32768.0)
                    }
                } else if bitsPerSample == 32 {
                    for i in stride(from: rawStart, to: rawEnd, by: 4) {
                        let val = Int32(data[i..<i+4].withUnsafeBytes { $0.load(as: Int32.self) })
                        samples.append(Float(val) / 2147483648.0)
                    }
                }

                // 如果多声道，只取左声道
                if numChannels > 1 {
                    var mono = [Float]()
                    for i in stride(from: 0, to: samples.count, by: numChannels) {
                        mono.append(samples[i])
                    }
                    return WAVFile(sampleRate: sampleRate, samples: mono)
                }

                return WAVFile(sampleRate: sampleRate, samples: samples)
            }
            offset += 8 + chunkSize
        }

        return nil
    }
}

