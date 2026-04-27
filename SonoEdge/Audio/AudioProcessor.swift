import Foundation
import Accelerate

// ================================================================
// 对齐 Pi 端预处理流水线：
//   load_wav → apply_bandpass → segment_audio → normalize → logmel_fixed_size
//
// 对应 config.yaml:
//   sr=2000  seg=2.0s  overlap=0.5  bp=25-400Hz
//   mel: n_fft=256  win=256  hop=128  n_mels=64  fmin=25  fmax=400  power=2.0
// ================================================================

// MARK: - 常量

private let kSampleRate: Float    = 2000.0
private let kSegSeconds: Float    = 2.0
private let kOverlap: Float       = 0.5
private let kSegSamples: Int      = Int(kSampleRate * kSegSeconds)       // 4000
private let kHopSamples: Int      = Int(Float(kSegSamples) * (1 - kOverlap)) // 2000

private let kMelFFT: Int           = 256
private let kMelWinLen: Int        = 256
private let kMelHopLen: Int        = 128
private let kMelBands: Int         = 64
private let kMelFMin: Float        = 25.0
private let kMelFMax: Float        = 400.0
private let kMelTargetFrames: Int  = 64
private let kMelPower: Float       = 2.0
private let kMelEps: Float         = 1e-6

// MARK: - Butterworth 带通滤波器 (order=5, 25-400Hz, zero-phase)

struct ButterworthBandpass {

    /// Pre-computed SOS coefficients for order-5 Butterworth bandpass [25, 400] Hz @ 2000 Hz
    /// Generated via: scipy.signal.butter(5, [25/1000, 400/1000], btype='band', output='sos')
    /// Each row: [b0, b1, b2, a0, a1, a2]  (a0 is always 1.0)
    private static let sos: [[Double]] = [
        [0.016987710409093026,  0.03397542081818605,   0.016987710409093026,  1.0, -0.43124111681185834,  0.16460451389960484],
        [1.0,                   2.0,                   1.0,                   1.0, -0.5007275936949009,   0.5809583354693726],
        [1.0,                   0.0,                  -1.0,                   1.0, -1.132363909413649,    0.19891236737965803],
        [1.0,                  -2.0,                   1.0,                   1.0, -1.8714268996406815,   0.8780755546064518],
        [1.0,                  -2.0,                   1.0,                   1.0, -1.9505964522000812,   0.9567321946304889],
    ]

    /// Apply zero-phase Butterworth bandpass (forward-backward filtfilt)
    /// Equivalent to scipy.signal.sosfiltfilt(sos, x)
    static func apply(to signal: [Float]) -> [Float] {
        // Convert to Double for numerical stability
        let input = signal.map { Double($0) }
        let forward = sosfilt(sos: sos, x: input)
        let reversed = Array(forward.reversed())
        let backward = sosfilt(sos: sos, x: reversed)
        let result = Array(backward.reversed())

        return result.map { Float($0) }
    }

    // MARK: - IIR internals

    /// Apply second-order sections filter (direct form II transposed)
    /// Equivalent to scipy.signal.sosfilt(sos, x)
    private static func sosfilt(sos: [[Double]], x: [Double]) -> [Double] {
        var y = x
        for section in sos {
            let b0 = section[0], b1 = section[1], b2 = section[2]
            let a1 = section[4], a2 = section[5]

            var w1: Double = 0, w2: Double = 0
            for i in 0..<y.count {
                let w0 = y[i] - a1 * w1 - a2 * w2
                y[i] = b0 * w0 + b1 * w1 + b2 * w2
                w2 = w1
                w1 = w0
            }
        }
        return y
    }
}

// MARK: - 音频分段 (对齐 segment_audio)

struct AudioSegments {
    /// 按 seg=2.0s, overlap=0.5 滑动切片
    static func extract(from signal: [Float]) -> [[Float]] {
        var segs = [[Float]]()
        var start = 0
        let total = signal.count
        while start + kSegSamples <= total {
            segs.append(Array(signal[start..<start + kSegSamples]))
            start += kHopSamples
        }
        return segs
    }
}

// MARK: - Log-Mel 频谱 (对齐 librosa melspectrogram + power_to_db + fix_length)

struct MelSpectrogram {

    /// 单窗口 → (64×64) log-mel 频谱展平为 [Float]
    /// Params match config.yaml + librosa defaults (center=True, hann window)
    static func compute(from window: [Float]) -> [Float] {
        // 1. peak normalize (per-window, as in main_pi.py)
        var w = window
        let mx = window.map(abs).max() ?? 1.0
        if mx > 0 {
            w = window.map { $0 / mx }
        }

        // 2. STFT with center=True (pad n_fft/2 zeros on both sides, matching librosa.stft)
        let spec = stftLibrosaStyle(signal: w)

        // 3. Apply mel filterbank
        let melFB = cachedMelFilterbank
        let melSpec = applyMelFB(spec: spec, filterbank: melFB)

        // 4. Power to dB: 10 * log10(val + eps) + top_db clamp — matches librosa.power_to_db
        var logMel = melSpec.map {
            $0.map { 10.0 * log10(max($0, kMelEps)) }
        }

        // top_db=80 clipping (librosa.power_to_db default)
        let globalMax = logMel.flatMap { $0 }.max() ?? 0
        logMel = logMel.map { $0.map { max($0, globalMax - 80.0) } }

        // 5. fix_length to 64 frames (pad or truncate along time axis)
        logMel = fixLength(mel: logMel, targetFrames: kMelTargetFrames)

        // 6. Flatten row-major → [Float] of length 4096
        return logMel.flatMap { $0 }
    }

    // ── STFT (librosa-compatible, center=True) ──

    private static func stftLibrosaStyle(signal: [Float]) -> [[Float]] {
        let nFFT   = kMelFFT
        let hopLen = kMelHopLen
        let winLen = kMelWinLen
        let pad    = nFFT / 2  // 128

        // Pad both ends
        let frontPad = [Float](repeating: 0, count: pad)
        let backPad  = [Float](repeating: 0, count: pad)
        let padded = frontPad + signal + backPad

        let nFrames = 1 + (padded.count - nFFT) / hopLen

        // Hann window
        var window = [Float](repeating: 0, count: winLen)
        vDSP_hann_window(&window, vDSP_Length(winLen), Int32(vDSP_HANN_NORM))

        // FFT setup (create once, reuse for all frames)
        let log2n = vDSP_Length(log2(Float(nFFT)))
        let fftSetup = vDSP.FFT(log2n: log2n, radix: .radix2,
                                 ofType: DSPSplitComplex.self)!

        var result = [[Float]]()

        for i in 0..<nFrames {
            let start = i * hopLen
            var frame = [Float](repeating: 0, count: nFFT)
            for j in 0..<winLen {
                frame[j] = padded[start + j] * window[j]
            }

            var realPart = [Float](repeating: 0, count: nFFT / 2)
            var imagPart = [Float](repeating: 0, count: nFFT / 2)

            realPart.withUnsafeMutableBufferPointer { rp in
                imagPart.withUnsafeMutableBufferPointer { ip in
                    var split = DSPSplitComplex(realp: rp.baseAddress!,
                                                imagp: ip.baseAddress!)
                    frame.withUnsafeBytes { ptr in
                        let typed = ptr.bindMemory(to: DSPComplex.self)
                        vDSP_ctoz(typed.baseAddress!, 2, &split, 1,
                                  vDSP_Length(nFFT / 2))
                    }
                    fftSetup.forward(input: split, output: &split)
                }
            }

            // Magnitude, then power (|S|^2)
            var row = [Float](repeating: 0, count: nFFT / 2 + 1)
            for k in 0..<(nFFT / 2 + 1) {
                let mag = sqrt(realPart[k] * realPart[k] + imagPart[k] * imagPart[k])
                row[k] = pow(mag, kMelPower)
            }
            result.append(row)
        }

        return result  // (nFrames, freqBins)
    }

    // ── Mel Filterbank (cached, parameters are compile-time constants) ──

    private static let cachedMelFilterbank: [[Float]] = {
        let nFreqs = kMelFFT / 2 + 1  // 129
        var fb = [[Float]](repeating: [Float](repeating: 0, count: nFreqs),
                           count: kMelBands)

        let melMin = hzToMel(kMelFMin)
        let melMax = hzToMel(kMelFMax)
        let melPoints = (0..<(kMelBands + 2)).map { i -> Float in
            melToHz(melMin + (melMax - melMin) * Float(i) / Float(kMelBands + 1))
        }
        let bins = melPoints.map { f -> Int in
            Int(floor(Float(nFreqs) * f / (kSampleRate / 2)))
        }

        for m in 0..<kMelBands {
            let start = bins[m]
            let center = bins[m + 1]
            let end   = min(bins[m + 2], nFreqs - 1)

            for k in start..<center where center > start {
                fb[m][k] = Float(k - start) / Float(center - start)
            }
            for k in center...end where end > center {
                fb[m][k] = Float(end - k) / Float(end - center)
            }

            // Slaney normalization (norm='slaney'): divide by bandwidth in Hz
            // Matches librosa 0.11.0 default: enorm = 2.0 / (mel_f[m+2] - mel_f[m])
            let bandwidth = melPoints[m + 2] - melPoints[m]
            if bandwidth > 0 {
                let enorm = 2.0 / bandwidth
                for k in 0..<nFreqs {
                    fb[m][k] *= enorm
                }
            }
        }
        return fb
    }()

    /// Apply mel filterbank: melFB (nMels × nFreqs) @ spec (nFreqs × nFrames)
    private static func applyMelFB(spec: [[Float]], filterbank fb: [[Float]]) -> [[Float]] {
        let nFrames  = spec.count
        let nFreqs   = spec[0].count
        var result = [[Float]](repeating: [Float](repeating: 0, count: nFrames),
                                count: kMelBands)

        for m in 0..<kMelBands {
            for t in 0..<nFrames {
                var sum: Float = 0
                for k in 0..<nFreqs {
                    sum += fb[m][k] * spec[t][k]
                }
                result[m][t] = sum
            }
        }
        return result  // (nMels, nFrames)
    }

    // ── Fix length ──

    private static func fixLength(mel: [[Float]], targetFrames: Int) -> [[Float]] {
        let currentFrames = mel[0].count
        if currentFrames == targetFrames { return mel }

        return mel.map { row in
            if currentFrames > targetFrames {
                return Array(row.prefix(targetFrames))
            } else {
                return row + [Float](repeating: 0, count: targetFrames - currentFrames)
            }
        }
    }

    // ── Mel scale utils ──

    private static func hzToMel(_ hz: Float) -> Float {
        2595.0 * log10(1.0 + hz / 700.0)
    }
    private static func melToHz(_ mel: Float) -> Float {
        700.0 * (pow(10.0, mel / 2595.0) - 1.0)
    }
}
