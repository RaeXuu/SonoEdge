import Foundation
import Accelerate

// ================================================================
// Aligns with Pi preprocessing pipeline:
//   load_wav → apply_bandpass → segment_audio → normalize → logmel_fixed_size
//
// Corresponds to config.yaml:
//   sr=2000  seg=2.0s  overlap=0.5  bp=25-400Hz
//   mel: n_fft=256  win=256  hop=128  n_mels=64  fmin=25  fmax=400  power=2.0
// ================================================================

// MARK: - Constants

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

// MARK: - Butterworth bandpass filter (order=5, 25-400Hz, zero-phase)
// Aligns with Pi: scipy.signal.butter(5, [25/1000, 400/1000], btype='band', output='ba') + filtfilt

struct ButterworthBandpass {

    /// b, a coefficients for order-5 Butterworth bandpass [25, 400] Hz @ 2000 Hz
    /// Generated via: scipy.signal.butter(5, [25/1000, 400/1000], output='ba')
    /// The Pi uses ba-form (transfer function), NOT SOS — so we match that exactly.
    /// Even though SOS is numerically equivalent, floating-point differences matter for INT8 models.
    private static let b: [Double] = [
         0.016987710409093026,  0.0, -0.08493855204546513,  0.0,
         0.16987710409093026,   0.0, -0.16987710409093026,  0.0,
         0.08493855204546513,   0.0, -0.016987710409093026
    ]
    private static let a: [Double] = [
         1.0, -5.88635597176117, 15.590871175483045, -24.902215524177628,
        27.031963965883904, -21.023701701970058, 11.819133833668117,
        -4.705870557530761, 1.2734414445826752, -0.21324429325710365,
         0.015979779753429305
    ]

    /// Apply zero-phase Butterworth bandpass using scipy's default method='pad'
    /// Matches scipy.signal.filtfilt(b, a, x) — odd extension + lfilter_zi + fwd/bwd + crop
    static func apply(to signal: [Float]) -> [Float] {
        let x = signal.map { Double($0) }
        return filtfiltPad(b: b, a: a, x: x).map { Float($0) }
    }

    // MARK: - IIR internals

    /// Matches scipy.signal.filtfilt(b, a, x) with method='pad' (scipy default).
    /// Odd extension (padlen = 3 * max(len(a), len(b))) → forward lfilter with
    /// zi*ext[0] → reverse lfilter with zi*yFwd[-1] → crop padding.
    private static func filtfiltPad(b: [Double], a: [Double], x: [Double]) -> [Double] {
        let nTaps = max(a.count, b.count)
        let padlen = 3 * nTaps  // = 33 for our filter

        guard x.count > padlen else { return [] }

        // Odd extension (matches scipy.signal._arraytools._pad_odd)
        let n = x.count
        var front = [Double](repeating: 0, count: padlen)
        for i in 0..<padlen {
            front[i] = 2 * x[0] - x[padlen - i]
        }
        var back = [Double](repeating: 0, count: padlen)
        for i in 0..<padlen {
            back[i] = 2 * x[n - 1] - x[n - 2 - i]
        }
        let ext = front + x + back

        // Forward / backward passes with lfilter_zi initial conditions
        let zi = lfilterZi(b: b, a: a)
        let yFwd = lfilterDf2t(b: b, a: a, x: ext, zi: zi.map { $0 * ext[0] })
        let yBwd = lfilterDf2t(b: b, a: a,
                               x: Array(yFwd.reversed()),
                               zi: zi.map { $0 * yFwd[yFwd.count - 1] })
        let y = Array(yBwd.reversed())

        // Crop padding
        return Array(y[padlen ..< (y.count - padlen)])
    }

    /// scipy.signal.lfilter_zi(b, a) — steady-state delay-line for df2t form.
    /// Solves (I − A)·zi = B via back-substitution, where
    ///   B[k]  = b[k+1] − a[k+1]·b[0]
    ///   zi[k] = Σ_{j=k}^{N-1} B[j]  −  zi[0] · Σ_{j=k+1}^{N} a[j]
    ///   zi[0] = Σ B / Σ a
    private static func lfilterZi(b: [Double], a: [Double]) -> [Double] {
        let N = a.count - 1
        var B = [Double](repeating: 0, count: N)
        for k in 0..<N {
            B[k] = (k + 1 < b.count ? b[k + 1] : 0.0) - a[k + 1] * b[0]
        }
        let zi0 = B.reduce(0, +) / a.reduce(0, +)
        var zi = [Double](repeating: 0, count: N)
        for k in 0..<N {
            var cumB = 0.0; for j in k..<N      { cumB += B[j] }
            var cumA = 0.0; for j in (k+1)...N  { cumA += a[j] }
            zi[k] = cumB - zi0 * cumA
        }
        return zi
    }

    /// scipy.signal.lfilter(b, a, x, zi=zi) — direct form II transposed.
    private static func lfilterDf2t(b: [Double], a: [Double], x: [Double], zi: [Double]) -> [Double] {
        let N = a.count - 1
        let len = x.count
        var y = [Double](repeating: 0, count: len)
        var z = zi
        for n in 0..<len {
            y[n] = b[0] * x[n] + z[0]
            for k in 0 ..< N - 1 {
                z[k] = (k + 1 < b.count ? b[k + 1] : 0.0) * x[n] - a[k + 1] * y[n] + z[k + 1]
            }
            z[N - 1] = (N < b.count ? b[N] : 0.0) * x[n] - a[N] * y[n]
        }
        return y
    }
}

// MARK: - Audio segmentation (aligned with segment_audio)

struct AudioSegments {
    /// Sliding window segmentation with seg=2.0s, overlap=0.5
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

// MARK: - Log-Mel spectrogram (aligned with librosa melspectrogram + power_to_db + fix_length)

struct MelSpectrogram {

    /// Single window → (64×64) log-mel spectrogram flattened to [Float]
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

        // Hann window — unnormalized, matching scipy.signal.hann / librosa default
        // (vDSP_HANN_NORM normalizes coherent gain to 1.0, which scales values ~2×;
        //  that causes INT8 input saturation for the diagnosis model)
        var window = [Float](repeating: 0, count: winLen)
        vDSP_hann_window(&window, vDSP_Length(winLen), Int32(vDSP_HANN_DENORM))
        do {  // Debug: print Hann window values once
            struct Once { static var done = false }
            if !Once.done { Once.done = true
                print("[Debug] Hann[0..4]=\(window.prefix(5).map{String(format:"%.6f",$0)}) "
                    + "[126..129]=\(window[126..<130].map{String(format:"%.6f",$0)}) "
                    + "max=\(window.max()!)")
            }
        }

        // FFT setup for N=256 real FFT: vDSP expects log2n=log2(256)=8 (real count).
        // Input must be even/odd interleaved into N/2=128 complex values.
        // vDSP internally unpacks the complex FFT output back to N/2+1 real bins.
        let log2n = vDSP_Length(log2(Float(nFFT)))  // log2(256) = 8
        let fftSetup = vDSP.FFT(log2n: log2n, radix: .radix2,
                                 ofType: DSPSplitComplex.self)!

        var result = [[Float]]()

        for i in 0..<nFrames {
            let start = i * hopLen
            let half = nFFT / 2  // 128

            // vDSP real FFT split-complex input packing (even/odd interleave):
            //   realPart[j] = x[2*j], imagPart[j] = x[2*j+1]
            var realPart = [Float](repeating: 0, count: half)
            var imagPart = [Float](repeating: 0, count: half)
            for j in 0..<half {
                realPart[j] = padded[start + 2 * j]       * window[2 * j]
                imagPart[j] = padded[start + 2 * j + 1]   * window[2 * j + 1]
            }

            realPart.withUnsafeMutableBufferPointer { rp in
                imagPart.withUnsafeMutableBufferPointer { ip in
                    var split = DSPSplitComplex(realp: rp.baseAddress!,
                                                imagp: ip.baseAddress!)
                    fftSetup.forward(input: split, output: &split)
                }
            }

            // Magnitude, then power (|S|^2)
            // vDSP real FFT packs N/2 complex numbers:
            //   DC in realPart[0], Nyquist in imagPart[0]
            //   other bins in (realPart[k], imagPart[k]) for k=1..<N/2
            // vDSP outputs 2x standard DFT magnitude — divide by 2 to match numpy/librosa
            var row = [Float](repeating: 0, count: nFFT / 2 + 1)
            row[0] = pow(abs(realPart[0] / 2.0), kMelPower)
            for k in 1..<(nFFT / 2) {
                let mag = sqrt(realPart[k] * realPart[k] + imagPart[k] * imagPart[k]) / 2.0
                row[k] = pow(mag, kMelPower)
            }
            row[nFFT / 2] = pow(abs(imagPart[0] / 2.0), kMelPower)
            result.append(row)
        }

        do {  // Debug: print first frame STFT once, cross-validate with DebugPreprocess
            struct Once { static var done = false }
            if !Once.done { Once.done = true
                let f0 = result[0]
                print("[Debug] stftLibrosaStyle[0]: first10=\(f0[0..<10].map { String(format: "%.2f", $0) })")
            }
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

        // FFT bin center frequencies: f_k = k * sr / n_fft
        let binHz: [Float] = (0..<nFreqs).map { Float($0) * kSampleRate / Float(kMelFFT) }

        for m in 0..<kMelBands {
            let fLow    = melPoints[m]
            let fCenter = melPoints[m + 1]
            let fHigh   = melPoints[m + 2]

            // Frequency-based triangle weights (matching librosa's np.searchsorted + linear interp)
            for k in 0..<nFreqs {
                let fk = binHz[k]
                if fk <= fLow || fk >= fHigh { continue }

                let denomRise = fCenter - fLow
                let denomFall = fHigh - fCenter

                if fk <= fCenter && denomRise > 0 {
                    fb[m][k] = (fk - fLow) / denomRise
                } else if fk > fCenter && denomFall > 0 {
                    fb[m][k] = (fHigh - fk) / denomFall
                }
            }

            // Slaney normalization (norm='slaney'): enorm = 2.0 / (mel_f[m+2] - mel_f[m])
            let bandwidth = fHigh - fLow
            if bandwidth > 0 {
                let enorm: Float = 2.0 / bandwidth
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

    // ── Mel scale utils (Slaney / librosa htk=False default) ──
    // Below 1000 Hz: linear region. Above: logarithmic.

    private static func hzToMel(_ hz: Float) -> Float {
        if hz < 1000.0 {
            return hz * 3.0 / 200.0
        } else {
            let logstep: Float = logf(6.4) / 27.0
            return 15.0 + logf(hz / 1000.0) / logstep
        }
    }
    private static func melToHz(_ mel: Float) -> Float {
        if mel < 15.0 {
            return mel * 200.0 / 3.0
        } else {
            let logstep: Float = logf(6.4) / 27.0
            return 1000.0 * expf(logstep * (mel - 15.0))
        }
    }
}
