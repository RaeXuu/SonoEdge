import Foundation
import Darwin
import TensorFlowLite

/// Aligns with Pi main_pi.py TFLite inference engine
/// Supports INT8 full-integer quantized models (input/output are INT8, manual quantize/dequantize required)
final class TFLiteEngine {
    private let interpreter: Interpreter
    private let inputIndex: Int
    private let outputIndex: Int

    let inputScale: Float
    let inputZeroPoint: Int
    let outputScale: Float
    let outputZeroPoint: Int
    let isInt8Input: Bool
    let isInt8Output: Bool
    let modelPath: String
    let modelName: String

    /// Input shape [1, 1, 64, 64]
    var inputShape: [Int] { [1, 1, 64, 64] }

    // MARK: - Init

    init(modelPath: String, threadCount: Int? = nil) throws {
        self.modelPath = modelPath
        modelName = URL(fileURLWithPath: modelPath).lastPathComponent

        let t0 = CFAbsoluteTimeGetCurrent()

        var options = Interpreter.Options()
        options.threadCount = threadCount ?? max(1, ProcessInfo.processInfo.activeProcessorCount - 1)
        interpreter = try Interpreter(modelPath: modelPath, options: options)
        try interpreter.allocateTensors()

        let loadMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
        print("[TFLite] \(modelName) load_time=\(String(format: "%.2f", loadMs))ms")

        inputIndex  = 0
        outputIndex = 0

        let inputTensor  = try interpreter.input(at: inputIndex)
        let outputTensor = try interpreter.output(at: outputIndex)

        let iqp = inputTensor.quantizationParameters
        inputScale     = iqp?.scale ?? 1.0
        inputZeroPoint = iqp?.zeroPoint ?? 0
        isInt8Input    = (iqp != nil)

        let oqp = outputTensor.quantizationParameters
        outputScale     = oqp?.scale ?? 1.0
        outputZeroPoint = oqp?.zeroPoint ?? 0
        isInt8Output    = (oqp != nil)

        print("[TFLite] \(modelName)"
            + "  in=\(isInt8Input ? "int8" : "float32")"
            + "  out=\(isInt8Output ? "int8" : "float32")"
            + "  in_scale=\(String(format: "%.6f", inputScale))"
            + "  in_zp=\(inputZeroPoint)"
            + "  out_scale=\(String(format: "%.6f", outputScale))"
            + "  out_zp=\(outputZeroPoint)")
    }

    // MARK: - Warmup

    func warmup(count: Int = 5) throws {
        let elementCount = inputShape.reduce(1, *)
        for _ in 0..<count {
            let randomInt8 = (0..<elementCount).map { _ in Int8(Int.random(in: -128...127)) }
            let data = randomInt8.withUnsafeBufferPointer { Data(buffer: $0) }
            try interpreter.copy(data, toInputAt: inputIndex)
            try interpreter.invoke()
            _ = try interpreter.output(at: outputIndex)
        }
    }

    // MARK: - Benchmark

    /// Full single-inference flow aligned with Pi bench_model():
    ///   copy input → invoke → read output tensor → dequantize (if INT8)
    /// Input data pre-quantized once outside loop, re-copied each iteration (matches Pi set_tensor behavior).
    func benchmark(iterations: Int = 100) throws {
        try warmup(count: 10)

        let elementCount = inputShape.reduce(1, *)
        let randomInt8 = (0..<elementCount).map { _ in Int8(Int.random(in: -128...127)) }
        let inputData = randomInt8.withUnsafeBufferPointer { Data(buffer: $0) }

        let rssBefore = Self.getResidentMemory()
        let cpuBefore = Self.getTaskThreadTimes()

        var samples: [Double] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let t0 = CFAbsoluteTimeGetCurrent()

            // 1. Write input (aligns with Pi: set_tensor)
            try interpreter.copy(inputData, toInputAt: inputIndex)
            // 2. Inference
            try interpreter.invoke()
            // 3. Read output (aligns with Pi: get_tensor)
            let outputTensor = try interpreter.output(at: outputIndex)
            // 4. Dequantize (aligns with Pi: if is_int8_out: _ = (raw - zp) * scale)
            if isInt8Output {
                let count = outputTensor.shape.dimensions.reduce(1, *)
                _ = outputTensor.data.withUnsafeBytes { ptr -> [Float] in
                    ptr.bindMemory(to: Int8.self).prefix(count).map { i8 in
                        (Float(i8) - Float(outputZeroPoint)) * outputScale
                    }
                }
            }

            samples.append((CFAbsoluteTimeGetCurrent() - t0) * 1000.0)
        }

        let cpuAfter = Self.getTaskThreadTimes()
        let rssAfter = Self.getResidentMemory()

        samples.sort()
        let median = samples[iterations / 2]
        let p95 = samples[Int(Double(iterations) * 0.95)]
        let minV = samples.first ?? 0
        let maxV = samples.last ?? 0
        let mean = samples.reduce(0, +) / Double(iterations)

        let wallMs = Double(iterations) * median
        let cpuUserDeltaUs = cpuAfter.user - cpuBefore.user
        let cpuSysDeltaUs  = cpuAfter.system - cpuBefore.system
        let cpuPct = wallMs > 0 ? (Double(cpuUserDeltaUs + cpuSysDeltaUs) / 1000.0 / wallMs * 100.0) : 0
        let rssDelta = Int64(rssAfter) - Int64(rssBefore)

        print("[TFLite:bench] \(modelName)  iters=\(iterations)"
            + "  median=\(String(format: "%.3f", median))ms"
            + "  p95=\(String(format: "%.3f", p95))ms"
            + "  mean=\(String(format: "%.3f", mean))ms"
            + "  min=\(String(format: "%.3f", minV))ms"
            + "  max=\(String(format: "%.3f", maxV))ms"
            + "  threads=\(ProcessInfo.processInfo.activeProcessorCount - 1)")
        print("[TFLite:bench] \(modelName)  rss_before=\(rssBefore)B  rss_after=\(rssAfter)B"
            + "  rss_delta=\(rssDelta > 0 ? "+" : "")\(rssDelta)B"
            + "  cpu_user=\(cpuUserDeltaUs)us  cpu_sys=\(cpuSysDeltaUs)us  cpu_pct=\(String(format: "%.1f", cpuPct))%")
    }

    // MARK: - Quantize / Dequantize

    private func quantizeInput(_ floatData: [Float]) -> [Int8] {
        return floatData.map { f in
            let q = (f / inputScale) + Float(inputZeroPoint)
            return Int8(max(-128, min(127, q.rounded())))
        }
    }

    private func dequantizeOutput(_ int8Data: [Int8]) -> [Float] {
        return int8Data.map { i8 in
            (Float(i8) - Float(outputZeroPoint)) * outputScale
        }
    }

    // MARK: - Debug helpers

#if DEBUG
    private func softmaxDebug(_ x: [Float]) -> [Float] {
        let maxX = x.max() ?? 0
        let expX = x.map { exp($0 - maxX) }
        let sum = expX.reduce(0, +)
        return expX.map { $0 / sum }
    }
#endif

    // MARK: - Inference

    /// Float32 input → quantize → inference → dequantize → Float32 output
    /// Aligns with Pi run_inference() per-window call
    func infer(floatInput: [Float]) throws -> [Float] {
        var quantizedInput: [Int8]? = nil

        if isInt8Input {
            let quantized = quantizeInput(floatInput)
            quantizedInput = quantized
            let data = quantized.withUnsafeBufferPointer { Data(buffer: $0) }
            try interpreter.copy(data, toInputAt: inputIndex)
        } else {
            let data = floatInput.withUnsafeBufferPointer { Data(buffer: $0) }
            try interpreter.copy(data, toInputAt: inputIndex)
        }

        try interpreter.invoke()

        let outputTensor = try interpreter.output(at: outputIndex)
        let count = outputTensor.shape.dimensions.reduce(1, *)

        if isInt8Output {
            let raw = outputTensor.data.withUnsafeBytes {
                Array($0.bindMemory(to: Int8.self).prefix(count))
            }
            let dequant = dequantizeOutput(raw)

#if DEBUG
            let isDiag = modelName.contains("heart_model")
            if isDiag, let qi = quantizedInput {
                let qiMin = qi.map { Int($0) }.min() ?? 0
                let qiMax = qi.map { Int($0) }.max() ?? 0
                let clipNeg = qi.filter { $0 == -128 }.count
                let clipPos = qi.filter { $0 == 127 }.count
                print("[TFLite:\(modelName)] quant_input: min=\(qiMin) max=\(qiMax) clip_neg=\(clipNeg) clip_pos=\(clipPos) (of \(qi.count))")
                print("[TFLite:\(modelName)] float_input_range: [\(floatInput.min()!), \(floatInput.max()!)]")
                print("[TFLite:\(modelName)] raw_int8=\(raw.prefix(4))")
                print("[TFLite:\(modelName)] dequant_logits=\(dequant.prefix(4).map { String(format: "%.6f", $0) })")
                let softmaxed = softmaxDebug(dequant)
                print("[TFLite:\(modelName)] softmax=\(softmaxed.map { String(format: "%.6f", $0) })")
            }
#endif

            return dequant
        } else {
            return outputTensor.data.withUnsafeBytes {
                Array($0.bindMemory(to: Float.self).prefix(count))
            }
        }
    }

    // MARK: - Single-thread sanity check

    /// Single-threaded version, flow fully aligned with benchmark() (copy → invoke → read → dequant).
    func benchmarkSingleThread(iterations: Int = 100) throws {
        print("[TFLite:bench-1t] \(modelName)  creating single-thread interpreter...")
        var options = Interpreter.Options()
        options.threadCount = 1
        let stInterpreter = try Interpreter(modelPath: modelPath, options: options)
        try stInterpreter.allocateTensors()

        let elementCount = inputShape.reduce(1, *)
        let randomInt8 = (0..<elementCount).map { _ in Int8(Int.random(in: -128...127)) }
        let inputData = randomInt8.withUnsafeBufferPointer { Data(buffer: $0) }

        // Warmup
        for _ in 0..<10 {
            try stInterpreter.copy(inputData, toInputAt: 0)
            try stInterpreter.invoke()
            _ = try stInterpreter.output(at: 0)
        }

        // Benchmark loop (aligns with Pi bench_model)
        var samples: [Double] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let t0 = CFAbsoluteTimeGetCurrent()

            try stInterpreter.copy(inputData, toInputAt: 0)
            try stInterpreter.invoke()
            let outputTensor = try stInterpreter.output(at: 0)
            if isInt8Output {
                let count = outputTensor.shape.dimensions.reduce(1, *)
                _ = outputTensor.data.withUnsafeBytes { ptr -> [Float] in
                    ptr.bindMemory(to: Int8.self).prefix(count).map { i8 in
                        (Float(i8) - Float(outputZeroPoint)) * outputScale
                    }
                }
            }

            samples.append((CFAbsoluteTimeGetCurrent() - t0) * 1000.0)
        }

        samples.sort()
        let median = samples[iterations / 2]
        let p95 = samples[Int(Double(iterations) * 0.95)]
        let minV = samples.first ?? 0
        let maxV = samples.last ?? 0
        let mean = samples.reduce(0, +) / Double(iterations)

        print("[TFLite:bench-1t] \(modelName)  iters=\(iterations)"
            + "  median=\(String(format: "%.3f", median))ms"
            + "  p95=\(String(format: "%.3f", p95))ms"
            + "  mean=\(String(format: "%.3f", mean))ms"
            + "  min=\(String(format: "%.3f", minV))ms"
            + "  max=\(String(format: "%.3f", maxV))ms"
            + "  threads=1")
    }

    // MARK: - System metrics helpers

    /// Current process resident memory (bytes)
    static func getResidentMemory() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return info.resident_size
    }

    /// Current process accumulated CPU time (microseconds), returns (user, system)
    static func getTaskThreadTimes() -> (user: UInt64, system: UInt64) {
        var info = task_thread_times_info()
        var count = mach_msg_type_number_t(MemoryLayout<task_thread_times_info>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(TASK_THREAD_TIMES_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return (0, 0) }
        let userUs = UInt64(info.user_time.seconds) * 1_000_000 + UInt64(info.user_time.microseconds)
        let sysUs  = UInt64(info.system_time.seconds) * 1_000_000 + UInt64(info.system_time.microseconds)
        return (userUs, sysUs)
    }
}
