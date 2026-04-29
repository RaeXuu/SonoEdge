import Foundation
import TensorFlowLite

/// 对齐 Pi 端 main_pi.py 的 TFLite 推理引擎
/// 支持 INT8 全整型量化模型（输入/输出均为 INT8，需手动量化和反量化）
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
    let modelName: String

    /// 输入形状 [1, 1, 64, 64]
    var inputShape: [Int] { [1, 1, 64, 64] }

    // MARK: - Init

    init(modelPath: String) throws {
        modelName = URL(fileURLWithPath: modelPath).lastPathComponent

        var options = Interpreter.Options()
        options.threadCount = max(1, ProcessInfo.processInfo.activeProcessorCount - 1)
        interpreter = try Interpreter(modelPath: modelPath, options: options)
        try interpreter.allocateTensors()

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

    // MARK: - 量化 / 反量化

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

    // MARK: - 推理

    /// Float32 输入 → 量化 → 推理 → 反量化 → Float32 输出
    /// 对齐 Pi 端 run_inference() 的 per-window 调用
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
}
