import Foundation
import CoreBluetooth
import Combine

// ================================================================
// 对齐 Pi 端 main_pi.py BLE 采集逻辑：
//   ESP32 电子听诊器 → BLE notification → int16 字节流 → 20s 分块
//   内建 maxsize=1 反压队列 (推理积压时丢弃旧块)
//
// ESP32 配置 (与 Pi 端完全一致):
//   MAC:  AC:A7:04:85:0D:42
//   UUID: beb5483e-36e1-4688-b7f5-ea07361b26a8
//   数据: int16 PCM @ 2000Hz, 每 20s 一块 (80000 bytes)
// ================================================================

final class BLERecorder: NSObject, ObservableObject {

    // BLE 参数 (对齐 main_pi.py)
    private let serviceUUID   = CBUUID(string: "4fafc201-1fb5-459e-8fcc-c5c9c331914b")
    private let charUUIDStr   = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
    private let charUUID      = CBUUID(string: "beb5483e-36e1-4688-b7f5-ea07361b26a8")
    private let connectTimeout: TimeInterval = 15.0
    private let sampleRate: Double    = 2000.0
    private let chunkDuration: Double = 20.0
    private let chunkBytes: Int       = 80000   // 40000 samples × 2 bytes

    // CoreBluetooth
    private var centralManager: CBCentralManager!
    private var esp32Peripheral: CBPeripheral?
    private var dataCharacteristic: CBCharacteristic?
    private var isConnected = false

    // 数据缓冲
    private var chunkBuffer = Data()

    // maxsize=1 反压队列
    private var pendingChunk: Data? = nil
    private let queueLock = NSLock()

    // 回调
    var onChunkReady: ((Data) -> Void)?

    @Published var connectionStatus = "未连接"
    @Published var isRunning = false

    override init() {
        super.init()
        centralManager = CBCentralManager(delegate: self, queue: .main)
    }

    // MARK: - Public

    func startScan() {
        connectionStatus = "扫描中..."
        guard centralManager.state == .poweredOn else {
            connectionStatus = "蓝牙未开启"
            return
        }
        // Scan for ESP32 by its service UUID (efficient + reliable)
        centralManager.scanForPeripherals(withServices: [serviceUUID], options: nil)

        // Connection timeout
        DispatchQueue.main.asyncAfter(deadline: .now() + connectTimeout) { [weak self] in
            guard let self = self, !self.isConnected else { return }
            self.centralManager.stopScan()
            self.connectionStatus = "连接超时"
        }
    }

    func disconnect() {
        centralManager.stopScan()
        if let p = esp32Peripheral, let c = dataCharacteristic {
            p.setNotifyValue(false, for: c)
        }
        if let p = esp32Peripheral {
            centralManager.cancelPeripheralConnection(p)
        }
        isConnected       = false
        isRunning         = false
        esp32Peripheral   = nil
        dataCharacteristic = nil
        chunkBuffer       = Data()
        connectionStatus  = "已断开"
    }

    /// 推理端消费完当前块后调用
    func markChunkConsumed() {
        queueLock.lock()
        pendingChunk = nil
        queueLock.unlock()
    }
}

// MARK: - CBCentralManagerDelegate (扫描 + 连接)

extension BLERecorder: CBCentralManagerDelegate {

    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            connectionStatus = "蓝牙就绪"
        case .poweredOff:
            connectionStatus = "蓝牙已关闭"
        case .unauthorized:
            connectionStatus = "蓝牙未授权"
        default:
            connectionStatus = "蓝牙状态: \(central.state.rawValue)"
        }
    }

    func centralManager(_ central: CBCentralManager,
                        didDiscover peripheral: CBPeripheral,
                        advertisementData: [String: Any],
                        rssi RSSI: NSNumber) {
        let name = peripheral.name ?? "(unnamed)"

        print("[BLE] 发现 ESP32: name=\(name) id=\(peripheral.identifier.uuidString)")
        centralManager.stopScan()

        esp32Peripheral = peripheral
        peripheral.delegate = self
        centralManager.connect(peripheral, options: nil)
        connectionStatus = "连接中..."
    }

    func centralManager(_ central: CBCentralManager,
                        didConnect peripheral: CBPeripheral) {
        isConnected = true
        connectionStatus = "已连接, 搜索服务..."
        peripheral.discoverServices([serviceUUID])
    }

    func centralManager(_ central: CBCentralManager,
                        didDisconnectPeripheral peripheral: CBPeripheral,
                        error: Error?) {
        isConnected       = false
        esp32Peripheral   = nil
        dataCharacteristic = nil
        isRunning         = false
        connectionStatus  = error.map { "断开: \($0.localizedDescription)" } ?? "已断开"
    }
}

// MARK: - CBPeripheralDelegate (服务 + 特征 + 数据通知)

extension BLERecorder: CBPeripheralDelegate {

    func peripheral(_ peripheral: CBPeripheral,
                    didDiscoverServices error: Error?) {
        if let e = error {
            connectionStatus = "服务发现失败: \(e.localizedDescription)"
            return
        }
        guard let svc = peripheral.services?.first else {
            connectionStatus = "未找到目标服务"
            return
        }
        peripheral.discoverCharacteristics([charUUID], for: svc)
    }

    func peripheral(_ peripheral: CBPeripheral,
                    didDiscoverCharacteristicsFor service: CBService,
                    error: Error?) {
        if let e = error {
            connectionStatus = "特征发现失败: \(e.localizedDescription)"
            return
        }
        guard let chr = service.characteristics?.first(
            where: { $0.uuid.uuidString.caseInsensitiveCompare(charUUIDStr) == .orderedSame }
        ) else {
            // Fallback: 取第一个 notify 属性特征
            guard let fallback = service.characteristics?.first(
                where: { $0.properties.contains(.notify) }
            ) else {
                connectionStatus = "未找到 notify 特征"
                return
            }
            dataCharacteristic = fallback
            peripheral.setNotifyValue(true, for: fallback)
            connectionStatus = "已订阅通知 (fallback)"
            return
        }
        dataCharacteristic = chr
        peripheral.setNotifyValue(true, for: chr)
        connectionStatus = "已连接, 等待数据流..."
    }

    func peripheral(_ peripheral: CBPeripheral,
                    didUpdateValueFor characteristic: CBCharacteristic,
                    error: Error?) {
        guard error == nil, let data = characteristic.value, !data.isEmpty else { return }

        if !isRunning {
            isRunning = true
            connectionStatus = "采集中..."
        }

        chunkBuffer.append(data)

        while chunkBuffer.count >= chunkBytes {
            let chunk = Data(chunkBuffer.prefix(chunkBytes))
            chunkBuffer = Data(chunkBuffer.dropFirst(chunkBytes))
            enqueueChunk(chunk)
        }
    }

    func peripheral(_ peripheral: CBPeripheral,
                    didUpdateNotificationStateFor characteristic: CBCharacteristic,
                    error: Error?) {
        if let e = error {
            connectionStatus = "订阅失败: \(e.localizedDescription)"
        }
    }
}

// MARK: - Queue (align main_pi.py line 105-113)

private extension BLERecorder {

    func enqueueChunk(_ chunk: Data) {
        queueLock.lock()
        if pendingChunk != nil {
            print("[BLE] 推理积压，丢弃旧块")
        }
        pendingChunk = chunk
        let latest = chunk
        queueLock.unlock()

        DispatchQueue.main.async { [weak self] in
            self?.onChunkReady?(latest)
        }
    }
}
