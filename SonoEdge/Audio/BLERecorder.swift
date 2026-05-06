import Foundation
import CoreBluetooth
import Combine

// ================================================================
// Aligns with Pi main_pi.py BLE acquisition logic:
//   ESP32 stethoscope → BLE notification → int16 byte stream → 20s chunks
//   Built-in maxsize=1 backpressure queue (drop old chunks when inference falls behind)
//
// ESP32 config (identical to Pi):
//   MAC:  AC:A7:04:85:0D:42
//   UUID: beb5483e-36e1-4688-b7f5-ea07361b26a8
//   Data: int16 PCM @ 2000Hz, one 20s chunk (80000 bytes)
// ================================================================

final class BLERecorder: NSObject, ObservableObject {

    // BLE parameters (aligned with main_pi.py)
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

    // Data buffer
    private var chunkBuffer = Data()

    // maxsize=1 backpressure queue
    private var pendingChunk: Data? = nil
    private let queueLock = NSLock()

    // Callback
    var onChunkReady: ((Data) -> Void)?

    @Published var connectionStatus = "Disconnected"
    @Published var isRunning = false

    override init() {
        super.init()
        centralManager = CBCentralManager(delegate: self, queue: .main)
    }

    // MARK: - Public

    func startScan() {
        connectionStatus = "Scanning..."
        guard centralManager.state == .poweredOn else {
            connectionStatus = "Bluetooth is off"
            return
        }
        // Scan without service UUID filter — some ESP32 BLE stacks don't
        // include the 128-bit UUID in the advertisement in a way iOS recognizes
        centralManager.scanForPeripherals(withServices: nil, options: nil)

        // Connection timeout
        DispatchQueue.main.asyncAfter(deadline: .now() + connectTimeout) { [weak self] in
            guard let self = self, !self.isConnected else { return }
            self.centralManager.stopScan()
            self.connectionStatus = "Connection timeout"
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
        connectionStatus  = "Disconnected"
    }

    /// Called by inference consumer after processing current chunk
    func markChunkConsumed() {
        queueLock.lock()
        pendingChunk = nil
        queueLock.unlock()
    }
}

// MARK: - CBCentralManagerDelegate (scan + connect)

extension BLERecorder: CBCentralManagerDelegate {

    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            connectionStatus = "Bluetooth Ready"
        case .poweredOff:
            connectionStatus = "Bluetooth Off"
        case .unauthorized:
            connectionStatus = "Bluetooth Unauthorized"
        default:
            connectionStatus = "Bluetooth State: \(central.state.rawValue)"
        }
    }

    func centralManager(_ central: CBCentralManager,
                        didDiscover peripheral: CBPeripheral,
                        advertisementData: [String: Any],
                        rssi RSSI: NSNumber) {
        let name = peripheral.name ?? "(unnamed)"

        print("[BLE] Discovered device: name=\(name) id=\(peripheral.identifier.uuidString)")

        guard name == "ESP32_Steth" else { return }

        centralManager.stopScan()

        esp32Peripheral = peripheral
        peripheral.delegate = self
        centralManager.connect(peripheral, options: nil)
        connectionStatus = "Connecting..."
    }

    func centralManager(_ central: CBCentralManager,
                        didConnect peripheral: CBPeripheral) {
        isConnected = true
        connectionStatus = "Connected, discovering services..."
        peripheral.discoverServices([serviceUUID])
    }

    func centralManager(_ central: CBCentralManager,
                        didDisconnectPeripheral peripheral: CBPeripheral,
                        error: Error?) {
        isConnected       = false
        esp32Peripheral   = nil
        dataCharacteristic = nil
        isRunning         = false
        connectionStatus  = error.map { "Disconnected: \($0.localizedDescription)" } ?? "Disconnected"
    }
}

// MARK: - CBPeripheralDelegate (services + characteristics + data notifications)

extension BLERecorder: CBPeripheralDelegate {

    func peripheral(_ peripheral: CBPeripheral,
                    didDiscoverServices error: Error?) {
        if let e = error {
            connectionStatus = "Service discovery failed: \(e.localizedDescription)"
            return
        }
        guard let svc = peripheral.services?.first else {
            connectionStatus = "Target service not found"
            return
        }
        peripheral.discoverCharacteristics([charUUID], for: svc)
    }

    func peripheral(_ peripheral: CBPeripheral,
                    didDiscoverCharacteristicsFor service: CBService,
                    error: Error?) {
        if let e = error {
            connectionStatus = "Characteristic discovery failed: \(e.localizedDescription)"
            return
        }
        guard let chr = service.characteristics?.first(
            where: { $0.uuid.uuidString.caseInsensitiveCompare(charUUIDStr) == .orderedSame }
        ) else {
            // Fallback: use first characteristic with notify property
            guard let fallback = service.characteristics?.first(
                where: { $0.properties.contains(.notify) }
            ) else {
                connectionStatus = "Notify characteristic not found"
                return
            }
            dataCharacteristic = fallback
            peripheral.setNotifyValue(true, for: fallback)
            connectionStatus = "Subscribed to notifications (fallback)"
            return
        }
        dataCharacteristic = chr
        peripheral.setNotifyValue(true, for: chr)
        connectionStatus = "Connected, waiting for data stream..."
    }

    func peripheral(_ peripheral: CBPeripheral,
                    didUpdateValueFor characteristic: CBCharacteristic,
                    error: Error?) {
        guard error == nil, let data = characteristic.value, !data.isEmpty else { return }

        if !isRunning {
            isRunning = true
            connectionStatus = "Collecting..."
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
            connectionStatus = "Subscribe failed: \(e.localizedDescription)"
        }
    }
}

// MARK: - Queue (align main_pi.py line 105-113)

private extension BLERecorder {

    func enqueueChunk(_ chunk: Data) {
        queueLock.lock()
        if pendingChunk != nil {
            print("[BLE] Inference backlog, dropping old chunk")
        }
        pendingChunk = chunk
        let latest = chunk
        queueLock.unlock()

        DispatchQueue.main.async { [weak self] in
            self?.onChunkReady?(latest)
        }
    }
}
