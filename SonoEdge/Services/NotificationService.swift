import Foundation
import UserNotifications

struct NotificationService {

    private init() {}

    static func requestAuthorization() async -> Bool {
        do {
            return try await UNUserNotificationCenter.current()
                .requestAuthorization(options: [.alert, .sound, .badge])
        } catch {
            print("[Notification] 授权失败: \(error)")
            return false
        }
    }

    static func sendAnomalyAlert(probNormal: Float) async {
        let center = UNUserNotificationCenter.current()

        let settings = await center.notificationSettings()
        guard settings.authorizationStatus == .authorized else { return }

        let content = UNMutableNotificationContent()
        content.title = "心律异常检测"
        content.body = String(format: "检测到异常心音信号 (置信度: %.0f%%)。请查看详情。", (1 - probNormal) * 100)
        content.sound = .default
        content.badge = 1

        let request = UNNotificationRequest(
            identifier: "anomaly-\(Date().timeIntervalSince1970)",
            content: content,
            trigger: nil
        )

        do {
            try await center.add(request)
        } catch {
            print("[Notification] 发送失败: \(error)")
        }
    }

    static func clearBadge() {
        UNUserNotificationCenter.current().removeAllDeliveredNotifications()
    }
}
