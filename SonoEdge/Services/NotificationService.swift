import Foundation
import UserNotifications

struct NotificationService {

    private init() {}

    static func requestAuthorization() async -> Bool {
        do {
            return try await UNUserNotificationCenter.current()
                .requestAuthorization(options: [.alert, .sound, .badge])
        } catch {
            print("[Notification] Authorization failed: \(error)")
            return false
        }
    }

    static func sendAnomalyAlert(probNormal: Float) async {
        let center = UNUserNotificationCenter.current()

        let settings = await center.notificationSettings()
        guard settings.authorizationStatus == .authorized else { return }

        let content = UNMutableNotificationContent()
        content.title = "Abnormal Heart Rhythm Detected"
        content.body = String(format: "Abnormal heart sound detected (confidence: %.0f%%). Please check details.", (1 - probNormal) * 100)
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
            print("[Notification] Send failed: \(error)")
        }
    }

    static func clearBadge() {
        UNUserNotificationCenter.current().removeAllDeliveredNotifications()
    }
}
