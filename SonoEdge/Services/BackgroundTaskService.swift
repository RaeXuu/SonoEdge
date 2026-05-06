import Foundation
import BackgroundTasks

struct BackgroundTaskService {

    static let taskIdentifier = "com.sonoedge.heart-monitor.refresh"

    static func register() {
        BGTaskScheduler.shared.register(forTaskWithIdentifier: taskIdentifier, using: nil) { task in
            handleAppRefresh(task: task as! BGAppRefreshTask)
        }
    }

    static func schedule() {
        let request = BGAppRefreshTaskRequest(identifier: taskIdentifier)
        request.earliestBeginDate = Date(timeIntervalSinceNow: 5 * 60)
        do {
            try BGTaskScheduler.shared.submit(request)
        } catch {
            print("[BackgroundTask] Schedule failed: \(error)")
        }
    }

    private static func handleAppRefresh(task: BGAppRefreshTask) {
        schedule()

        task.expirationHandler = {
            // Cancel current chunk if needed
        }

        task.setTaskCompleted(success: true)
    }
}
