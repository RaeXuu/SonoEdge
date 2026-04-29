import SwiftUI
import BackgroundTasks

@main
struct SonoEdgeApp: App {
    @Environment(\.scenePhase) var scenePhase

    init() {
        BackgroundTaskService.register()
        Task {
            _ = await NotificationService.requestAuthorization()
        }
    }

    var body: some Scene {
        WindowGroup {
            MainTabView()
        }
        .onChange(of: scenePhase) { newPhase in
            switch newPhase {
            case .background:
                BackgroundTaskService.schedule()
            case .active:
                NotificationService.clearBadge()
            default:
                break
            }
        }
    }
}
