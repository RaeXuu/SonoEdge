import SwiftUI

struct MainTabView: View {
    @StateObject private var service = MonitoringService()
    @State private var selectedTab = 0

    var body: some View {
        TabView(selection: $selectedTab) {
            DashboardView(service: service)
                .tabItem {
                    Image(systemName: "heart.fill")
                    Text("监测")
                }
                .tag(0)

            HistoryView(service: service)
                .tabItem {
                    Image(systemName: "clock.arrow.circlepath")
                    Text("记录")
                }
                .tag(1)

            SettingsView(service: service)
                .tabItem {
                    Image(systemName: "person.circle")
                    Text("设置")
                }
                .tag(2)
        }
        .accentColor(.blue)
    }
}
