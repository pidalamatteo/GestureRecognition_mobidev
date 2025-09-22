//
//  AppDelegate.swift
//  mobidev_exam2
//
//  Created by Matteo on 09/09/25.
//
import SwiftUI

// AppDelegate "vecchio stile" se serve per librerie esterne
class AppDelegate: NSObject, UIApplicationDelegate {
    
    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey : Any]? = nil
    ) -> Bool {
            print("AppDelegate did finish launching")
            _ = LandmarkUtils.loadSelectedFeatureIndices()
            LandmarkUtils.verifyNormalizationConsistency()
            LandmarkUtils.verifyFeatureIndices()
            LandmarkUtils.verifyFeatureConsistency()
            
        return true
    }
}

@main
struct MyApp: App {
    @StateObject private var metricsModel = MetricsViewModel()
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(metricsModel)
                .onAppear {
                    if let url = Bundle.main.url(forResource: "metrics", withExtension: "json") {
                        metricsModel.loadMetrics(from: url)
                    }
                }
        }
    }
}


