/*import SwiftUI

class MetricsViewModel: ObservableObject {
    @Published var metricsStruct: MetricsStruct?
    @Published var isLoading: Bool = false
    @Published var errorMessage: String?
    @Published var selectedTab: Int = 0
    
    
    func loadMetrics(from url: URL) {
        isLoading = true
        errorMessage = nil
   
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let data = try Data(contentsOf: url)
                let metrics = try JSONDecoder().decode(MetricsStruct.self, from: data)
                
                DispatchQueue.main.async {
                    self.metricsStruct = metrics
                    self.isLoading = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = error.localizedDescription
                    self.isLoading = false
                }
            }
        }
    }
}
*/
