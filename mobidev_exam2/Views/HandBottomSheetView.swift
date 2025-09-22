import SwiftUI

struct HandBottomSheetView: View {
    @ObservedObject var config: DefaultCostants
    @ObservedObject var handLandmarkerVM: HandLandmarkerViewModel
    var switchCameraAction: (() -> Void)?
    @State private var isExpanded: Bool = true
    
    // Binding sicuri per le proprietà config
    private var numHandsBinding: Binding<Int> {
        Binding(
            get: { config.numHands },
            set: { newValue in
                DispatchQueue.main.async {
                    config.numHands = newValue
                    handLandmarkerVM.updateOptions()
                }
            }
        )
    }
    
    private var minHandPresenceConfidenceBinding: Binding<Double> {
        Binding(
            get: { Double(config.minHandPresenceConfidence) },
            set: { newValue in
                DispatchQueue.main.async {
                    config.minHandPresenceConfidence = Float(newValue)
                    handLandmarkerVM.updateOptions()
                }
            }
        )
    }
    
    private var minHandDetectionConfidenceBinding: Binding<Double> {
        Binding(
            get: { Double(config.minHandDetectionConfidence) },
            set: { newValue in
                DispatchQueue.main.async {
                    config.minHandDetectionConfidence = Float(newValue)
                    handLandmarkerVM.updateOptions()
                }
            }
        )
    }
    
    private var minTrackingConfidenceBinding: Binding<Double> {
        Binding(
            get: { Double(config.minTrackingConfidence) },
            set: { newValue in
                DispatchQueue.main.async {
                    config.minTrackingConfidence = Float(newValue)
                    handLandmarkerVM.updateOptions()
                }
            }
        )
    }
    
    private var minFrameDistanceBinding: Binding<Double> {
        Binding(
            get: { config.minFrameDistance },
            set: { newValue in
                DispatchQueue.main.async {
                    config.minFrameDistance = newValue
                }
            }
        )
    }

    // Funzione per ripristinare tutti i parametri ai valori di default
    private func resetToDefaults() {
        DispatchQueue.main.async {
            // Crea una nuova istanza di DefaultConstants per ottenere i valori di default
            let defaults = DefaultCostants()
            
            // Ripristina i valori prendendoli da DefaultConstants
            config.numHands = defaults.numHands
            config.minHandPresenceConfidence = defaults.minHandPresenceConfidence
            config.minHandDetectionConfidence = defaults.minHandDetectionConfidence
            config.minTrackingConfidence = defaults.minTrackingConfidence
            config.minFrameDistance = defaults.minFrameDistance
            config.delegate = defaults.delegate
            
            // Aggiorna le opzioni del handLandmarkerVM con i nuovi valori
            handLandmarkerVM.updateOptions()
            
            print("Parametri ripristinati ai valori di default da DefaultConstants")
        }
    }

    var body: some View {
        VStack(spacing: 16) {
    
                VStack(spacing: 12) {
                    // Performance metrics
                    HStack{
                        MetricItem(
                            title: "Frame Interval:",
                            value: String(format: "%.2f ms", handLandmarkerVM.frameInterval),
                            valueColor: .orange
                        )
                        
                        Spacer()
                        
                        MetricItem(
                            title: "Inference Time:",
                            value: String(format: "%.2f ms", handLandmarkerVM.inferenceTime),
                            valueColor: .orange
                        )
                    }
                    
                    // Number of hands
                    SettingRow(
                        title: "Number of Hands:",
                        value: "\(config.numHands)",
                        control: Stepper("", value: numHandsBinding, in: 1...4)
                            .labelsHidden()
                    )
                    
                    // Min Hand Presence Confidence
                    SettingRow(
                        title: "Min Hand Presence Confidence:",
                        value: String(format: "%.2f", config.minHandPresenceConfidence),
                        control: Slider(
                            value: minHandPresenceConfidenceBinding,
                            in: 0.05...0.95,
                            step: 0.05
                        )
                    )
                    
                    // Min Hand Detection Confidence
                    SettingRow(
                        title: "Min Hand Detection Confidence:",
                        value: String(format: "%.2f", config.minHandDetectionConfidence),
                        control: Slider(
                            value: minHandDetectionConfidenceBinding,
                            in: 0.05...0.95,
                            step: 0.05
                        )
                    )
                    
                    // Min Tracking Confidence
                    SettingRow(
                        title: "Min Tracking Confidence:",
                        value: String(format: "%.2f", config.minTrackingConfidence),
                        control: Slider(
                            value: minTrackingConfidenceBinding,
                            in: 0.05...1.0,
                            step: 0.05
                        )
                    )
                    
                    // Distance Threshold
                    SettingRow(
                        title: "Distance Threshold:",
                        value: String(format: "%.3f", config.minFrameDistance),
                        control: Slider(
                            value: minFrameDistanceBinding,
                            in: 0.00...0.05,
                            step: 0.005
                        )
                    )
                    
                    // Delegate selection
                    HStack {
                        Text("Delegate:")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        
                        Spacer()
                        
                        Menu {
                            ForEach(HandLandmarkerDelegate.allCases, id: \.self) { delegate in
                                Button(delegate.name) {
                                    DispatchQueue.main.async {
                                        config.delegate = delegate
                                        print("Selected delegate: \(delegate.name)")
                                        handLandmarkerVM.updateOptions()
                                    }
                                }
                            }
                        } label: {
                            HStack {
                                Text(config.delegate.name)
                                    .foregroundColor(.blue)
                                Image(systemName: "chevron.down")
                                    .font(.caption)
                            }
                            .padding(8)
                            .background(Color(.systemGray5))
                            .cornerRadius(6)
                        }
                    }
                    .padding(10)
                    .background(RoundedRectangle(cornerRadius: 8).fill(Color(.systemGray6)))
                    
                    Button(action: {
                        resetToDefaults()
                    }) {
                        Text("Reset to Defaults")
                            .foregroundColor(.white)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(Color.red.opacity(0.8))
                            .cornerRadius(8)
                    }
                    .padding(.top, 8)
                    // Switch camera button
                    Button(action: {
                        DispatchQueue.main.async {
                            switchCameraAction?()
                        }
                    }) {
                        HStack {
                            Image(systemName: "camera.rotate")
                            Text("Switch Camera")
                        }
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.orange)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                    }
                }
                .padding()
                .background(Color.black.opacity(0.4))
                .cornerRadius(10)
            
        }
        //.padding(.horizontal)
    }
}
