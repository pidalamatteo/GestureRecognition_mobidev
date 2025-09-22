import SwiftUI
import AVFoundation

struct CameraView: View {
    @StateObject private var cameraManager = CameraManager()
    @StateObject private var handLandmarkerVM = HandLandmarkerViewModel()
    @StateObject private var gesturePredictor = GesturePredictor()
    @State private var selectedTab: Int = 0
    @State private var isSheetExpanded = false
    @State private var hasLoadedInitialMetrics = false
    @State private var contentHeight: CGFloat = 0

    // Definizione dei colori per ogni tab
    let tabColors: [Color] = [
        .red,      // Registrazione
        .green,    // Predizione
        .orange,   // Mano
        .purple    // Metriche
    ]
    
    let tabIcons = ["record.circle", "chart.bar", "hand.raised", "gauge"]
    let tabTitles = ["Record", "Prediction", "Hand", "Metrics"]
    
    var body: some View {
        ZStack {
            if let session = cameraManager.session {
                CameraPreviewView(session: session)
                    .ignoresSafeArea()
            } else {
                Text("Nessuna fotocamera disponibile")
            }

            if let imageSize = handLandmarkerVM.currentImageSize {
                ZStack {
                    HandOverlayView(
                        config: handLandmarkerVM.config,
                        hands: handLandmarkerVM.detectedHands,
                        originalImageSize: imageSize,
                        isPreviewMirrored: cameraManager.cameraPosition == .front
                    )
                    .ignoresSafeArea()
                    
                    GeometryReader { geo in
                        ZStack {
                            // Draw a bounding box and label for each detected hand (up to 2)
                            ForEach(Array(handLandmarkerVM.detectedHands.prefix(2).enumerated()), id: \.offset) { index, hand in
                                if let box = handLandmarkerVM.boundingBox(
                                    for: hand,
                                    originalSize: handLandmarkerVM.currentImageSize!,
                                    viewSize: geo.size,
                                    isBackCamera: cameraManager.cameraPosition == .back
                                ) {
                                    HandBBoxOverlay(vm: handLandmarkerVM, index: index, box: box, containerSize: geo.size)
                                }
                            }
                        }
                    }
                    .ignoresSafeArea()
                }
            }

            // Tab bar in basso
            VStack(spacing: 0) {
                Spacer()
                
                ZStack(alignment: .bottom) {
                    // Sfondo colorato per la tab selezionata
                    VStack {
                        RoundedRectangle(cornerRadius: 15)
                            .fill(tabColors[selectedTab].opacity(0.3))
                            .mask(
                                LinearGradient(
                                    gradient: Gradient(colors: [.clear, .black, .black]),
                                    startPoint: .top,
                                    endPoint: .bottom
                                )
                            )
                            .frame(maxHeight: UIScreen.main.bounds.height * 0.2)
                            .offset(y: isSheetExpanded ? 0 : 300)
                            .opacity(isSheetExpanded ? 1 : 0)
                    }
                    
                    // Pulsanti delle tab
                    VStack {
                        HStack {
                            ForEach(0..<4) { index in
                                TabButton(
                                    title: tabTitles[index],
                                    icon: tabIcons[index],
                                    isSelected: selectedTab == index,
                                    color: tabColors[index],
                                    action: { selectTab(index) }
                                )
                            }
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .background(Color(.systemBackground).opacity(0.9))
                        .cornerRadius(15)
                    }
                    .shadow(radius: 5)
                    .padding(.horizontal)
                    .padding(.bottom, 10)
                    .zIndex(1)
                    
                    // Contenuto delle tab
                    VStack {
                        ScrollView {
                            VStack {
                                Group {
                                    switch selectedTab {
                                    case 0:
                                        GestureRegistrationView(handLandmarkerVM: handLandmarkerVM)
                                            .padding()
                                        
                                    case 1:
                                        PredictionSheetView(
                                            gesturePredictor: gesturePredictor,
                                            handLandmarkerVM: handLandmarkerVM
                                        )
                                        .padding()
                                        
                                    case 2:
                                        HandBottomSheetView(
                                            config: handLandmarkerVM.config,
                                            handLandmarkerVM: handLandmarkerVM,
                                            switchCameraAction: { cameraManager.switchCamera() }
                                        )
                                        .padding()
                                        
                                    case 3:
                                        MetricsBottomSheetView()
                                            .padding()
                                    default:
                                        EmptyView()
                                    }
                                }
                                /*
                                .transition(
                                    .asymmetric(
                                        insertion: .opacity.combined(with: .scale(scale: 0.95)),
                                        removal: .opacity
                                    )
                                )*/
                                .id(selectedTab)
                            }
                            .padding(.bottom, 70)
                            .padding(.top, 50)
                        }
                        .scrollIndicators(.hidden)
                        .frame(maxHeight: UIScreen.main.bounds.height * 0.6)
                        .mask(
                            LinearGradient(
                                gradient: Gradient(colors: [.clear, .black, .black, .black, .black, .clear]),
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                    }
                    .background(
                        RoundedRectangle(cornerRadius: 15)
                            .fill(tabColors[selectedTab].opacity(0.3))
                            .mask(
                                LinearGradient(
                                    gradient: Gradient(colors: [.clear, .black, .black]),
                                    startPoint: .top,
                                    endPoint: .bottom
                                )
                            )
                    )
                    .offset(y: isSheetExpanded ? 0 : 300)
                    .opacity(isSheetExpanded ? 1 : 0)
                    .zIndex(0)
                }
                // Animazioni applicate al contenitore principale
                .animation(.spring(response: 0.6, dampingFraction: 0.8), value: isSheetExpanded)
                .animation(.easeInOut(duration: 0.3), value: selectedTab)
            }
        }
        .onAppear {
            cameraManager.start()
        }
        .onDisappear { cameraManager.stop() }
        .onReceive(cameraManager.$lastSampleBuffer) { sampleBuffer in
            guard let sampleBuffer = sampleBuffer else { return }
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
            
            let width = CVPixelBufferGetWidth(pixelBuffer)
            let height = CVPixelBufferGetHeight(pixelBuffer)
            handLandmarkerVM.currentImageSize = CGSize(width: width, height: height)

            handLandmarkerVM.processFrame(sampleBuffer, orientation: .up)
        }
    }
    
    private func selectTab(_ tabIndex: Int) {
        if selectedTab == tabIndex {
            // Se clicco sulla tab già selezionata, toggle l'espansione
            isSheetExpanded.toggle()
        } else {
            // Se clicco su una nuova tab, selezionala e espandi
            selectedTab = tabIndex
            isSheetExpanded = true
        }
        
        // Se la tab delle metriche è selezionata, carica le metriche
        if tabIndex == 3 && !hasLoadedInitialMetrics {
            loadMetrics()
            hasLoadedInitialMetrics = true
        }
    }
    
    // Funzione per caricare le metriche
    private func loadMetrics() {
        print("Caricamento metriche...")
        // Aggiungi qui la logica specifica per caricare le metriche
    }
}

// Componente per i pulsanti delle tab
struct TabButton: View {
    let title: String
    let icon: String
    let isSelected: Bool
    let color: Color
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack {
                Image(systemName: icon)
                    .font(.system(size: 20))
                    .foregroundColor(isSelected ? color : .primary)
                Text(title)
                    .font(.caption)
                    .foregroundColor(isSelected ? color : .primary)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
            .background(isSelected ? color.opacity(0.2) : Color.clear)
            .cornerRadius(8)
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(isSelected ? color : Color.clear, lineWidth: 1)
            )
        }
    }
}

private struct HandBBoxOverlay: View {
    @ObservedObject var vm: HandLandmarkerViewModel
    let index: Int
    let box: CGRect
    let containerSize: CGSize
    
    private var isSingle: Bool { vm.detectedHands.count == 1 }
    private var label: String {
        isSingle ? vm.gestureRecognized : (index == 0 ? vm.leftHandGesture : vm.rightHandGesture)
    }
    private var confidence: Double {
        isSingle ? vm.predictionConfidence : (index == 0 ? vm.leftHandConfidence : vm.rightHandConfidence)
    }
    private var threshold: Double {
        label == "Unknown" ? .infinity : vm.threshold(for: label)
    }
    private var isRecognized: Bool { confidence >= threshold }
    private var color: Color { isSingle ? .green : (index == 0 ? .cyan : .yellow) }
    
    var body: some View {
        ZStack(alignment: .topLeading) {
            Rectangle()
                .stroke(isRecognized ? color : Color.red, lineWidth: 3)
                .frame(width: box.width, height: box.height)
                .position(x: box.midX, y: box.midY)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(label)
                    .font(.caption)
                    .fontWeight(.bold)
                Text(String(format: "%.0f%%", confidence * 100))
                    .font(.caption2)
            }
            .foregroundColor(.black)
            .padding(.horizontal, 6)
            .padding(.vertical, 4)
            .background((isRecognized ? color : Color.red).opacity(0.9))
            .cornerRadius(6)
            .shadow(radius: 2)
            .position(x: min(max(box.minX + 60, 0), containerSize.width), y: max(box.minY - 14, 12))
        }
        .allowsHitTesting(false)
    }
}
