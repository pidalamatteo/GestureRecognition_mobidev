import SwiftUI
import AVFoundation
import MediaPipeTasksVision
import CoreML

@MainActor
class HandLandmarkerViewModel: NSObject, ObservableObject {
    
    // MARK: - Configurazioni
    @ObservedObject var config = DefaultCostants()
    
    // MARK: - Risultati
    @Published var detectedHands: [[NormalizedLandmark]] = []
    @Published var inferenceTime: Double = 0.0
    @Published var frameInterval: Double = 0.0
    @Published var currentImageSize: CGSize? = nil
    
    // MARK: - Gestione Gesture Prediction
    private var gesturePredictor = GesturePredictor()
    @Published var confidenceThresholds: [String: Double] = [:]
    @Published var gestureRecognized: String = "Unknown"
    @Published var isGestureRecognized: Bool = false
    @Published var predictionConfidence: Double = 0.0
    
    // MARK: - Dual Hand Prediction Support
    @Published var leftHandGesture: String = "Unknown"
    @Published var rightHandGesture: String = "Unknown"
    @Published var combinedGesture: String = "Unknown"
    @Published var isDualHandMode: Bool = true // Enable dual-hand mode by default
    @Published var processingDualHands: Bool = false
    
    // Performance metrics for dual hands
    @Published var leftHandConfidence: Double = 0.0
    @Published var rightHandConfidence: Double = 0.0
    @Published var dualHandProcessingTime: Double = 0.0
    
    // Latest predictions to drive overlays (moved into main class)
    private var latestHandPredictions: [GesturePredictor.HandPredictionResult] = []

    struct HandOverlay {
        let boundingBox: CGRect
        let label: String
        let confidence: Double
        let handType: GesturePredictor.HandType
    }

    @Published var handOverlays: [HandOverlay] = []

    // MARK: - Gestione Registrazione
    @Published var gestureLabelToRegister: String = ""
    @Published var isRecordingGesture = false
    @Published var totalRecordingTime: Double = 0 // ms
    @Published var avgPresence: Float = 0.0
    
    // MARK: - Gestione Thread-Safe delle samples
    private let samplesQueue = DispatchQueue(label: "com.yourapp.samplesQueue", attributes: .concurrent)
    private let saveQueue = DispatchQueue(label: "com.yourapp.saveQueue", qos: .utility)
    private var internalRecordedSamples: [LandmarkSample] = []
    private var saveTimer: Timer?
    private var isRecordingSession = false
    
    @Published var recordedSamples: [LandmarkSample] = [] {
        didSet {
            // Only save immediately if not in a recording session
            if !isRecordingSession {
                self.saveSamplesToFileAsync()
            } else {
                // During recording, batch saves to reduce I/O overhead
                self.scheduleBatchedSave()
            }
        }
    }
    
    private var startTime: CFTimeInterval = 0
    private var lastFrameTimestamp: CFTimeInterval? = nil
    private var handLandmarkerService: HandLandmarkerServiceLiveStream?
    private var recordingStartTime: CFTimeInterval?
    private var recordingFrameCounter: Int = 0
    
    private let saveFileName = "gestures.json"
    private var saveURL: URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return documents.appendingPathComponent(saveFileName)
    }
    
    // MARK: - Inizializzazione
    override init() {
        super.init()
        setupLandmarker()
        loadSamples()
        loadModelMetrics()
        LandmarkUtils.debugFileLocations()
        loadFeatureIndices()
        
        // Esegui test di consistenza dopo un breve delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            self.testPredictionConsistency()
        }
    }
    
    // MARK: - Setup Landmarker
    private func setupLandmarker() {
        guard let modelPath = config.modelPath else {
            fatalError("Modello hand_landmarker.task non trovato nel bundle")
        }
        
        handLandmarkerService = HandLandmarkerServiceLiveStream(
            modelPath: modelPath,
            numHands: config.numHands,
            minHandDetectionConfidence: config.minHandDetectionConfidence,
            minHandPresenceConfidence: config.minHandPresenceConfidence,
            minTrackingConfidence: config.minTrackingConfidence
        )
        
        handLandmarkerService?.delegate = self
    }
    
    // MARK: - Caricamento metriche del modello e feature selection
    private func loadModelMetrics() {
        if let metricsURL = Bundle.main.url(forResource: "model_metrics", withExtension: "json") {
            let success = gesturePredictor.loadMetrics(from: metricsURL)
            if success {
                confidenceThresholds = gesturePredictor.getAllThresholds()
                print("Metriche del modello caricate con successo")
            }
        }
    }
    
    private func loadFeatureIndices() {
        // Carica gli indici delle feature selezionate
        _ = LandmarkUtils.loadSelectedFeatureIndices()
        print("Indici delle feature selezionate caricati")
    }
    
    // MARK: - Aggiornamento opzioni
    func updateOptions() {
        guard let modelPath = config.modelPath else { return }
        
        handLandmarkerService = HandLandmarkerServiceLiveStream(
            modelPath: modelPath,
            numHands: config.numHands,
            minHandDetectionConfidence: config.minHandDetectionConfidence,
            minHandPresenceConfidence: config.minHandPresenceConfidence,
            minTrackingConfidence: config.minTrackingConfidence
        )
        
        handLandmarkerService?.delegate = self
        print("HandLandmarker aggiornato: numHands=\(config.numHands), minDetection=\(config.minHandDetectionConfidence)")
    }
    
    // MARK: - Processamento frame
    func processFrame(_ sampleBuffer: CMSampleBuffer, orientation: UIImage.Orientation) {
        startTime = CACurrentMediaTime()
        handLandmarkerService?.detectAsync(sampleBuffer: sampleBuffer, orientation: orientation)
    }
    
    // MARK: - Gestione Registrazione Campioni
    func startRecordingGesture(label: String) {
        gestureLabelToRegister = label
        recordingFrameCounter = 0
        isRecordingGesture = true
        isRecordingSession = true // Attiva la modalità batch per il salvataggio
        totalRecordingTime = 0
        recordingStartTime = CACurrentMediaTime()
        print("Inizio registrazione gesto [\(label)]")
        
        // Resetta la history delle prediction quando inizi a registrare
        gesturePredictor.resetTemporalHistory()
    }
    
    func stopRecordingGesture() {
        isRecordingGesture = false
        isRecordingSession = false // Disattiva la modalità batch
        
        // Forza il salvataggio immediato alla fine della sessione
        saveSamplesToFileAsync()
        
        // Cancella eventuali timer pendenti
        saveTimer?.invalidate()
        saveTimer = nil
        
        if let start = recordingStartTime {
            totalRecordingTime = (CACurrentMediaTime() - start)
        }
        printRecordedSamples()
    }
    
    func printRecordedSamples() {
        print("--- Gesti registrati ---")
        for (index, sample) in getSamples().enumerated() {
            print("Sample #\(index + 1) | Label: \(sample.label) | Landmarks: \(sample.landmarks.count)")
        }
        print("Totale campioni registrati: \(getSamples().count)")
    }
    
    // MARK: - Metodi Thread-Safe per recordedSamples
    private func addSample(_ sample: LandmarkSample) {
        samplesQueue.async(flags: .barrier) {
            self.internalRecordedSamples.append(sample)
            DispatchQueue.main.async {
                self.recordedSamples = self.internalRecordedSamples
            }
        }
    }
    
    func getSamples() -> [LandmarkSample] {
        return samplesQueue.sync {
            return self.internalRecordedSamples
        }
    }
    
    private func clearSamples() {
        samplesQueue.async(flags: .barrier) {
            self.internalRecordedSamples.removeAll()
            DispatchQueue.main.async {
                self.recordedSamples = []
            }
        }
    }
    
    private func saveSamplesToFile() {
        do {
            let data = try JSONEncoder().encode(self.recordedSamples)
            try data.write(to: saveURL)
            print("Gestures salvati in \(saveURL)")
        } catch {
            print("Errore salvataggio gestures: \(error)")
        }
    }
    
    private func saveSamplesToFileAsync() {
        // Salvataggio asincrono ottimizzato
        saveQueue.async {
            autoreleasepool {
                do {
                    let data = try JSONEncoder().encode(self.recordedSamples)
                    try data.write(to: self.saveURL)
                    print("Gestures salvati in \(self.saveURL)")
                } catch {
                    print("Errore salvataggio gestures: \(error)")
                }
            }
        }
    }
    
    private func scheduleBatchedSave() {
        // Pianifica un salvataggio batch ogni 2 secondi durante la registrazione
        if saveTimer == nil {
            saveTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: false) { [weak self] _ in
                self?.saveSamplesToFileAsync()
                self?.saveTimer = nil
            }
        }
    }
    
    private func loadSamples() {
        do {
            let data = try Data(contentsOf: saveURL)
            let samples = try JSONDecoder().decode([LandmarkSample].self, from: data)
            samplesQueue.async(flags: .barrier) {
                self.internalRecordedSamples = samples
                DispatchQueue.main.async {
                    self.recordedSamples = samples
                }
            }
            print("Gestures caricati (\(samples.count))")
        } catch {
            print("Nessun file gestures trovato o errore: \(error)")
        }
    }
    
    func clearSavedSamples() {
        do {
            try FileManager.default.removeItem(at: saveURL)
            clearSamples()
            print("File gestures eliminato")
        } catch {
            print("Errore eliminazione file: \(error)")
        }
    }
    
    func removeSamples(for label: String) {
        samplesQueue.async(flags: .barrier) {
            self.internalRecordedSamples.removeAll { $0.label == label }
            DispatchQueue.main.async {
                self.recordedSamples = self.internalRecordedSamples
            }
        }
    }
    /*
    
    // MARK: - Predizione gesto (usando feature selection)
    func predictGesture(from landmarks: [LandmarkPoint]) -> (String, Double)? {
        // Valida i landmark prima della prediction
        guard LandmarkUtils.validateLandmarks(landmarks) else {
            print("[DEBUGGING] Landmark non validi, prediction annullata")
            return nil
        }
        
        // Usa prepareForPrediction che applica il feature selection
        let features = LandmarkUtils.prepareForPrediction(from: landmarks)
        
        // DEBUG: stampa le feature estratte
        print("[DEBUGGING] Features estratte: \(features.count)")
        print("[DEBUGGING] Prime 5 feature: \(features.prefix(5))")
        
        if let result = gesturePredictor.predictWithTemporalSmoothing(from: features) {
            return (result.label, result.confidence)
        } else {
            // Se la prediction fallisce, prova a usare il metodo di debug per avere più informazioni
            if let debugResult = gesturePredictor.predictWithDebug(features: features) {
                print("[DEBUGGING] Predizione di debug: \(debugResult.label) con confidenza \(debugResult.confidence)")
            }
            return nil
        }
    }
     */
    
    // MARK: - Utility
    func boundingBox(for landmarks: [NormalizedLandmark], originalSize: CGSize, viewSize: CGSize, isBackCamera: Bool) -> CGRect? {
        guard !landmarks.isEmpty else { return nil }

        var xs = landmarks.map { CGFloat($0.y) * viewSize.width }
        let ys = landmarks.map { CGFloat($0.x) * viewSize.height }
       
        if isBackCamera {
            xs = landmarks.map { viewSize.width - (CGFloat($0.y) * viewSize.width) }
        }
        let minX = xs.min() ?? 0
        let maxX = xs.max() ?? 0
        let minY = ys.min() ?? 0
        let maxY = ys.max() ?? 0
        
        let padding: CGFloat = 20
        return CGRect(
            x: CGFloat(minX) - padding,
            y: CGFloat(minY) - padding,
            width: CGFloat(maxX - minX) + 2*padding,
            height: CGFloat(maxY - minY) + 2*padding
        )
    }
    
    func getSavedFileURL() -> URL? {
        let url = saveURL
        if FileManager.default.fileExists(atPath: url.path) {
            return url
        } else {
            return nil
        }
    }
    
    func testPredictionConsistency() {
        // Crea landmark di test
        let testLandmarks = (0..<21).map { i in
            LandmarkPoint(
                x: Float(0.1 * Double(i)),
                y: Float(0.2 * Double(i)),
                z: Float(0.3 * Double(i))
            )
        }
        
        /*
        // Testa la prediction
        if let result = predictGesture(from: testLandmarks) {
            print("[DEBUGGING] Test prediction: \(result.0) - \(result.1)")
        } else {
            print("[DEBUGGING] Test prediction fallita")
        }
        */
        // Verifica la consistenza della normalizzazione
        LandmarkUtils.verifyNormalizationConsistency()
        
        // Verifica gli indici delle feature
        LandmarkUtils.verifyFeatureIndices()
    }
    
    func testConsistencyAfterFix() {
        let testLandmarks = (0..<21).map { i in
            LandmarkPoint(x: Float(0.1 * Double(i)), y: Float(0.2 * Double(i)), z: Float(0.3 * Double(i)))
        }
        
        // Testa la normalizzazione
        LandmarkUtils.verifyNormalizationConsistency()
        
        // Testa l'intera pipeline
        let features = LandmarkUtils.prepareForPrediction(from: testLandmarks)
        print("[DEBUGGING] Features after fix: \(features.prefix(5))...")
        
        /*
        if let result = predictGesture(from: testLandmarks) {
            print("[DEBUGGING] Prediction after fix: \(result.0) - \(result.1)")
        }
         */
    }
    
    // Reset della history quando necessario
    func resetPredictionHistory() {
        gesturePredictor.resetTemporalHistory()
    }
    
    // Esegui test di debug
    func runDebugTests() {
        testPredictionConsistency()
        
        // Verifica gli indici delle feature selezionate
        if let indices = LandmarkUtils.selectedFeatureIndices {
            print("[DEBUGGING] Indici feature selezionate: \(indices)")
            print("[DEBUGGING] Numero di indici: \(indices.count)")
        } else {
            print("[DEBUGGING] Nessun indice di feature selezionate disponibile")
        }
        
        // Verifica le thresholds
        print("[DEBUGGING] Thresholds: \(confidenceThresholds)")
        
        // Stampa lo stato della history
        print("[DEBUGGING] Stato history: \(gesturePredictor.getHistoryState())")
    }
    
    func threshold(for label: String) -> Double {
        return confidenceThresholds[label] ?? gesturePredictor.config.minConfidenceThreshold
    }
}

// MARK: - Delegate
extension HandLandmarkerViewModel: HandLandmarkerServiceLiveStreamDelegate {
    func didDetectHands(_ result: HandLandmarkerResult) {
        let now = CACurrentMediaTime()
        let elapsed = (now - startTime) * 1000.0
        var delta: Double = 0
        if let last = lastFrameTimestamp {
            delta = (now - last) * 1000.0
        }
        lastFrameTimestamp = now
        
        // Check if we have multiple hands for dual-hand prediction
        if isDualHandMode && result.landmarks.count >= 2 {
            // Use dual-hand prediction system
            predictDualHandGestures(from: result.landmarks)
        } else if let firstHand = result.landmarks.first {
            // Single hand fallback
            let points = firstHand.map { LandmarkPoint(from: $0) }
            
            if self.isRecordingGesture {
                self.handleRecording(points: points)
            } else {
                self.handleSingleHandPrediction(points: points)
            }
        } else {
            // No hands detected
            gesturePredictor.resetTemporalHistory()
            
            DispatchQueue.main.async {
                self.updateUI(elapsed: elapsed, delta: delta, landmarks: result.landmarks,
                              gesture: "No hand", confidence: 0.0, isRecognized: false)
                self.resetDualHandValues()
            }
            return
        }
        
        // Update general UI metrics
        DispatchQueue.main.async {
            self.inferenceTime = elapsed
            self.frameInterval = delta
            self.detectedHands = result.landmarks
            
            if self.isRecordingGesture, let start = self.recordingStartTime {
                self.totalRecordingTime = now - start
            }
            
            // --- Bounding box + prediction overlay ---
            var overlays: [HandOverlay] = []
            let viewSize = self.currentImageSize ?? CGSize(width: 1, height: 1)
            let isBackCamera = true // Or get from config if needed
            for (i, handLandmarks) in result.landmarks.prefix(2).enumerated() {
                if let bbox = self.boundingBox(for: handLandmarks, originalSize: viewSize, viewSize: viewSize, isBackCamera: isBackCamera) {
                    let prediction = self.latestHandPredictions.first(where: { $0.handIndex == i })
                    overlays.append(HandOverlay(
                        boundingBox: bbox,
                        label: prediction?.label ?? "Unknown",
                        confidence: prediction?.confidence ?? 0.0,
                        handType: prediction?.handType ?? .unknown
                    ))
                }
            }
            self.handOverlays = overlays
        }
    }
    private func handleRecording(points: [LandmarkPoint]) {
        recordingFrameCounter += 1
        
        if recordingFrameCounter % 2 == 0,
           shouldSaveSample(newPoints: points,
                            lastSaved: getSamples().last?.landmarks,
                            config: config,
                            distanceThreshold: config.minFrameDistance,
                            minVisibleLandmarks: 5) {
            
            let sample = LandmarkSample(label: gestureLabelToRegister, landmarks: points)
            addSample(sample)
            
            DispatchQueue.main.async {
                self.isGestureRecognized = true
                self.gestureRecognized = "Saved"
                
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    self.isGestureRecognized = false
                    self.gestureRecognized = "Recording..."
                }
            }
        }
    }
    private func handleSingleHandPrediction(points: [LandmarkPoint]) {
        let features = LandmarkUtils.prepareForPrediction(from: points)
        
        gesturePredictor.predictWithTemporalSmoothing(from: features) { result in
            DispatchQueue.main.async {
                if let result = result {
                    self.gestureRecognized = result.label
                    self.predictionConfidence = result.confidence
                    let thr = result.label == "Unknown" ? Double.infinity : self.threshold(for: result.label)
                    self.isGestureRecognized = result.confidence >= thr
                    
                    // Reset dual-hand specific values
                    self.leftHandGesture = "Unknown"
                    self.rightHandGesture = "Unknown"
                    self.combinedGesture = result.label
                } else {
                    self.gestureRecognized = "Unknown"
                    self.predictionConfidence = 0.0
                    self.isGestureRecognized = false
                    self.resetDualHandValues()
                }
            }
        }
    }
    
    private func updateUI(elapsed: Double, delta: Double, landmarks: [[NormalizedLandmark]],
                          gesture: String, confidence: Double, isRecognized: Bool) {
        inferenceTime = elapsed
        frameInterval = delta
        detectedHands = landmarks
        gestureRecognized = gesture
        predictionConfidence = confidence
        isGestureRecognized = isRecognized
    }
    
    func shouldSaveSample(
        newPoints: [LandmarkPoint],
        lastSaved: [LandmarkPoint]?,
        config: DefaultCostants,
        distanceThreshold: Double = 0.02,
        minVisibleLandmarks: Int = 5
    ) -> Bool {
        
        func handPresenceProxy(from landmarks: [LandmarkPoint]) -> Float {
            guard !landmarks.isEmpty else { return 0.0 }
            
            let inside = landmarks.map { (0.0...1.0).contains($0.x) && (0.0...1.0).contains($0.y) ? Float(1.0) : Float(0.0) }
            let avgInside = inside.reduce(Float(0.0), +) / Float(landmarks.count)
            
            return min(avgInside, 1.0)
        }
        
        let presence = handPresenceProxy(from: newPoints)
        
        DispatchQueue.main.async {
            self.avgPresence = presence
        }
        
        let passedPresence = presence >= config.minHandPresenceConfidence
        
        // Riduci il logging di debug durante la registrazione per migliorare le performance
        #if DEBUG
        if recordingFrameCounter % 10 == 0 { // Log ogni 10 frame invece di ogni frame
            print("""
            [DEBUG] Presence check (frame \(recordingFrameCounter)):
            - Landmarks: \(newPoints.count), Proxy: \(String(format: "%.3f", presence))
            - Threshold: \(String(format: "%.2f", config.minHandPresenceConfidence))
            - Result: \(passedPresence ? "✓" : "✗")
            """)
        }
        #endif
        
        if !passedPresence { return false }
        
        let validCount = newPoints.filter { (0.0...1.0).contains($0.x) && (0.0...1.0).contains($0.y) }.count
        if validCount < minVisibleLandmarks {
            #if DEBUG
            if recordingFrameCounter % 10 == 0 {
                print("[DEBUG] Scartato: solo \(validCount) landmarks visibili (< \(minVisibleLandmarks))")
            }
            #endif
            return false
        }
        
        guard let last = lastSaved else {
            #if DEBUG
            print("[DEBUG] Primo sample → salvo")
            #endif
            return true
        }
        
        let dist = meanEuclideanDistance(newPoints, last)
        let result = dist >= distanceThreshold
        
        #if DEBUG
        if recordingFrameCounter % 10 == 0 {
            print("[DEBUG] Distanza media: \(String(format: "%.4f", dist)) → \(result ? "Accettato" : "Troppo simile")")
        }
        #endif
        
        return result
    }
    
    func meanEuclideanDistance(_ a: [LandmarkPoint], _ b: [LandmarkPoint]) -> Double {
        guard a.count == b.count, a.count > 0 else { return Double.infinity }
        var sum: Double = 0
        for i in 0..<a.count {
            let dx = Double(a[i].x - b[i].x)
            let dy = Double(a[i].y - b[i].y)
            let dz = Double(a[i].z - b[i].z)
            sum += dx*dx + dy*dy + dz*dz
        }
        return sqrt(sum) / Double(a.count)
    }
    
    // MARK: - Dual Hand Prediction Methods
    func predictDualHandGestures(from landmarks: [[NormalizedLandmark]]) {
        guard isDualHandMode && landmarks.count >= 2 else {
            // Fallback to single hand prediction
            if let firstHand = landmarks.first {
                let points = firstHand.map { LandmarkPoint(from: $0) }
                handleSingleHandPrediction(points: points)
            }
            return
        }
        
        let startTime = CACurrentMediaTime()
        DispatchQueue.main.async { self.processingDualHands = true }
        
        // Convert landmarks to features for both hands
        var handFeatures: [[Double]] = []
        for handLandmarks in landmarks.prefix(2) {
            let points = handLandmarks.map { LandmarkPoint(from: $0) }
            let features = LandmarkUtils.prepareForPrediction(from: points)
            handFeatures.append(features)
        }
        
        // Use the enhanced dual-hand prediction
        let leftFeatures = handFeatures.count > 0 ? handFeatures[0] : nil
        let rightFeatures = handFeatures.count > 1 ? handFeatures[1] : nil
        
        gesturePredictor.predictFromDualHands(
            leftHandFeatures: leftFeatures,
            rightHandFeatures: rightFeatures
        ) { [weak self] results in
            let processingTime = CACurrentMediaTime() - startTime
            DispatchQueue.main.async {
                self?.latestHandPredictions = results
                self?.updateDualHandUI(results: results, processingTime: processingTime)
                self?.processingDualHands = false
            }
        }
    }
    
    private func updateDualHandUI(results: [GesturePredictor.HandPredictionResult], processingTime: TimeInterval) {
        // Update individual hand predictions
        if let leftResult = results.first(where: { $0.handType == .left }) {
            leftHandGesture = leftResult.label
            leftHandConfidence = leftResult.confidence
        } else {
            leftHandGesture = "Unknown"
            leftHandConfidence = 0.0
        }
        
        if let rightResult = results.first(where: { $0.handType == .right }) {
            rightHandGesture = rightResult.label
            rightHandConfidence = rightResult.confidence
        } else {
            rightHandGesture = "Unknown"
            rightHandConfidence = 0.0
        }
        
        // Update combined gesture
        combinedGesture = gesturePredictor.combinedGesture
        
        // Update legacy properties for compatibility
        if !results.isEmpty {
            let bestResult = results.max(by: { $0.confidence < $1.confidence })!
            gestureRecognized = bestResult.label
            predictionConfidence = bestResult.confidence
            let thr = bestResult.label == "Unknown" ? Double.infinity : threshold(for: bestResult.label)
            isGestureRecognized = bestResult.confidence >= thr
        } else {
            gestureRecognized = "Unknown"
            predictionConfidence = 0.0
            isGestureRecognized = false
        }
        
        // Update performance metrics
        dualHandProcessingTime = processingTime * 1000 // Convert to ms
    }
    
    private func resetDualHandValues() {
        leftHandGesture = "Unknown"
        rightHandGesture = "Unknown"
        combinedGesture = "Unknown"
        leftHandConfidence = 0.0
        rightHandConfidence = 0.0
        dualHandProcessingTime = 0.0
    }
    
    // MARK: - Configuration Methods for Dual Hand
    func enableDualHandMode(_ enabled: Bool) {
        isDualHandMode = enabled
        if enabled {
            // Optimize for dual hands
            config.numHands = 2
            let optimizedConfig = GesturePredictor.SmoothingConfig(
                timeWindow: 0.8,
                minConfidenceThreshold: 0.35,
                minStableFrames: 2,
                requiredConsensusRatio: 0.3,
                enableBatchProcessing: true,
                maxConcurrentPredictions: 2
            )
            gesturePredictor.updateDualHandConfig(optimizedConfig)
            updateOptions()
        } else {
            config.numHands = 1
            updateOptions()
        }
        print("Dual hand mode: \(enabled ? "Enabled" : "Disabled")")
    }
    
    // MARK: - Performance Optimization Methods
    func optimizeForDualHands() {
        // Set optimal configuration for dual-hand performance
        config.numHands = 2
        config.minHandDetectionConfidence = 0.3  // Lower for better dual detection
        config.minHandPresenceConfidence = 0.3   // Lower for better dual detection
        config.minTrackingConfidence = 0.3       // Lower for better tracking
        
        // Update prediction configuration
        let dualHandConfig = GesturePredictor.SmoothingConfig(
            timeWindow: 0.8,                    // Faster response
            minConfidenceThreshold: 0.35,       // More sensitive
            minStableFrames: 2,                 // Less stable frames needed
            requiredConsensusRatio: 0.3,        // More lenient consensus
            enableBatchProcessing: true,         // Enable concurrent processing
            maxConcurrentPredictions: 2          // Support 2 concurrent predictions
        )
        
        gesturePredictor.updateDualHandConfig(dualHandConfig)
        updateOptions()
        
        print("Optimized for dual-hand prediction")
    }
    
    // MARK: - Dual Hand Performance Methods
    func getDualHandPerformanceMetrics() -> (averageTime: TimeInterval, count: Int) {
        let metrics = gesturePredictor.getPerformanceMetrics()
        return (averageTime: metrics.average, count: metrics.count)
    }
    
    func resetDualHandHistory() {
        gesturePredictor.resetDualHandHistory()
        leftHandGesture = "Unknown"
        rightHandGesture = "Unknown"
        combinedGesture = "Unknown"
        leftHandConfidence = 0.0
        rightHandConfidence = 0.0
    }
    
    func updateDualHandConfiguration(
        timeWindow: TimeInterval = 1.0,
        minConfidenceThreshold: Double = 0.4,
        enableBatchProcessing: Bool = true
    ) {
        var newConfig = gesturePredictor.config
        newConfig.timeWindow = timeWindow
        newConfig.minConfidenceThreshold = minConfidenceThreshold
        newConfig.enableBatchProcessing = enableBatchProcessing
        gesturePredictor.updateDualHandConfig(newConfig)
    }
}
