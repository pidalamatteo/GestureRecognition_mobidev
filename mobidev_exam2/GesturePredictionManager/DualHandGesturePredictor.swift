// DualHandGesturePredictor.swift
import Foundation
import CoreML
import Combine
import QuartzCore

/// Enhanced gesture prediction system optimized for dual-hand performance
class DualHandGesturePredictor: ObservableObject {
    
    // MARK: - Hand Identification
    enum HandType: String, CaseIterable {
        case left = "left"
        case right = "right"
        case unknown = "unknown"
    }
    
    struct HandPrediction {
        let handType: HandType
        let label: String
        let confidence: Double
        let timestamp: Date
        let handIndex: Int
    }
    
    // MARK: - Models and Configuration
    private var leftHandModel: GestureClassifier?
    private var rightHandModel: GestureClassifier?
    private var universalModel: GestureClassifier?
    
    // MARK: - Performance Optimization
    // ...removed unused queues/pool to simplify and avoid unused warnings...
    
    // MARK: - Dual Hand History Management
    private var leftHandHistory: [(label: String, confidence: Double, timestamp: Date)] = []
    private var rightHandHistory: [(label: String, confidence: Double, timestamp: Date)] = []
    private let maxHistoryCount = 30 // Reduced for dual hands
    private let historyLock = NSLock()
    
    // MARK: - Published Properties
    @Published var leftHandPrediction: HandPrediction?
    @Published var rightHandPrediction: HandPrediction?
    @Published var combinedGesture: String = "Unknown"
    @Published var isProcessing: Bool = false
    @Published var config = SmoothingConfig()
    
    struct SmoothingConfig {
        var timeWindow: TimeInterval = 1.0 // Reduced for faster response
        var minConfidenceThreshold: Double = 0.4 // Lowered for dual-hand scenarios
        var minStableFrames: Int = 2
        var requiredConsensusRatio: Double = 0.4 // More lenient for dual hands
        var enableBatchProcessing: Bool = true
        // ...removed unused batchSize to avoid confusion...
        var maxConcurrentPredictions: Int = 4
    }
    
    // MARK: - Performance Metrics
    private var processingTimes: [TimeInterval] = []
    private var lastProcessingTime: TimeInterval = 0
    private let metricsLock = NSLock()
    
    init() {
        loadModels()
        setupPerformanceMonitoring()
    }
    
    // MARK: - Model Loading with Optimization
    private func loadModels() {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = .cpuAndGPU // Use both CPU and GPU
        
        do {
            // Load universal model for general predictions
            universalModel = try GestureClassifier(configuration: modelConfig)
            print("Universal gesture model loaded successfully")
            
            // Optionally load specialized models if available
            leftHandModel = universalModel
            rightHandModel = universalModel
            
        } catch {
            print("Error loading gesture models: \(error.localizedDescription)")
        }
    }
    
    // MARK: - Main Prediction Interface for Dual Hands
    func predictFromDualHands(
        hands: [[Double]], // Array of hand feature arrays
        completion: @escaping ([HandPrediction]) -> Void
    ) {
        guard !hands.isEmpty else {
            completion([])
            return
        }
        
        // Update isProcessing on main thread
        DispatchQueue.main.async {
            self.isProcessing = true
        }
        let startTime = CACurrentMediaTime()
        
        if config.enableBatchProcessing && hands.count > 1 {
            batchProcessDualHands(hands: hands) { [weak self] predictions in
                self?.finalizePrediction(predictions: predictions, startTime: startTime, completion: completion)
            }
        } else {
            sequentialProcessDualHands(hands: hands) { [weak self] predictions in
                self?.finalizePrediction(predictions: predictions, startTime: startTime, completion: completion)
            }
        }
    }
    
    // MARK: - Batch Processing for Better Performance
    private func batchProcessDualHands(
        hands: [[Double]],
        completion: @escaping ([HandPrediction]) -> Void
    ) {
        let group = DispatchGroup()
        let concurrentQueue = DispatchQueue(label: "com.yourapp.concurrentPredictions", qos: .userInitiated, attributes: .concurrent)
        var predictions: [HandPrediction] = []
        let predictionsLock = NSLock()
        
        // Limit concurrent predictions for memory management
        let semaphore = DispatchSemaphore(value: config.maxConcurrentPredictions)
        
        for (index, handFeatures) in hands.enumerated() {
            group.enter()
            
            concurrentQueue.async {
                semaphore.wait()
                defer {
                    semaphore.signal()
                    group.leave()
                }
                
                if let prediction = self.predictSingleHand(features: handFeatures, handIndex: index) {
                    predictionsLock.lock()
                    predictions.append(prediction)
                    predictionsLock.unlock()
                }
            }
        }
        
        group.notify(queue: DispatchQueue.main) {
            completion(predictions)
        }
    }
    
    // MARK: - Sequential Processing (fallback)
    private func sequentialProcessDualHands(
        hands: [[Double]],
        completion: @escaping ([HandPrediction]) -> Void
    ) {
        var predictions: [HandPrediction] = []
        
        for (index, handFeatures) in hands.enumerated() {
            if let prediction = predictSingleHand(features: handFeatures, handIndex: index) {
                predictions.append(prediction)
            }
        }
        
        completion(predictions)
    }
    
    // MARK: - Single Hand Prediction with Hand Type Detection
    private func predictSingleHand(features: [Double], handIndex: Int) -> HandPrediction? {
        guard features.count == 30 else { return nil }
        
        // Determine hand type based on features or hand index
        let handType = determineHandType(features: features, handIndex: handIndex)
        
        // Select appropriate model
        let model = selectModelForHand(handType: handType)
        
        guard let selectedModel = model else { return nil }
        
        do {
            let input = try createModelInput(from: features)
            let prediction = try selectedModel.prediction(input: input)
            
            guard let (label, confidence) = prediction.labelProbability.max(by: { $0.value < $1.value }) else {
                return nil
            }
            
            let handPrediction = HandPrediction(
                handType: handType,
                label: label,
                confidence: confidence,
                timestamp: Date(),
                handIndex: handIndex
            )
            
            // Apply temporal smoothing (thread-safe)
            return applyTemporalSmoothing(prediction: handPrediction)
            
        } catch {
            print("Error during single hand prediction: \(error.localizedDescription)")
            return nil
        }
    }
    
    // MARK: - Hand Type Determination
    private func determineHandType(features: [Double], handIndex: Int) -> HandType {
        // Use hand position or other features to determine hand type
        // For now, use index-based assignment (can be improved with ML)
        if handIndex == 0 {
            return .left
        } else if handIndex == 1 {
            return .right
        } else {
            return .unknown
        }
    }
    
    // MARK: - Model Selection
    private func selectModelForHand(handType: HandType) -> GestureClassifier? {
        switch handType {
        case .left:
            return leftHandModel ?? universalModel
        case .right:
            return rightHandModel ?? universalModel
        case .unknown:
            return universalModel
        }
    }
    
    // MARK: - Temporal Smoothing for Dual Hands
    private func applyTemporalSmoothing(prediction: HandPrediction) -> HandPrediction? {
        let now = Date()
        var snapshot: [(label: String, confidence: Double, timestamp: Date)] = []
        
        // Thread-safe history update and snapshot
        historyLock.lock()
        switch prediction.handType {
        case .left:
            leftHandHistory.append((prediction.label, prediction.confidence, now))
            cleanupHistory(history: &leftHandHistory, currentTime: now)
            snapshot = leftHandHistory
        case .right:
            rightHandHistory.append((prediction.label, prediction.confidence, now))
            cleanupHistory(history: &rightHandHistory, currentTime: now)
            snapshot = rightHandHistory
        case .unknown:
            historyLock.unlock()
            return prediction // No smoothing for unknown hands
        }
        historyLock.unlock()
        
        // Apply filter on snapshot outside the lock
        return applyTemporalFilter(history: snapshot, prediction: prediction)
    }
    
    // MARK: - History Management
    private func cleanupHistory(history: inout [(label: String, confidence: Double, timestamp: Date)], currentTime: Date) {
        history = history.filter {
            currentTime.timeIntervalSince($0.timestamp) <= config.timeWindow
        }
        
        if history.count > maxHistoryCount {
            history.removeFirst(history.count - maxHistoryCount)
        }
    }
    
    // MARK: - Temporal Filter
    private func applyTemporalFilter(
        history: [(label: String, confidence: Double, timestamp: Date)],
        prediction: HandPrediction
    ) -> HandPrediction? {
        guard history.count >= config.minStableFrames else {
            return prediction
        }
        
        var classConfidences: [String: (total: Double, count: Int)] = [:]
        
        for (label, confidence, _) in history {
            classConfidences[label, default: (0, 0)].total += confidence
            classConfidences[label]?.count += 1
        }
        
        guard let bestClass = classConfidences.max(by: {
            ($0.value.total / Double($0.value.count)) < ($1.value.total / Double($1.value.count))
        }) else {
            return prediction
        }
        
        let averageConfidence = bestClass.value.total / Double(bestClass.value.count)
        let bestClassRatio = Double(bestClass.value.count) / Double(history.count)
        
        if bestClassRatio >= config.requiredConsensusRatio && averageConfidence >= config.minConfidenceThreshold {
            return HandPrediction(
                handType: prediction.handType,
                label: bestClass.key,
                confidence: averageConfidence,
                timestamp: prediction.timestamp,
                handIndex: prediction.handIndex
            )
        }
        
        return nil
    }
    
    // MARK: - Finalization and Performance Tracking
    private func finalizePrediction(
        predictions: [HandPrediction],
        startTime: TimeInterval,
        completion: @escaping ([HandPrediction]) -> Void
    ) {
        let endTime = CACurrentMediaTime()
        let processing = endTime - startTime
        lastProcessingTime = processing
        updatePerformanceMetrics(processingTime: processing)
        // Update published properties
        updatePublishedPredictions(predictions: predictions)
        DispatchQueue.main.async {
            self.isProcessing = false
            completion(predictions)
        }
    }
    
    // MARK: - Published Properties Update
    private func updatePublishedPredictions(predictions: [HandPrediction]) {
        DispatchQueue.main.async {
            // Update individual hand predictions
            self.leftHandPrediction = predictions.first { $0.handType == .left }
            self.rightHandPrediction = predictions.first { $0.handType == .right }
            
            // Generate combined gesture
            self.combinedGesture = self.generateCombinedGesture(predictions: predictions)
        }
    }
    
    // MARK: - Combined Gesture Logic
    private func generateCombinedGesture(predictions: [HandPrediction]) -> String {
        let validPredictions = predictions.filter { $0.confidence >= config.minConfidenceThreshold }
        
        if validPredictions.isEmpty {
            return "Unknown"
        }
        
        if validPredictions.count == 1 {
            return validPredictions[0].label
        }
        
        // For dual hands, create combined gestures
        let sortedPredictions = validPredictions.sorted { $0.confidence > $1.confidence }
        if sortedPredictions.count >= 2 {
            return "\(sortedPredictions[0].label) + \(sortedPredictions[1].label)"
        }
        
        return sortedPredictions[0].label
    }
    
    // MARK: - Performance Monitoring
    private func setupPerformanceMonitoring() {
        // Reset performance metrics periodically
        Timer.scheduledTimer(withTimeInterval: 10.0, repeats: true) { [weak self] _ in
            self?.resetPerformanceMetrics()
        }
    }
    
    private func updatePerformanceMetrics(processingTime: TimeInterval) {
        metricsLock.lock()
        processingTimes.append(processingTime)
        if processingTimes.count > 100 {
            processingTimes.removeFirst(50) // Keep last 50 measurements
        }
        metricsLock.unlock()
    }
    
    private func resetPerformanceMetrics() {
        metricsLock.lock()
        processingTimes.removeAll()
        metricsLock.unlock()
    }
    
    // MARK: - Utility Methods
    private func createModelInput(from features: [Double]) throws -> GestureClassifierInput {
        guard features.count == 30 else {
            throw PredictionError.invalidFeatureCount(expected: 30, actual: features.count)
        }
        
        return GestureClassifierInput(
            feature_0: features[0], feature_1: features[1], feature_2: features[2],
            feature_3: features[3], feature_4: features[4], feature_5: features[5],
            feature_6: features[6], feature_7: features[7], feature_8: features[8],
            feature_9: features[9], feature_10: features[10], feature_11: features[11],
            feature_12: features[12], feature_13: features[13], feature_14: features[14],
            feature_15: features[15], feature_16: features[16], feature_17: features[17],
            feature_18: features[18], feature_19: features[19], feature_20: features[20],
            feature_21: features[21], feature_22: features[22], feature_23: features[23],
            feature_24: features[24], feature_25: features[25], feature_26: features[26],
            feature_27: features[27], feature_28: features[28], feature_29: features[29]
        )
    }
    
    // MARK: - Public Interface Methods
    func resetHistory() {
        historyLock.lock()
        leftHandHistory.removeAll()
        rightHandHistory.removeAll()
        historyLock.unlock()
    }
    
    func getPerformanceMetrics() -> (averageTime: TimeInterval, lastTime: TimeInterval) {
        metricsLock.lock()
        let average = processingTimes.isEmpty ? 0 : processingTimes.reduce(0, +) / Double(processingTimes.count)
        let last = lastProcessingTime
        metricsLock.unlock()
        return (average, last)
    }
    
    func updateConfiguration(_ newConfig: SmoothingConfig) {
        // Publish config updates on main to avoid background-thread publish warnings
        DispatchQueue.main.async {
            self.config = newConfig
        }
    }
}
