// GesturePredictor.swift
import Foundation
import CoreML
import QuartzCore

/// Enhanced gesture prediction using a CoreML model with dual-hand support and performance optimizations
class GesturePredictor: ObservableObject {
    private var model: GestureClassifier?
    private let thresholdManager = ConfidenceThresholdManager()
    private var metrics: MetricsStruct?
    // Thread-safety for performance metrics
    private let metricsLock = NSLock()
    // Thread-safety for left/right hand histories
    private let handHistoryLock = NSLock()
    
    // MARK: - Dual Hand Support
    enum HandType: String {
        case left = "left"
        case right = "right"
        case unknown = "unknown"
    }
    
    struct HandPredictionResult {
        let handType: HandType
        let handIndex: Int
        let label: String
        let confidence: Double
        let timestamp: Date
    }
    
    // Enhanced caching for dual hands with separate histories
    private var leftHandHistory: [(label: String, confidence: Double, timestamp: Date)] = []
    private var rightHandHistory: [(label: String, confidence: Double, timestamp: Date)] = []
    private var predictionHistory: [(label: String, confidence: Double, timestamp: Date)] = []
    private let maxHistoryCount = 30 // Optimized for dual hands
    
    // MARK: - Performance Optimizations
    private let predictionQueue = DispatchQueue(label: "com.yourapp.predictionQueue", qos: .userInitiated, attributes: .concurrent)
    private let batchProcessingQueue = DispatchQueue(label: "com.yourapp.batchProcessing", qos: .utility)
    
    // Model pooling for better performance
    private var modelPool: [GestureClassifier] = []
    private let poolSemaphore = DispatchSemaphore(value: 2) // Allow 2 concurrent predictions
    private let poolLock = NSLock()
    
    // Memory management
    private var featureCache: [String: [Double]] = [:]
    private let maxCacheSize = 20
    private let featureCacheLock = NSLock()
    
    // Configuration temporal smoothing - optimized for dual hands
    @Published var config = SmoothingConfig()

    struct SmoothingConfig {
        var timeWindow: TimeInterval = 1.0 // Reduced for faster dual-hand response
        var minConfidenceThreshold: Double = 0.4 // Lowered for dual-hand scenarios
        var minStableFrames: Int = 2
        var requiredConsensusRatio: Double = 0.4 // More lenient for dual hands
        var enableBatchProcessing: Bool = true
        var maxConcurrentPredictions: Int = 2

        static func defaultValue() -> SmoothingConfig {
            return SmoothingConfig(
                timeWindow: 1.0,
                minConfidenceThreshold: 0.4,
                minStableFrames: 2,
                requiredConsensusRatio: 0.4,
                enableBatchProcessing: true,
                maxConcurrentPredictions: 2
            )
        }
    }
    
    // Dedicated code for prediction with performance metrics
    private var lastPrediction: (label: String, confidence: Double)?
    private var performanceMetrics: [TimeInterval] = []
    
    // MARK: - Published properties for dual hands
    @Published var leftHandGesture: String = "Unknown"
    @Published var rightHandGesture: String = "Unknown"
    @Published var combinedGesture: String = "Unknown"
    @Published var isProcessingDualHands: Bool = false
    
    init() {
        loadModel()
        setupModelPool()
        setupPerformanceMonitoring()
    }
    
    // MARK: - Enhanced Model Loading with Pooling
    private func setupModelPool() {
        poolLock.lock()
        defer { poolLock.unlock() }
        
        // Create model pool for concurrent predictions
        for _ in 0..<config.maxConcurrentPredictions {
            do {
                let pooledModel = try GestureClassifier(configuration: MLModelConfiguration())
                modelPool.append(pooledModel)
            } catch {
                print("Error creating pooled model: \(error.localizedDescription)")
            }
        }
    }
    
    private func borrowModel() -> GestureClassifier? {
        poolSemaphore.wait()
        poolLock.lock()
        defer { poolLock.unlock() }
        
        if !modelPool.isEmpty {
            return modelPool.removeFirst()
        }
        return model // Fallback to main model
    }
    
    private func returnModel(_ borrowedModel: GestureClassifier) {
        poolLock.lock()
        defer {
            poolLock.unlock()
            poolSemaphore.signal()
        }
        
        if borrowedModel !== model {
            modelPool.append(borrowedModel)
        }
    }
    
    // MARK: - Dual Hand Prediction Interface
    func predictFromDualHands(
        leftHandFeatures: [Double]?,
        rightHandFeatures: [Double]?,
        completion: @escaping ([HandPredictionResult]) -> Void
    ) {
        let startTime = CACurrentMediaTime()
        let cfg = self.config // Snapshot config to avoid background thread reads
        DispatchQueue.main.async { self.isProcessingDualHands = true }
        
        var handsToProcess: [(features: [Double], handType: HandType, index: Int)] = []
        
        if let leftFeatures = leftHandFeatures {
            handsToProcess.append((leftFeatures, .left, 0))
        }
        
        if let rightFeatures = rightHandFeatures {
            handsToProcess.append((rightFeatures, .right, 1))
        }
        
        guard !handsToProcess.isEmpty else {
            DispatchQueue.main.async {
                self.isProcessingDualHands = false
                completion([])
            }
            return
        }
        
        if cfg.enableBatchProcessing && handsToProcess.count > 1 {
            processConcurrentHands(hands: handsToProcess, cfg: cfg) { [weak self] results in
                self?.finalizeDualHandPrediction(results: results, startTime: startTime, completion: completion)
            }
        } else {
            processSequentialHands(hands: handsToProcess, cfg: cfg) { [weak self] results in
                self?.finalizeDualHandPrediction(results: results, startTime: startTime, completion: completion)
            }
        }
    }
    
    // MARK: - Concurrent Processing for Performance
    private func processConcurrentHands(
        hands: [(features: [Double], handType: HandType, index: Int)],
        cfg: SmoothingConfig,
        completion: @escaping ([HandPredictionResult]) -> Void
    ) {
        let group = DispatchGroup()
        var results: [HandPredictionResult] = []
        let resultsLock = NSLock()
        
        for handData in hands {
            group.enter()
            
            predictionQueue.async {
                defer { group.leave() }
                
                if let result = self.predictSingleHandOptimized(
                    features: handData.features,
                    handType: handData.handType,
                    handIndex: handData.index,
                    cfg: cfg
                ) {
                    resultsLock.lock()
                    results.append(result)
                    resultsLock.unlock()
                }
            }
        }
        
        group.notify(queue: DispatchQueue.main) {
            completion(results)
        }
    }
    
    private func processSequentialHands(
        hands: [(features: [Double], handType: HandType, index: Int)],
        cfg: SmoothingConfig,
        completion: @escaping ([HandPredictionResult]) -> Void
    ) {
        var results: [HandPredictionResult] = []
        
        for handData in hands {
            if let result = predictSingleHandOptimized(
                features: handData.features,
                handType: handData.handType,
                handIndex: handData.index,
                cfg: cfg
            ) {
                results.append(result)
            }
        }
        
        DispatchQueue.main.async {
            completion(results)
        }
    }
    
    // MARK: - Optimized Single Hand Prediction
    private func predictSingleHandOptimized(
        features: [Double],
        handType: HandType,
        handIndex: Int,
        cfg: SmoothingConfig
    ) -> HandPredictionResult? {
        let finalFeatures: [Double]
        
        // Feature processing with caching
        let cacheKey = "\(handType.rawValue)_\(handIndex)"
        if features.count == 30 {
            finalFeatures = features
            updateFeatureCache(key: cacheKey, features: features)
        } else {
            guard let selectedIndices = LandmarkUtils.selectedFeatureIndices else {
                return nil
            }
            
            guard selectedIndices.allSatisfy({ $0 < features.count }) else {
                return nil
            }
            
            finalFeatures = selectedIndices.map { features[$0] }
            updateFeatureCache(key: cacheKey, features: finalFeatures)
        }
        
        // Get model from pool for concurrent processing
        guard let borrowedModel = borrowModel() else {
            return nil
        }
        
        defer { returnModel(borrowedModel) }
        
        do {
            let input = try createModelInput(from: finalFeatures)
            let prediction = try borrowedModel.prediction(input: input)
            
            guard let (label, confidence) = prediction.labelProbability.max(by: { $0.value < $1.value }) else {
                return nil
            }
            
            let result = HandPredictionResult(
                handType: handType,
                handIndex: handIndex,
                label: label,
                confidence: confidence,
                timestamp: Date()
            )
            
            // Apply hand-specific temporal smoothing
            return applyHandSpecificSmoothing(result: result, cfg: cfg)
            
        } catch {
            print("[DEBUGGING] Error during optimized prediction: \(error.localizedDescription)")
            return nil
        }
    }
    
    // MARK: - Hand-Specific Temporal Smoothing
    private func applyHandSpecificSmoothing(result: HandPredictionResult, cfg: SmoothingConfig) -> HandPredictionResult? {
        let now = Date()
        
        switch result.handType {
        case .left:
            handHistoryLock.lock()
            leftHandHistory.append((result.label, result.confidence, now))
            cleanupHandHistory(history: &leftHandHistory, currentTime: now, timeWindow: cfg.timeWindow)
            let snapshot = leftHandHistory
            handHistoryLock.unlock()
            return applyTemporalFilterToHand(history: snapshot, result: result, cfg: cfg)
            
        case .right:
            handHistoryLock.lock()
            rightHandHistory.append((result.label, result.confidence, now))
            cleanupHandHistory(history: &rightHandHistory, currentTime: now, timeWindow: cfg.timeWindow)
            let snapshot = rightHandHistory
            handHistoryLock.unlock()
            return applyTemporalFilterToHand(history: snapshot, result: result, cfg: cfg)
            
        case .unknown:
            return result // No smoothing for unknown hands
        }
    }
    
    private func cleanupHandHistory(
        history: inout [(label: String, confidence: Double, timestamp: Date)],
        currentTime: Date,
        timeWindow: TimeInterval
    ) {
        history = history.filter {
            currentTime.timeIntervalSince($0.timestamp) <= timeWindow
        }
        
        if history.count > maxHistoryCount {
            history.removeFirst(history.count - maxHistoryCount)
        }
    }
    
    private func applyTemporalFilterToHand(
        history: [(label: String, confidence: Double, timestamp: Date)],
        result: HandPredictionResult,
        cfg: SmoothingConfig
    ) -> HandPredictionResult? {
        guard history.count >= cfg.minStableFrames else {
            return result
        }
        
        var classConfidences: [String: (total: Double, count: Int)] = [:]
        
        for (label, confidence, _) in history {
            classConfidences[label, default: (0, 0)].total += confidence
            classConfidences[label]?.count += 1
        }
        
        guard let bestClass = classConfidences.max(by: {
            ($0.value.total / Double($0.value.count)) < ($1.value.total / Double($1.value.count))
        }) else {
            return result
        }
        
        let averageConfidence = bestClass.value.total / Double(bestClass.value.count)
        let bestClassRatio = Double(bestClass.value.count) / Double(history.count)
        
        if bestClassRatio >= cfg.requiredConsensusRatio && averageConfidence >= cfg.minConfidenceThreshold {
            return HandPredictionResult(
                handType: result.handType,
                handIndex: result.handIndex,
                label: bestClass.key,
                confidence: averageConfidence,
                timestamp: result.timestamp
            )
        }
        
        return nil
    }
    
    // MARK: - Feature Caching for Performance
    private func updateFeatureCache(key: String, features: [Double]) {
        featureCacheLock.lock()
        defer { featureCacheLock.unlock() }
        if featureCache.count >= maxCacheSize, let firstKey = featureCache.keys.first {
            featureCache.removeValue(forKey: firstKey)
        }
        featureCache[key] = features
    }
    
    // MARK: - Finalization and UI Updates
    private func finalizeDualHandPrediction(
        results: [HandPredictionResult],
        startTime: TimeInterval,
        completion: @escaping ([HandPredictionResult]) -> Void
    ) {
        let processingTime = CACurrentMediaTime() - startTime
        updatePerformanceMetrics(time: processingTime)
        
        // Update published properties and call completion on main thread
        DispatchQueue.main.async {
            self.updateUIForDualHands(results: results)
            self.isProcessingDualHands = false
            completion(results)
        }
    }
    
    private func updateUIForDualHands(results: [HandPredictionResult]) {
        // Update individual hand gestures
        if let leftResult = results.first(where: { $0.handType == .left }) {
            leftHandGesture = leftResult.label
        } else {
            leftHandGesture = "Unknown"
        }
        
        if let rightResult = results.first(where: { $0.handType == .right }) {
            rightHandGesture = rightResult.label
        } else {
            rightHandGesture = "Unknown"
        }
        
        // Generate combined gesture
        combinedGesture = generateCombinedGesture(results: results)
    }
    
    private func generateCombinedGesture(results: [HandPredictionResult]) -> String {
        let validResults = results.filter { $0.confidence >= config.minConfidenceThreshold }
        
        if validResults.isEmpty {
            return "Unknown"
        }
        
        if validResults.count == 1 {
            return validResults[0].label
        }
        
        // For dual hands, create meaningful combinations
        let sortedResults = validResults.sorted { $0.confidence > $1.confidence }
        let leftGesture = sortedResults.first { $0.handType == .left }?.label ?? "Unknown"
        let rightGesture = sortedResults.first { $0.handType == .right }?.label ?? "Unknown"
        
        if leftGesture != "Unknown" && rightGesture != "Unknown" {
            return "\(leftGesture) + \(rightGesture)"
        }
        
        return sortedResults[0].label
    }
    
    // MARK: - Performance Monitoring
    private func setupPerformanceMonitoring() {
        // Initialize performance tracking
        performanceMetrics.reserveCapacity(100)
        
        Timer.scheduledTimer(withTimeInterval: 15.0, repeats: true) { _ in
            self.cleanupPerformanceMetrics()
        }
    }
    
    private func updatePerformanceMetrics(time: TimeInterval) {
        metricsLock.lock(); defer { metricsLock.unlock() }
        performanceMetrics.append(time)
        if performanceMetrics.count > 50 {
            performanceMetrics.removeFirst(25)
        }
    }
    
    private func cleanupPerformanceMetrics() {
        metricsLock.lock(); defer { metricsLock.unlock() }
        if performanceMetrics.count > 100 {
            performanceMetrics.removeFirst(50)
        }
    }
    
    // MARK: - Performance Metrics
    func getPerformanceMetrics() -> (average: TimeInterval, count: Int) {
        metricsLock.lock()
        let metricsCopy = performanceMetrics
        metricsLock.unlock()
        guard !metricsCopy.isEmpty else {
            return (average: 0.0, count: 0)
        }
        let sum = metricsCopy.reduce(0, +)
        let average = sum / Double(metricsCopy.count)
        return (average: average, count: metricsCopy.count)
    }
    
    private func recordPerformanceMetric(_ time: TimeInterval) {
        metricsLock.lock(); defer { metricsLock.unlock() }
        performanceMetrics.append(time)
        
        // Keep only recent metrics to prevent memory growth
        if performanceMetrics.count > 100 {
            performanceMetrics.removeFirst(performanceMetrics.count - 100)
        }
    }
    
    // MARK: - Enhanced Legacy Support
    func predictFromFeatures(_ features: [Double], completion: @escaping ((label: String, confidence: Double)?) -> Void) {
        // Enhanced legacy method with performance optimizations
        predictionQueue.async {
            let finalFeatures: [Double]
            
            if features.count == 30 {
                finalFeatures = features
            } else {
                guard let selectedIndices = LandmarkUtils.selectedFeatureIndices else {
                    DispatchQueue.main.async { completion(nil) }
                    return
                }
                
                guard selectedIndices.allSatisfy({ $0 < features.count }) else {
                    DispatchQueue.main.async { completion(nil) }
                    return
                }
                
                finalFeatures = selectedIndices.map { features[$0] }
            }
            
            let result = self.predictWithFeatures(finalFeatures)
            DispatchQueue.main.async { completion(result) }
        }
    }

    /// Internal prediction method with feature validation
    private func predictWithFeatures(_ features: [Double]) -> (label: String, confidence: Double)? {
       guard let model = model, features.count == 30 else {
           return nil
       }
       
       do {
           let input = try createModelInput(from: features)
           let prediction = try model.prediction(input: input)
           
           guard let (label, confidence) = prediction.labelProbability.max(by: { $0.value < $1.value }) else {
               return nil
           }
           
           lastPrediction = (label, confidence)
           
           // Return raw prediction; apply thresholds after temporal smoothing at a higher layer
           return (label, confidence)
       } catch {
           print("[DEBUGGING] Error during prediction: \(error.localizedDescription)")
           return nil
       }
   }
    
    /// Makes a prediction with temporal smoothing
    func predictWithTemporalSmoothing(from features: [Double], completion: @escaping ((label: String, confidence: Double)?) -> Void) {
        predictionQueue.async {
            guard let currentPrediction = self.predictWithFeatures(features) else {
                DispatchQueue.main.async { completion(nil) }
                return
            }
            
            let now = Date()
            self.predictionHistory.append((currentPrediction.label, currentPrediction.confidence, now))
            self.cleanupHistory(currentTime: now)
            
            // Se la confidence è molto alta, restituisci immediatamente
            if currentPrediction.confidence > 0.85 {
                DispatchQueue.main.async { completion(currentPrediction) }
                return
            }
            
            let result = self.applyTemporalFilter()
            DispatchQueue.main.async { completion(result) }
        }
    }
    
    /// Removes old predictions from history
    private func cleanupHistory(currentTime: Date) {
        // Remove predictions older than the time window
        predictionHistory = predictionHistory.filter {
            currentTime.timeIntervalSince($0.timestamp) <= config.timeWindow
        }
        
        // Also limit the total count to prevent memory issues
        if predictionHistory.count > maxHistoryCount {
            predictionHistory.removeFirst(predictionHistory.count - maxHistoryCount)
        }
    }
    
    /// Applies temporal filtering to smooth predictions
    private func applyTemporalFilter() -> (label: String, confidence: Double)? {
       guard predictionHistory.count >= config.minStableFrames else {
           return predictionHistory.last.map { ($0.label, $0.confidence) }
       }
       
       var classConfidences: [String: (total: Double, count: Int)] = [:]
       var totalConfidence: Double = 0
       var totalCount = 0
       
       for (label, confidence, _) in predictionHistory {
           classConfidences[label, default: (0, 0)].total += confidence
           classConfidences[label]?.count += 1
           totalConfidence += confidence
           totalCount += 1
       }
       
       // Calcola la confidence media totale
       let averageTotalConfidence = totalConfidence / Double(totalCount)
       
       // Se la confidence media totale è bassa, restituisci nil
       guard averageTotalConfidence >= config.minConfidenceThreshold else {
           return nil
       }
       
       // Trova la label con la confidence media più alta
       guard let bestClass = classConfidences.max(by: {
           ($0.value.total / Double($0.value.count)) < ($1.value.total / Double($1.value.count))
       }) else {
           return nil
       }
       
       let averageConfidence = bestClass.value.total / Double(bestClass.value.count)
       let bestClassRatio = Double(bestClass.value.count) / Double(totalCount)
       
       return (bestClassRatio >= config.requiredConsensusRatio &&
               averageConfidence >= config.minConfidenceThreshold) ?
               (bestClass.key, averageConfidence) : nil
   }
    /// Alternative: simple voting mode with threshold
    private func applySimpleVoting() -> (label: String, confidence: Double)? {
        var voteCounts: [String: Int] = [:]
        
        // Count votes only for predictions above the threshold
        for (label, confidence, _) in predictionHistory where confidence >= config.minConfidenceThreshold {
            voteCounts[label, default: 0] += 1
        }
        
        guard let bestLabel = voteCounts.max(by: { $0.value < $1.value })?.key else {
            return nil
        }
        
        // Calculate average confidence for the winning label
        let confidences = predictionHistory
            .filter { $0.label == bestLabel && $0.confidence >= config.minConfidenceThreshold }
            .map { $0.confidence }
        
        guard !confidences.isEmpty else {
            return nil
        }
        
        let averageConfidence = confidences.reduce(0, +) / Double(confidences.count)
        return (bestLabel, averageConfidence)
    }
    
    /// Reset the prediction history
    func resetTemporalHistory() {
        predictionHistory.removeAll()
    }
    
    /// Gets the current history state for debugging
    func getHistoryState() -> String {
        var state = "History count: \(predictionHistory.count)\n"
        
        let grouped = Dictionary(grouping: predictionHistory, by: { $0.label })
        for (label, predictions) in grouped {
            let avgConf = predictions.map { $0.confidence }.reduce(0, +) / Double(predictions.count)
            state += "\(label): \(predictions.count) frames, avg conf: \(String(format: "%.2f", avgConf))\n"
        }
        
        return state
    }
    
    /// Loads metrics from a file and calculates optimal thresholds
    @discardableResult
    func loadMetrics(from url: URL) -> Bool {
        do {
            let data = try Data(contentsOf: url)
            let metrics = try JSONDecoder().decode(MetricsStruct.self, from: data)
            self.metrics = metrics
            thresholdManager.calculateOptimalThresholds(from: metrics)
            print("Metrics loaded successfully")
            return true
        } catch {
            print("Error loading metrics: \(error.localizedDescription)")
            return false
        }
    }
    
    /// Returns all current thresholds
    func getAllThresholds() -> [String: Double] {
        return thresholdManager.getAllThresholds()
    }
    
    /// Gets the last prediction made by the model
    func getLastPrediction() -> (label: String, confidence: Double)? {
        return lastPrediction
    }
    
    // MARK: - Model Input Creation
    private func createModelInput(from features: [Double]) throws -> GestureClassifierInput {
        // Ensure we have exactly 30 features
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
    
    private func loadModel() {
        do {
            model = try GestureClassifier(configuration: MLModelConfiguration())
            print("Gesture model loaded successfully")
        } catch {
            print("Error loading gesture model: \(error.localizedDescription)")
        }
    }
    
    // MARK: - Debug Functions
    
    /// Makes prediction with detailed debug information
    func predictWithDebug(features: [Double]) -> (label: String, confidence: Double)? {
        guard let model = model else {
            print("[DEBUGGING] Error: Model not loaded")
            return nil
        }
        
        do {
            let input = try createModelInput(from: features)
            let prediction = try model.prediction(input: input)
            
            // Print all probabilities
            print("[DEBUGGING] All class probabilities: \(prediction.labelProbability)")
            
            // Find the prediction with the highest probability
            guard let (label, confidence) = prediction.labelProbability.max(by: { $0.value < $1.value }) else {
                print("[DEBUGGING] Error: Could not get prediction probabilities")
                return nil
            }
            
            print("[DEBUGGING] Raw prediction: \(label)")
            print("[DEBUGGING] Confidence: \(confidence)")
            
            // Find the second best prediction
            let sorted = prediction.labelProbability.sorted(by: { $0.value > $1.value })
            if sorted.count >= 2 {
                print("[DEBUGGING] Top prediction: \(sorted[0].key) = \(sorted[0].value)")
                print("[DEBUGGING] Second prediction: \(sorted[1].key) = \(sorted[1].value)")
            }
            
            return (label, confidence)
        } catch {
            print("[DEBUGGING] Error during prediction: \(error.localizedDescription)")
            return nil
        }
    }
    
    /// Gets all probabilities from the model
    private func getAllProbabilities(from features: [Double]) -> [String: Double]? {
        guard let model = model else {
            print("Error: Model not loaded")
            return nil
        }
        
        do {
            let input = try createModelInput(from: features)
            let prediction = try model.prediction(input: input)
            return prediction.labelProbability
        } catch {
            print("Error getting probabilities: \(error.localizedDescription)")
            return nil
        }
    }
    
    // MARK: - Dual Hand Support Methods
    
    /// Reset histories for both hands
    func resetDualHandHistory() {
        leftHandHistory.removeAll()
        rightHandHistory.removeAll()
        print("Dual hand history cleared")
    }
    
    /// Update configuration for dual hands
    func updateDualHandConfig(_ newConfig: SmoothingConfig) {
        DispatchQueue.main.async {
            self.config = newConfig
            print("Dual hand configuration updated")
        }
    }
}

enum PredictionError: LocalizedError {
    case invalidFeatureCount(expected: Int, actual: Int)
    
    var errorDescription: String? {
        switch self {
        case .invalidFeatureCount(let expected, let actual):
            return "Expected \(expected) features, but got \(actual)"
        }
    }
}
