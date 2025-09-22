import CreateML
import CoreML
import Foundation
import TabularData

// ==========================
// MARK: - Modello dei Dati
// ==========================
struct PreprocessedSample: Codable {
    let features: [Double]
    let label: String
    
    var trimmedLabel: String {
        return label.trimmingCharacters(in: .whitespaces)
    }
}

// Small tuning config to bound work and memory
struct TuningConfig {
    // Grid search knobs
    static let maxDepths: [Int] = [6, 8, 10, 12]
    static let maxIterations: [Int] = [50, 100, 150]
    static let gridTimeBudgetSeconds: TimeInterval = 15 * 60 // 15 minutes cap
    static let gridNoImproveEarlyStop: Int = 6 // break after N combos without improvement
    
    // I/O parallelism (bounded)
    static var ioParallelism: Int {
        let cpu = ProcessInfo.processInfo.activeProcessorCount
        return max(2, min(8, cpu * 2))
    }
    static let ioBatchSize: Int = 64
}

// ==========================
// MARK: - Label Utilities
// ==========================
@inline(__always)
func extractLabels(from data: DataFrame, labelColumn: String = "label") -> [String] {
    var labels = [String]()
    labels.reserveCapacity(data.rows.count)
    for i in 0..<data.rows.count {
        guard let v = data[labelColumn][i] as? String else {
            fatalError("Label at row \(i) is not String or is nil")
        }
        labels.append(v.trimmingCharacters(in: .whitespaces))
    }
    return labels
}

// Ensure splits share the same label set
func assertSameLabelSet(train: DataFrame, val: DataFrame, test: DataFrame) {
    let tr = Set(extractLabels(from: train))
    let va = Set(extractLabels(from: val))
    let te = Set(extractLabels(from: test))
    guard tr == va, tr == te else {
        fatalError("Inconsistent label sets across splits.\nTrain: \(tr.sorted())\nVal: \(va.sorted())\nTest: \(te.sorted())")
    }
}

// ==========================
// MARK: - Metrics Calculation (Balanced Acc = Macro Recall)
// ==========================
struct MetricsResult {
    let accuracy: Double
    let balancedAccuracy: Double
    let confusionMatrix: [String: [String: Int]]
    let classMetrics: [String: (precision: Double, recall: Double, f1: Double)]
}

func calculateMetrics(groundTruth: [String], predictions: [String]) -> MetricsResult {
    guard groundTruth.count == predictions.count else {
        fatalError("Ground truth and predictions arrays must have the same length")
    }
    
    let uniqueClasses = Array(Set(groundTruth + predictions)).sorted()
    var confusionMatrix: [String: [String: Int]] = [:]
    var classMetrics: [String: (precision: Double, recall: Double, f1: Double)] = [:]
    
    // Initialize confusion matrix
    for actualClass in uniqueClasses {
        confusionMatrix[actualClass] = [:]
        for predictedClass in uniqueClasses {
            confusionMatrix[actualClass]![predictedClass] = 0
        }
    }
    
    // Fill confusion matrix
    for i in 0..<groundTruth.count {
        let actual = groundTruth[i]
        let predicted = predictions[i]
        confusionMatrix[actual]![predicted]! += 1
    }
    
    var correctPredictions = 0
    var recalls: [Double] = []
    
    for className in uniqueClasses {
        let tp = confusionMatrix[className]![className]!
        let fp = uniqueClasses.map { confusionMatrix[$0]![className]! }.reduce(0, +) - tp
        let fn = uniqueClasses.map { confusionMatrix[className]![$0]! }.reduce(0, +) - tp
        
        let precision = tp + fp > 0 ? Double(tp) / Double(tp + fp) : 0.0
        let recall = tp + fn > 0 ? Double(tp) / Double(tp + fn) : 0.0
        let f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0.0
        
        classMetrics[className] = (precision: precision, recall: recall, f1: f1)
        recalls.append(recall)
        correctPredictions += tp
    }
    
    let accuracy = Double(correctPredictions) / Double(groundTruth.count)
    let balancedAccuracy = recalls.isEmpty ? 0.0 : (recalls.reduce(0, +) / Double(recalls.count))
    
    return MetricsResult(
        accuracy: accuracy,
        balancedAccuracy: balancedAccuracy,
        confusionMatrix: confusionMatrix,
        classMetrics: classMetrics
    )
}

@inline(__always)
func macroF1(_ m: MetricsResult) -> Double {
    let f1s = m.classMetrics.values.map { $0.f1 }
    guard !f1s.isEmpty else { return 0.0 }
    return f1s.reduce(0, +) / Double(f1s.count)
}

// ==========================
// MARK: - DataFrame Utilities
// ==========================
func createDataFrameSubset(from originalData: DataFrame, indices: [Int]) -> DataFrame {
    var dataFrame = DataFrame()
    
    for column in originalData.columns {
        let columnName = column.name
        if columnName == "label" {
            var stringValues: [String] = []
            stringValues.reserveCapacity(indices.count)
            for index in indices {
                if let value = originalData[columnName][index] as? String {
                    stringValues.append(value)
                }
            }
            dataFrame.append(column: Column(name: columnName, contents: stringValues))
        } else {
            var doubleValues: [Double] = []
            doubleValues.reserveCapacity(indices.count)
            for index in indices {
                if let value = originalData[columnName][index] as? Double {
                    doubleValues.append(value)
                }
            }
            dataFrame.append(column: Column(name: columnName, contents: doubleValues))
        }
    }
    
    return dataFrame
}

// ==========================
// MARK: - Stratified K-Fold Utilities
// ==========================
private func stratifiedFoldIndices(data: DataFrame, k: Int) -> [[Int]] {
    precondition(k > 1, "k must be > 1")
    let labels = extractLabels(from: data)
    var indicesByClass: [String: [Int]] = [:]
    for (idx, lbl) in labels.enumerated() {
        indicesByClass[lbl, default: []].append(idx)
    }
    for (cls, idxs) in indicesByClass {
        indicesByClass[cls] = idxs.shuffled()
    }
    var folds = Array(repeating: [Int](), count: k)
    for (_, idxs) in indicesByClass {
        let count = idxs.count
        let base = count / k
        let remainder = count % k
        var start = 0
        for fold in 0..<k {
            let extra = fold < remainder ? 1 : 0
            let end = start + base + extra
            if start < end { folds[fold].append(contentsOf: idxs[start..<end]) }
            start = end
        }
    }
    return folds
}

// ==========================
// MARK: - Imbalance Handling (Oversampling)
// ==========================
func oversampleTrainingData(_ data: DataFrame, labelColumn: String = "label", maxUpsampleMultiplier: Int = 3) -> DataFrame {
    let labels = extractLabels(from: data, labelColumn: labelColumn)
    var indicesByClass: [String: [Int]] = [:]
    for (i, l) in labels.enumerated() { indicesByClass[l, default: []].append(i) }
    guard !indicesByClass.isEmpty else { return data }
    let maxCount = indicesByClass.values.map { $0.count }.max() ?? 0
    if maxCount == 0 { return data }
    
    var newIndices: [Int] = []
    newIndices.reserveCapacity(indicesByClass.values.reduce(0) { $0 + min(maxCount, $1.count * maxUpsampleMultiplier) })
    for (_, idxs) in indicesByClass {
        let count = idxs.count
        if count == 0 { continue }
        let target = min(maxCount, count * maxUpsampleMultiplier)
        // Repeat full sets
        let fullRepeats = target / count
        for _ in 0..<fullRepeats { newIndices.append(contentsOf: idxs) }
        // Add remainder with sampling without replacement
        let remainder = target - fullRepeats * count
        if remainder > 0 {
            newIndices.append(contentsOf: Array(idxs.shuffled().prefix(remainder)))
        }
    }
    return createDataFrameSubset(from: data, indices: newIndices.shuffled())
}

// ==========================
// MARK: - K-Fold Cross Validation
// ==========================
struct CrossValidationResult {
    let accuracy: Double
    let balancedAccuracy: Double
    let parameters: MLBoostedTreeClassifier.ModelParameters
    let confusionMatrix: [String: [String: Int]]
    let classMetrics: [String: (precision: Double, recall: Double, f1: Double)]
}

func performKFoldCrossValidation(
    data: DataFrame,
    k: Int = 5,
    parameters: MLBoostedTreeClassifier.ModelParameters
) throws -> CrossValidationResult {
    let totalRows = data.rows.count
    precondition(totalRows >= k, "Not enough rows for k-fold")
    var accuracies: [Double] = []
    var balancedAccuracies: [Double] = []
    var lastConfusion: [String: [String: Int]] = [:]
    var lastClassMetrics: [String: (precision: Double, recall: Double, f1: Double)] = [:]
    
    // Use stratified folds to preserve class distribution in each fold
    let folds = stratifiedFoldIndices(data: data, k: k)
    
    for i in 0..<k {
        var loopError: Error?
        autoreleasepool {
            do {
                print("\n=== Fold \(i+1)/\(k) ===")
                
                let validationIndices = folds[i]
                let validationSet = Set(validationIndices)
                let trainingIndices = Array(0..<totalRows).filter { !validationSet.contains($0) }
                
                // Create validation and training DataFrames
                let validationFold = createDataFrameSubset(from: data, indices: validationIndices)
                var trainingFold = createDataFrameSubset(from: data, indices: trainingIndices)
                
                // Oversample only the training fold (smaller multiplier to limit RAM)
                trainingFold = oversampleTrainingData(trainingFold, maxUpsampleMultiplier: 2)
                
                print("Training samples: \(trainingFold.rows.count), Validation samples: \(validationFold.rows.count)")
                
                // Train the model
                let classifier = try MLBoostedTreeClassifier(
                    trainingData: trainingFold,
                    targetColumn: "label",
                    parameters: parameters
                )
                
                // Evaluate the model (use clean labels)
                let predictions = try classifier.predictions(from: validationFold)
                let truthsArray = extractLabels(from: validationFold)
                var predsArray: [String] = []
                predsArray.reserveCapacity(truthsArray.count)
                for case let s as String in predictions { predsArray.append(s) }
                
                let metrics = calculateMetrics(groundTruth: truthsArray, predictions: predsArray)
                
                accuracies.append(metrics.accuracy)
                balancedAccuracies.append(metrics.balancedAccuracy)
                lastConfusion = metrics.confusionMatrix
                lastClassMetrics = metrics.classMetrics
                
                print("📊 Fold \(i+1) - Accuracy: \(String(format: "%.4f", metrics.accuracy))")
                print("📊 Fold \(i+1) - Balanced Accuracy: \(String(format: "%.4f", metrics.balancedAccuracy))")
            } catch {
                loopError = error
            }
        }
        if let e = loopError { throw e }
    }
    
    let avgAccuracy = accuracies.reduce(0, +) / Double(k)
    let avgBalancedAccuracy = balancedAccuracies.reduce(0, +) / Double(k)
    
    print("\n📊 K-Fold Results Summary:")
    print("   Average Accuracy: \(String(format: "%.4f", avgAccuracy))")
    print("   Average Balanced Accuracy: \(String(format: "%.4f", avgBalancedAccuracy))")
    print("   Accuracy Std Dev: \(String(format: "%.4f", calculateStandardDeviation(accuracies)))")
    
    return CrossValidationResult(
        accuracy: avgAccuracy,
        balancedAccuracy: avgBalancedAccuracy,
        parameters: parameters,
        confusionMatrix: lastConfusion,
        classMetrics: lastClassMetrics
    )
}

func calculateStandardDeviation(_ values: [Double]) -> Double {
    let mean = values.reduce(0, +) / Double(values.count)
    let variance = values.map { pow($0 - mean, 2) }.reduce(0, +) / Double(values.count)
    return sqrt(variance)
}

// ==========================
// MARK: - Grid Search (balanced selection + macro-F1 tie-break + time budget)
// ==========================
func performGridSearch(
    trainingData: DataFrame,
    validationData: DataFrame
) throws -> (bestParameters: MLBoostedTreeClassifier.ModelParameters, bestMetrics: MetricsResult) {
    let maxDepths = TuningConfig.maxDepths
    let maxIterations = TuningConfig.maxIterations
    var bestMetrics: MetricsResult?
    var bestParameters = MLBoostedTreeClassifier.ModelParameters(
        maxDepth: maxDepths.first ?? 6,
        maxIterations: maxIterations.first ?? 50
    )
    
    print("🔍 Starting Grid Search with \(maxDepths.count * maxIterations.count) combinations...")
    print("🔍 Training samples: \(trainingData.rows.count), Validation samples: \(validationData.rows.count)")
    
    // Precompute stable inputs to avoid repeated work and allocations
    let startTime = Date()
    let timeBudget = TuningConfig.gridTimeBudgetSeconds
    let valTruths = extractLabels(from: validationData)
    
    // Oversample the training split ONCE to avoid repeated large allocations
    print("🔁 Preparing balanced training set once for all combinations...")
    let balancedTrainOnce = oversampleTrainingData(trainingData)
    
    // Check if data has valid labels
    let uniqueLabels = Set(trainingData["label"].map { String(describing: $0) })
    print("🏷️ Found \(uniqueLabels.count) unique labels: \(uniqueLabels.sorted())")
    
    var noImproveCount = 0
    var skipDepth = false
outerLoop: for depth in maxDepths {
        skipDepth = false
        for iterations in maxIterations {
            // Time budget guard
            if Date().timeIntervalSince(startTime) > timeBudget {
                print("⏱️ Grid search time budget reached (\(Int(timeBudget))s). Stopping early.")
                break outerLoop
            }
            
            autoreleasepool {
                let parameters = MLBoostedTreeClassifier.ModelParameters(
                    maxDepth: depth,
                    maxIterations: iterations
                )
                
                do {
                    let classifier = try MLBoostedTreeClassifier(
                        trainingData: balancedTrainOnce,
                        targetColumn: "label",
                        parameters: parameters
                    )
                    
                    let predictions = try classifier.predictions(from: validationData)
                    var predsArray: [String] = []
                    predsArray.reserveCapacity(valTruths.count)
                    for case let s as String in predictions { predsArray.append(s) }
                    
                    let metrics = calculateMetrics(groundTruth: valTruths, predictions: predsArray)
                    
                    print("   Depth: \(depth) | Iter: \(iterations) | Acc: \(String(format: "%.4f", metrics.accuracy)) | Bal: \(String(format: "%.4f", metrics.balancedAccuracy)) | F1: \(String(format: "%.4f", macroF1(metrics)))")
                    
                    let isBetter: Bool = {
                        guard let b = bestMetrics else { return true }
                        if metrics.balancedAccuracy != b.balancedAccuracy { return metrics.balancedAccuracy > b.balancedAccuracy }
                        let f1m = macroF1(metrics), f1b = macroF1(b)
                        if f1m != f1b { return f1m > f1b }
                        return metrics.accuracy > b.accuracy
                    }()
                    
                    if isBetter {
                        bestMetrics = metrics
                        bestParameters = parameters
                        noImproveCount = 0
                        print("   ⭐️ New best parameters found!")
                    } else {
                        noImproveCount += 1
                        if noImproveCount >= TuningConfig.gridNoImproveEarlyStop {
                            print("🛑 Early stop after \(noImproveCount) non-improving combos for depth \(depth). Moving to next depth.")
                            noImproveCount = 0
                            skipDepth = true
                        }
                    }
                } catch {
                    print("   ❌ Error training with these parameters: \(error)")
                }
            }
            if skipDepth { break }
        }
        // continue with next depth if we decided to skip
    }
    
    guard let finalBestMetrics = bestMetrics else {
        throw NSError(domain: "GridSearchError", code: 1, userInfo: [NSLocalizedDescriptionKey: "No valid model found during grid search"])
    }
    
    return (bestParameters, finalBestMetrics)
}

// ==========================
// MARK: - Caricamento Dati Ottimizzato (streaming, low‑memory)
// ==========================
func loadPreprocessedData(from folderURL: URL) throws -> DataFrame {
    print("📂 Loading data from: \(folderURL.path)")
    
    guard FileManager.default.fileExists(atPath: folderURL.path) else {
        throw NSError(domain: "DataLoadingError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Directory not found: \(folderURL.path)"])
    }
    
    let queue = DispatchQueue(label: "samplesQueue", attributes: .concurrent)
    let group = DispatchGroup()
    let lock = NSLock()
    let semaphore = DispatchSemaphore(value: TuningConfig.ioParallelism)
    
    let labelFolders = try FileManager.default.contentsOfDirectory(
        at: folderURL,
        includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles]
    ).filter { $0.hasDirectoryPath }
    
    print("📂 Found \(labelFolders.count) label folders")
    
    // Global accumulators (low-memory): labels column + per-feature columns
    var labels: [String] = []
    labels.reserveCapacity(4096)
    var featureColumns: [[Double]] = [] // sized lazily when first sample arrives
    var globalFeatureCount: Int = -1
    
    for labelFolder in labelFolders {
        let label = labelFolder.lastPathComponent
        let jsonFiles = try FileManager.default.contentsOfDirectory(
            at: labelFolder,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ).filter { $0.pathExtension == "json" }
        
        print("📂 Processing \(jsonFiles.count) files for label '\(label)'")
        
        let batchSize = TuningConfig.ioBatchSize
        let batches = stride(from: 0, to: jsonFiles.count, by: batchSize).map {
            Array(jsonFiles[$0..<min($0 + batchSize, jsonFiles.count)])
        }
        
        for batch in batches {
            semaphore.wait()
            group.enter()
            queue.async {
                defer {
                    semaphore.signal()
                    group.leave()
                }
                
                // Local batch accumulators
                var localLabels: [String] = []
                localLabels.reserveCapacity(batch.count)
                var localFeatureColumns: [[Double]] = []
                var localFeatureCount: Int = -1
                let decoder = JSONDecoder()
                
                for jsonFile in batch {
                    autoreleasepool {
                        do {
                            let data = try Data(contentsOf: jsonFile, options: .mappedIfSafe)
                            let sample = try decoder.decode(PreprocessedSample.self, from: data)
                            let trimmed = sample.trimmedLabel
                            if localFeatureCount == -1 {
                                localFeatureCount = sample.features.count
                                localFeatureColumns = Array(repeating: [], count: localFeatureCount)
                                // reserve a bit for speed without large peak
                                for i in 0..<localFeatureCount { localFeatureColumns[i].reserveCapacity(64) }
                            } else if sample.features.count != localFeatureCount {
                                print("⚠️ Skipping file with inconsistent feature count (expected \(localFeatureCount), got \(sample.features.count)): \(jsonFile.lastPathComponent)")
                                return
                            }
                            localLabels.append(trimmed)
                            for i in 0..<localFeatureCount { localFeatureColumns[i].append(sample.features[i]) }
                        } catch {
                            print("⚠️ Error processing file \(jsonFile.lastPathComponent): \(error)")
                        }
                    }
                }
                
                lock.lock()
                // Initialize global feature columns once
                if globalFeatureCount == -1 {
                    globalFeatureCount = localFeatureCount
                    featureColumns = Array(repeating: [], count: max(0, globalFeatureCount))
                }
                if localFeatureCount != globalFeatureCount {
                    print("⚠️ Skipping a batch due to mismatched feature count (global \(globalFeatureCount), local \(localFeatureCount))")
                } else {
                    labels.append(contentsOf: localLabels)
                    if globalFeatureCount > 0 {
                        for i in 0..<globalFeatureCount { featureColumns[i].append(contentsOf: localFeatureColumns[i]) }
                    }
                }
                lock.unlock()
            }
        }
    }
    
    group.wait()
    
    print("📂 Loaded \(labels.count) total samples")
    
    guard !labels.isEmpty else {
        throw NSError(domain: "DataLoadingError", code: 2, userInfo: [NSLocalizedDescriptionKey: "No valid samples found"])
    }
    guard globalFeatureCount > 0 else {
        throw NSError(domain: "DataLoadingError", code: 3, userInfo: [NSLocalizedDescriptionKey: "No features found"])
    }
    
    // Create DataFrame with proper Column initialization
    var dataFrame = DataFrame()
    dataFrame.append(column: Column(name: "label", contents: labels))
    
    print("📊 Feature count: \(globalFeatureCount)")
    print("📊 Unique labels: \(Set(labels).sorted())")
    
    for i in 0..<globalFeatureCount {
        dataFrame.append(column: Column(name: "feature_\(i)", contents: featureColumns[i]))
    }
    
    return dataFrame
}

// ==========================
// MARK: - JSON Saving Utilities
// ==========================
func metricsToDictionary(metrics: MetricsResult) -> [String: Any] {
    var classMetricsDict = [String: [String: Double]]()
    for (className, metrics) in metrics.classMetrics {
        classMetricsDict[className] = [
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1
        ]
    }
    
    return [
        "accuracy": metrics.accuracy,
        "balanced_accuracy": metrics.balancedAccuracy,
        "confusion_matrix": metrics.confusionMatrix,
        "class_metrics": classMetricsDict
    ]
}

func saveMetricsToJSON(validationMetrics: MetricsResult, testingMetrics: MetricsResult, filePath: String) {
    let metricsDict: [String: Any] = [
        "validation": metricsToDictionary(metrics: validationMetrics),
        "testing": metricsToDictionary(metrics: testingMetrics)
    ]
    
    do {
        let jsonData = try JSONSerialization.data(withJSONObject: metricsDict, options: .prettyPrinted)
        let jsonURL = URL(fileURLWithPath: filePath)
        try jsonData.write(to: jsonURL)
        print("✅ Metrics saved to: \(filePath)")
    } catch {
        print("❌ Error saving metrics to JSON: \(error)")
    }
}

// ==========================
// MARK: - DataFrame Concatenation (Train+Val for final model)
// ==========================
func concatenateDataFrames(_ frames: [DataFrame]) -> DataFrame {
    precondition(!frames.isEmpty, "No dataframes to concatenate")
    let base = frames[0]
    var out = DataFrame()
    for col in base.columns {
        let name = col.name
        if name == "label" {
            var values: [String] = []
            values.reserveCapacity(frames.reduce(0) { $0 + $1.rows.count })
            for f in frames { values.append(contentsOf: extractLabels(from: f, labelColumn: name)) }
            out.append(column: Column(name: name, contents: values))
        } else {
            var values: [Double] = []
            values.reserveCapacity(frames.reduce(0) { $0 + $1.rows.count })
            for f in frames {
                for i in 0..<f.rows.count {
                    if let v = f[name][i] as? Double { values.append(v) }
                    else { fatalError("Non-Double feature value in column \(name) at row \(i)") }
                }
            }
            out.append(column: Column(name: name, contents: values))
        }
    }
    return out
}

// ==========================
// MARK: - Main
// ==========================
let dataPath = URL(fileURLWithPath: "/Users/matteo/Downloads/dataset/CreateML_JSON")

do {
    print("📥 Caricamento dati preprocessati...")
    
    let trainingPath = dataPath.appendingPathComponent("Training")
    let validationPath = dataPath.appendingPathComponent("Validation")
    let testingPath = dataPath.appendingPathComponent("Testing")
    
    print("🔄 Caricamento training set...")
    let trainingData = try loadPreprocessedData(from: trainingPath)
    
    print("🔄 Caricamento validation set...")
    let validationData = try loadPreprocessedData(from: validationPath)
    
    print("🔄 Caricamento testing set...")
    let testingData = try loadPreprocessedData(from: testingPath)
    
    print("✅ Dati caricati: \(trainingData.rows.count) train, \(validationData.rows.count) val, \(testingData.rows.count) test")
    
    let featureCount = trainingData.columns.filter { $0.name.starts(with: "feature_") }.count
    guard featureCount == 30 else {
        print("❌ ERRORE: Il numero di feature dovrebbe essere 30, ma è \(featureCount)")
        exit(1)
    }
    
    // Validate data quality
    let trainingLabels = Set(trainingData["label"].map { String(describing: $0) })
    let validationLabels = Set(validationData["label"].map { String(describing: $0) })
    let testingLabelsSet = Set(testingData["label"].map { String(describing: $0) })
    
    print("🏷️ Training labels: \(trainingLabels.sorted())")
    print("🏷️ Validation labels: \(validationLabels.sorted())")
    print("🏷️ Testing labels: \(testingLabelsSet.sorted())")
    
    guard !trainingLabels.isEmpty && !validationLabels.isEmpty else {
        print("❌ ERRORE: Labels vuote trovate nei dati")
        exit(1)
    }
    
    // Ensure label sets match across splits (fail fast)
    assertSameLabelSet(train: trainingData, val: validationData, test: testingData)
    
    print("🔎 Avvio Grid Search per trovare i migliori parametri...")
    let (bestParameters, bestValidationMetrics) = try performGridSearch(
        trainingData: trainingData,
        validationData: validationData
    )
    
    print("✨ Migliori parametri trovati:")
    print("   maxDepth: \(bestParameters.maxDepth)")
    print("   maxIterations: \(bestParameters.maxIterations)")
    print("   Balanced Accuracy di validazione: \(String(format: "%.4f", bestValidationMetrics.balancedAccuracy)) | Accuracy: \(String(format: "%.4f", bestValidationMetrics.accuracy))")
    
    if bestValidationMetrics.accuracy > 0 {
        print("🔄 Esecuzione K-Fold Cross Validation (stratified)...")
        let crossValidationResult = try performKFoldCrossValidation(
            data: trainingData,
            k: 5,
            parameters: bestParameters
        )
        
        print("📊 Risultati Cross Validation:")
        print("   Accuracy media: \(String(format: "%.4f", crossValidationResult.accuracy))")
        print("   Balanced Accuracy media: \(String(format: "%.4f", crossValidationResult.balancedAccuracy))")
        
        // Train final model on Train + Val with oversampling for robustness
        print("🤖 Addestramento modello finale (Train+Val) con i migliori parametri...")
        let trainPlusVal = concatenateDataFrames([trainingData, validationData])
        let balancedTrainPlusVal = oversampleTrainingData(trainPlusVal)
        
        var tmpClassifier: MLBoostedTreeClassifier?
        var tmpError: Error?
        autoreleasepool {
            do {
                tmpClassifier = try MLBoostedTreeClassifier(
                    trainingData: balancedTrainPlusVal,
                    targetColumn: "label",
                    parameters: bestParameters
                )
            } catch {
                tmpError = error
            }
        }
        if let e = tmpError { throw e }
        guard let finalClassifier = tmpClassifier else { fatalError("Failed to create final classifier") }
        
        print("🔬 Valutazione finale sul test set...")
        let testingPred = try finalClassifier.predictions(from: testingData)
        let testingLabelsArray = extractLabels(from: testingData)
        var testingPredArray: [String] = []
        testingPredArray.reserveCapacity(testingLabelsArray.count)
        for case let s as String in testingPred { testingPredArray.append(s) }
        
        let finalMetrics = calculateMetrics(groundTruth: testingLabelsArray, predictions: testingPredArray)
        
        print("📈 Metriche finali sul test set:")
        print("   Accuracy: \(String(format: "%.4f", finalMetrics.accuracy))")
        print("   Balanced Accuracy: \(String(format: "%.4f", finalMetrics.balancedAccuracy))")
        
        // Save metrics to JSON
        let metricsPath = "/Users/matteo/Downloads/dataset/metrics.json"
        saveMetricsToJSON(validationMetrics: bestValidationMetrics, testingMetrics: finalMetrics, filePath: metricsPath)
        
        let outputURL = URL(fileURLWithPath: "/Users/matteo/Downloads/dataset/GestureClassifier.mlmodel")
        let metadata = MLModelMetadata(
            author: "Matteo",
            shortDescription: "Modello di classificazione dei gesti ottimizzato con Grid Search, Stratified K-Fold e metriche bilanciate",
            version: "2.1"
        )
        
        try finalClassifier.write(to: outputURL, metadata: metadata)
        print("✅ Modello salvato in: \(outputURL.path)")
    } else {
        print("❌ ERRORE: Grid Search ha fallito - nessun modello valido trovato")
        print("💡 Suggerimenti:")
        print("   - Verificare la qualità dei dati")
        print("   - Controllare che le labels siano corrette")
        print("   - Provare con parametri diversi")
        exit(1)
    }
    
} catch {
    print("❌ Errore durante il training: \(error)")
    if let nsError = error as NSError? {
        print("   Domain: \(nsError.domain)")
        print("   Code: \(nsError.code)")
        if let description = nsError.userInfo[NSLocalizedDescriptionKey] as? String {
            print("   Description: \(description)")
        }
    }
    exit(1)
}
