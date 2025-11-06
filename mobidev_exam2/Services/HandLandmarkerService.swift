import Foundation
import MediaPipeTasksVision
import AVFoundation

// Delegate per comunicare i risultati del rilevamento
protocol HandLandmarkerServiceLiveStreamDelegate: AnyObject {
    func didDetectHands(_ result: HandLandmarkerResult)
}

class HandLandmarkerServiceLiveStream: NSObject {
    
    private var handLandmarker: HandLandmarker?
    weak var delegate: HandLandmarkerServiceLiveStreamDelegate?
    
    init(modelPath: String,
         numHands: Int,
         minHandDetectionConfidence: Float,
         minHandPresenceConfidence: Float,
         minTrackingConfidence: Float,
         computeDelegate: Delegate = .CPU
    ) {
        super.init()
        do {
            let options = HandLandmarkerOptions()
            options.baseOptions.modelAssetPath = modelPath
            options.baseOptions.delegate = computeDelegate
            options.numHands = numHands
            options.minHandDetectionConfidence = minHandDetectionConfidence
            options.minHandPresenceConfidence = minHandPresenceConfidence
            options.minTrackingConfidence = minTrackingConfidence
            options.runningMode = .liveStream
            options.handLandmarkerLiveStreamDelegate = self
            
            handLandmarker = try HandLandmarker(options: options)
        } catch {
            print("Errore inizializzazione HandLandmarker: \(error)")
        }
    }
    
    // MARK: - Rilevamento mani in async
    //13
    func detectAsync(sampleBuffer: CMSampleBuffer, orientation: UIImage.Orientation) {
        //chiamo detectAsync di mediapipe
        guard
            let handLandmarker = handLandmarker,
            let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        else { return }
        
        do {
            let mpImage = try MPImage(pixelBuffer: pixelBuffer, orientation: orientation)
            //eseguito in background per rilevare le mani -> metodo di mediapipe
            //dopo averle rilevate -> func handLandmarker
            try handLandmarker.detectAsync(
                image: mpImage,
                timestampInMilliseconds: Int(Date().timeIntervalSince1970 * 1000)
            )
        } catch {
            print("Errore nel detectAsync: \(error)")
        }
    }
}

// MARK: - HandLandmarkerLiveStreamDelegate
extension HandLandmarkerServiceLiveStream: HandLandmarkerLiveStreamDelegate {
    //14 viene chiamato alla fine del rilevamento da mediapipe
    func handLandmarker(
        _ handLandmarker: HandLandmarker,
        didFinishDetection result: HandLandmarkerResult?,
        timestampInMilliseconds: Int,
        error: Error?
    ) {
        if let error = error {
            print("Errore rilevamento mano: \(error)")
            return
        }
        guard let result = result else { return }
        //mi ritorna un result HandLandmarkerResult che passo a hlmvm.diddetecthands
        delegate?.didDetectHands(result)
    }
}
