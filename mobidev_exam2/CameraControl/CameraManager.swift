//Acquisisco tramite CameraFeedService i frame e li manda a HandLandmarkerService
import AVFoundation
import MediaPipeTasksVision


class CameraManager: NSObject, ObservableObject {
    private let service = CameraFeedService()
    
    //private var handService: HandLandmarkerService?
    @Published var lastSampleBuffer: CMSampleBuffer? = nil
    @Published var session: AVCaptureSession?
    //@Published var detectedHands: HandLandmarkerResult? // singolo risultato
    @Published var cameraPosition: AVCaptureDevice.Position = .front
    
    @Published var lastFrameSize: CGSize? = nil
    @Published var isUsingFrontCamera: Bool = true

    @Published var currentOrientation: UIImage.Orientation = .up


    override init() {
        super.init()
        //inizializzo CameraFeedService
        service.delegate = self
    }
    //3
    func start() {
        //chiano per i permessi
        checkCameraPermission { [weak self] granted in
            guard let self = self else { return }
            if granted {
                //5
                //CameraFeedService
                self.service.startSession()
                self.session = self.service.session
            } else {
                print("Permesso fotocamera negato")
            }
        }
    }

    func stop() {
        service.stopSession()
    }

    func switchCamera() {
        service.switchCamera()
        cameraPosition = (cameraPosition == .front) ? .back : .front
        isUsingFrontCamera = (cameraPosition == .front)
    }

    //4
    private func checkCameraPermission(completion: @escaping (Bool) -> Void) {
        //Accedo allo stato dell'autorizzazione per la camera
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async { completion(granted) }
            }
        default: completion(false)
        }
    }
   

}

// MARK: - CameraFeedServiceDelegate
extension CameraManager: CameraFeedServiceDelegate {
    //10
    func didOutput(sampleBuffer: CMSampleBuffer, orientation: UIImage.Orientation) {
        //riceve frame catturati e li pubblica
        DispatchQueue.main.async {
            if let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
                let width = CVPixelBufferGetWidth(pixelBuffer)
                let height = CVPixelBufferGetHeight(pixelBuffer)
                DispatchQueue.main.async {
                    self.lastFrameSize = CGSize(width: width, height: height)
                    self.lastSampleBuffer = sampleBuffer
                    self.currentOrientation = orientation
                }
            }

        }
    }

    func sessionWasInterrupted() { print("Sessione interrotta") }
    func sessionInterruptionEnded() { print("Sessione ripresa") }
    func didEncounterSessionRuntimeError() { print("Errore runtime") }
}
