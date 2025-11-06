//
//  CameraFeedService.swift
//  mobidev_exam2
//
//  Created by Matteo on 09/09/25.
//

//prende i frame della camera in tempo reale
import AVFoundation
import UIKit

//sottoscrive -> cameraManager
protocol CameraFeedServiceDelegate: AnyObject {
    func didOutput(sampleBuffer: CMSampleBuffer, orientation: UIImage.Orientation)
    func sessionWasInterrupted()
    func sessionInterruptionEnded()
    func didEncounterSessionRuntimeError()
}

class CameraFeedService: NSObject {
    var cameraPosition: AVCaptureDevice.Position = .front
    private let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "CameraFeedSessionQueue")
    
    var session: AVCaptureSession {
        return captureSession
    }
    weak var delegate: CameraFeedServiceDelegate?

    private var videoOutput: AVCaptureVideoDataOutput?

    //6
    func startSession() {
        sessionQueue.async {
            //configura la sessione se non è configurata
            self.configureSession(to: self.cameraPosition)
            //8 avvia la sessione
            self.captureSession.startRunning()
        }
    }

    func stopSession() {
        sessionQueue.async {
            if self.captureSession.isRunning {
                self.captureSession.stopRunning()
            }
        }
    }
    
    
    //7 Configura la session e le impostazioni della camera
    private func configureSession(to cameraPosition: AVCaptureDevice.Position) {
        guard captureSession.inputs.isEmpty else { return }

        captureSession.beginConfiguration()
        captureSession.sessionPreset = .high

        // Input: fotocamera posteriore - collego la camera alla sessione
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video,
                                                   position: cameraPosition),
            let input = try? AVCaptureDeviceInput(device: device),
            captureSession.canAddInput(input) else {
            print("Errore: impossibile aggiungere input camera")
            captureSession.commitConfiguration()
            return
        }
        captureSession.addInput(input)

        // Output - collego output video alla sessione
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
                                    kCVPixelFormatType_32BGRA]
        //ogni frame catturato -> captureOutput
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "VideoOutputQueue"))
        guard captureSession.canAddOutput(output) else {
            print("Errore: impossibile aggiungere output")
            captureSession.commitConfiguration()
            return
        }
        captureSession.addOutput(output)
        videoOutput = output

        captureSession.commitConfiguration()
    }
   
    
    func switchCamera() {
        sessionQueue.async {
            guard let currentInput = self.captureSession.inputs.first as? AVCaptureDeviceInput else { return }

            self.captureSession.beginConfiguration()
            self.captureSession.removeInput(currentInput)

            // cambia posizione
            self.cameraPosition = (self.cameraPosition == .front) ? .back : .front

            // riusa configureSession passando cameraPosition
            self.configureSession(to: self.cameraPosition)

            self.captureSession.commitConfiguration()
        }
    }
}

extension CameraFeedService: AVCaptureVideoDataOutputSampleBufferDelegate {
    //9 chiamato in automatico
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
    
        let orientation = currentOrientation(for: cameraPosition)
        //avvisa il delegate -> cameraManager
        delegate?.didOutput(sampleBuffer: sampleBuffer, orientation: orientation)
    }
    
    func currentOrientation(for position: AVCaptureDevice.Position) -> UIImage.Orientation{
        switch UIDevice.current.orientation {
        case .portrait: return position == .front ? .leftMirrored: .right
        case .portraitUpsideDown: return position == .front ? .rightMirrored: .left
        case .landscapeLeft: return position == .front ? .downMirrored: .up
        case .landscapeRight: return position == .front ? .upMirrored: .down
        default: return .up
        }
    }
}

