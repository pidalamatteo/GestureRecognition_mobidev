# Camio AR Gesture Recognition System

A comprehensive gesture recognition system designed for augmented reality (AR) applications. This system provides real-time hand tracking and gesture classification capabilities using computer vision and machine learning.

## Features

- **Real-time Hand Tracking**: Uses MediaPipe for accurate hand landmark detection
- **Gesture Recognition**: Supports 7+ common gestures useful for AR interactions
- **AR Integration**: Designed specifically for augmented reality applications
- **Easy Integration**: Simple API for embedding in AR projects
- **Customizable**: Configurable confidence thresholds and gesture callbacks

## Supported Gestures

| Gesture | Description | AR Use Case |
|---------|-------------|-------------|
| 👉 Point | Index finger extended | Selection, direction indication |
| ✋ Open Palm | All fingers extended | Stop command, show gesture |
| ✊ Fist | All fingers closed | Grab action, power gesture |
| 👍 Thumbs Up | Only thumb extended | Confirmation, approval |
| ✌️ Peace Sign | Index and middle fingers extended | Victory, number two |
| 👌 OK Sign | Thumb and index forming circle | Confirmation, perfect |
| 🤏 Pinch | Thumb and index close together | Precise selection, zoom |

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pidalamatteo/GestureRecognition_mobidev.git
cd GestureRecognition_mobidev
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Running the Demo

```bash
python demo.py
```

This will start the gesture recognition demo with camera input and real-time visualization.

### Basic Usage

```python
from gesture_recognition import ARInterface, GestureType

# Initialize the AR interface
ar_interface = ARInterface()

# Register gesture callbacks
def on_point_gesture(gesture_info):
    print(f"Point gesture detected: {gesture_info['confidence']:.2f}")

ar_interface.register_gesture_callback(GestureType.POINT, on_point_gesture)

# Start real-time recognition
ar_interface.run_realtime()
```

### Advanced Usage

```python
from gesture_recognition import HandTracker, GestureClassifier
import cv2

# Initialize components
hand_tracker = HandTracker()
gesture_classifier = GestureClassifier()

# Process single frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Detect hands and classify gesture
annotated_frame, hand_landmarks = hand_tracker.detect_hands(frame)
if hand_landmarks:
    gesture_result = gesture_classifier.classify_gesture(hand_landmarks[0]['landmarks'])
    print(f"Detected: {gesture_result['gesture'].value} ({gesture_result['confidence']:.2f})")
```

## API Reference

### ARInterface

Main interface for AR gesture recognition.

#### Methods

- `start_camera()`: Initialize camera capture
- `process_frame(frame)`: Process single frame for gesture recognition
- `register_gesture_callback(gesture_type, callback)`: Register gesture event handler
- `run_realtime()`: Start real-time gesture recognition
- `get_current_gesture()`: Get current gesture and confidence
- `cleanup()`: Clean up resources

### HandTracker

Hand detection and landmark extraction.

#### Methods

- `detect_hands(image)`: Detect hands in image and return landmarks
- `get_finger_positions(landmarks)`: Extract finger tip positions
- `calculate_distances(landmarks)`: Calculate distances between key points

### GestureClassifier

Gesture classification from hand landmarks.

#### Methods

- `classify_gesture(landmarks)`: Classify gesture from landmarks
- `get_gesture_description(gesture_type)`: Get human-readable gesture description

## Configuration

### Camera Settings

```python
ar_interface = ARInterface(
    camera_index=0,          # Camera device index
    frame_width=640,         # Camera frame width
    frame_height=480         # Camera frame height
)
```

### Hand Tracking Settings

```python
hand_tracker = HandTracker(
    max_num_hands=2,                    # Maximum hands to detect
    min_detection_confidence=0.7,       # Detection confidence threshold
    min_tracking_confidence=0.5         # Tracking confidence threshold
)
```

### Gesture Classification Settings

```python
gesture_classifier = GestureClassifier(
    confidence_threshold=0.7            # Minimum confidence for recognition
)
```

## Integration with AR Applications

### Unity Integration Example

```csharp
// C# wrapper for Unity
public class GestureRecognitionManager : MonoBehaviour
{
    private Process pythonProcess;
    
    void Start()
    {
        // Start Python gesture recognition process
        StartPythonGestureRecognition();
    }
    
    private void StartPythonGestureRecognition()
    {
        // Launch Python script and read gesture events
        // Implement IPC communication (TCP, named pipes, etc.)
    }
}
```

### Web AR Integration

```javascript
// JavaScript integration for WebAR
class GestureRecognitionAPI {
    constructor() {
        this.websocket = new WebSocket('ws://localhost:8080/gestures');
    }
    
    onGestureDetected(callback) {
        this.websocket.onmessage = (event) => {
            const gesture = JSON.parse(event.data);
            callback(gesture);
        };
    }
}
```

## Performance Optimization

- **Camera Resolution**: Lower resolution (640x480) for better performance
- **Model Complexity**: Use model_complexity=0 for faster processing
- **Hand Tracking**: Limit max_num_hands to 1 if only single-hand gestures needed
- **Frame Rate**: Process every nth frame for resource-constrained devices

## Troubleshooting

### Common Issues

1. **Camera not detected**: Check camera index and permissions
2. **Low accuracy**: Ensure good lighting and clear hand visibility
3. **High latency**: Reduce camera resolution or model complexity
4. **Import errors**: Verify all dependencies are installed correctly

### Debug Mode

Enable debug output for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **MediaPipe**: Google's MediaPipe framework for hand tracking
- **OpenCV**: Computer vision library for image processing
- **NumPy**: Numerical computing library for efficient calculations

## Future Enhancements

- [ ] Support for two-handed gestures
- [ ] Custom gesture training interface
- [ ] 3D gesture recognition for depth cameras
- [ ] Mobile AR SDK integration (ARCore/ARKit)
- [ ] WebRTC streaming for remote AR applications
- [ ] Machine learning model optimization for edge devices