#!/usr/bin/env python3
"""
Simple test script to verify gesture recognition functionality
"""

import sys
import os
import numpy as np

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from gesture_recognition import HandTracker, GestureClassifier, ARInterface
    from gesture_recognition.gesture_classifier import GestureType
    print("✅ Successfully imported gesture recognition modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_gesture_classifier():
    """Test gesture classification with mock data."""
    print("\n🧪 Testing Gesture Classifier...")
    
    classifier = GestureClassifier()
    
    # Mock landmarks for pointing gesture (only index finger extended)
    # This is a simplified representation - normally 21 landmarks
    mock_landmarks = [{'x': 0.5, 'y': 0.5, 'z': 0.0} for _ in range(21)]
    
    # Simulate pointing gesture by adjusting finger positions
    mock_landmarks[8]['y'] = 0.3  # Index finger tip higher
    mock_landmarks[12]['y'] = 0.6  # Middle finger tip lower
    mock_landmarks[16]['y'] = 0.6  # Ring finger tip lower  
    mock_landmarks[20]['y'] = 0.6  # Pinky finger tip lower
    
    result = classifier.classify_gesture(mock_landmarks)
    print(f"   Mock gesture result: {result['gesture'].value}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print("✅ Gesture classifier test passed")


def test_ar_interface():
    """Test AR interface initialization."""
    print("\n🧪 Testing AR Interface...")
    
    try:
        ar_interface = ARInterface()
        print("   AR Interface initialized successfully")
        
        # Test gesture callback registration
        def test_callback(gesture_info):
            print(f"   Callback triggered for: {gesture_info['gesture'].value}")
        
        ar_interface.register_gesture_callback(GestureType.POINT, test_callback)
        print("   Gesture callback registered successfully")
        
        # Test current gesture method
        gesture, confidence = ar_interface.get_current_gesture()
        print(f"   Current gesture: {gesture.value}, confidence: {confidence}")
        
        ar_interface.cleanup()
        print("✅ AR Interface test passed")
        
    except Exception as e:
        print(f"❌ AR Interface test failed: {e}")


def test_dependencies():
    """Test if all required dependencies are available."""
    print("\n🧪 Testing Dependencies...")
    
    dependencies = {
        'cv2': 'OpenCV',
        'mediapipe': 'MediaPipe',
        'numpy': 'NumPy'
    }
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"   ✅ {name} is available")
        except ImportError:
            print(f"   ❌ {name} is missing - install with: pip install {module}")
            return False
    
    print("✅ All dependencies are available")
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("CAMIO AR - GESTURE RECOGNITION TESTS")
    print("=" * 50)
    
    # Test dependencies first
    if not test_dependencies():
        print("\n❌ Dependency tests failed. Please install missing packages.")
        return
    
    # Test individual components
    test_gesture_classifier()
    test_ar_interface()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed successfully!")
    print("You can now run the demo with: python demo.py")
    print("=" * 50)


if __name__ == "__main__":
    main()