#!/usr/bin/env python3
"""
Camio AR Gesture Recognition Demo
Demonstrates the gesture recognition capabilities for AR applications.
"""

import sys
import os
import time

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gesture_recognition import HandTracker, GestureClassifier, ARInterface
from gesture_recognition.gesture_classifier import GestureType


def on_point_gesture(gesture_info):
    """Callback for pointing gesture."""
    print(f"👉 Point gesture detected! Confidence: {gesture_info['confidence']:.2f}")


def on_thumbs_up_gesture(gesture_info):
    """Callback for thumbs up gesture."""
    print(f"👍 Thumbs up! Great job! Confidence: {gesture_info['confidence']:.2f}")


def on_fist_gesture(gesture_info):
    """Callback for fist gesture."""
    print(f"✊ Fist detected! Power mode activated! Confidence: {gesture_info['confidence']:.2f}")


def on_open_palm_gesture(gesture_info):
    """Callback for open palm gesture."""
    print(f"✋ Open palm - Stop command received! Confidence: {gesture_info['confidence']:.2f}")


def on_peace_gesture(gesture_info):
    """Callback for peace gesture."""
    print(f"✌️ Peace sign detected! Confidence: {gesture_info['confidence']:.2f}")


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("CAMIO AR - GESTURE RECOGNITION SYSTEM")
    print("=" * 60)
    print()
    print("This demo showcases gesture recognition capabilities for AR applications.")
    print("Supported gestures:")
    print("  👉 Point - Select or indicate direction")
    print("  ✋ Open Palm - Stop or show command")
    print("  ✊ Fist - Grab or power gesture")
    print("  👍 Thumbs Up - Approval or confirmation")
    print("  ✌️ Peace Sign - Victory or two items")
    print("  👌 OK Sign - Confirmation")
    print("  🤏 Pinch - Precise selection")
    print()
    print("Instructions:")
    print("  - Position your hand in front of the camera")
    print("  - Try different gestures to see real-time recognition")
    print("  - Press 'q' to quit the demo")
    print()
    input("Press Enter to start the demo...")
    
    # Initialize AR interface
    ar_interface = ARInterface()
    
    # Register gesture callbacks
    ar_interface.register_gesture_callback(GestureType.POINT, on_point_gesture)
    ar_interface.register_gesture_callback(GestureType.THUMBS_UP, on_thumbs_up_gesture)
    ar_interface.register_gesture_callback(GestureType.FIST, on_fist_gesture)
    ar_interface.register_gesture_callback(GestureType.OPEN_PALM, on_open_palm_gesture)
    ar_interface.register_gesture_callback(GestureType.PEACE, on_peace_gesture)
    
    try:
        # Run the real-time gesture recognition
        ar_interface.run_realtime("Camio AR - Gesture Recognition Demo")
        
    except Exception as e:
        print(f"Error running demo: {e}")
        
    finally:
        ar_interface.cleanup()
        print("\nDemo finished. Thank you for trying Camio AR Gesture Recognition!")


if __name__ == "__main__":
    main()