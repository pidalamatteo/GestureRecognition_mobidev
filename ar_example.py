#!/usr/bin/env python3
"""
Simple gesture recognition example for AR integration
"""

import sys
import os
import time

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gesture_recognition import ARInterface, GestureType


def simple_ar_example():
    """Simple example showing how to integrate gesture recognition in AR applications."""
    print("Simple AR Gesture Recognition Example")
    print("=====================================")
    
    # Initialize the AR interface
    ar_interface = ARInterface()
    
    # Define gesture callbacks for AR interactions
    def on_point(gesture_info):
        print(f"🎯 AR Action: Object selected at point location (confidence: {gesture_info['confidence']:.2f})")
    
    def on_open_palm(gesture_info):
        print(f"🛑 AR Action: Menu opened / Stop interaction (confidence: {gesture_info['confidence']:.2f})")
    
    def on_fist(gesture_info):
        print(f"✊ AR Action: Object grabbed (confidence: {gesture_info['confidence']:.2f})")
    
    def on_thumbs_up(gesture_info):
        print(f"👍 AR Action: Confirmed selection (confidence: {gesture_info['confidence']:.2f})")
    
    def on_pinch(gesture_info):
        print(f"🤏 AR Action: Zoom/Scale operation (confidence: {gesture_info['confidence']:.2f})")
    
    # Register callbacks
    ar_interface.register_gesture_callback(GestureType.POINT, on_point)
    ar_interface.register_gesture_callback(GestureType.OPEN_PALM, on_open_palm)
    ar_interface.register_gesture_callback(GestureType.FIST, on_fist)
    ar_interface.register_gesture_callback(GestureType.THUMBS_UP, on_thumbs_up)
    ar_interface.register_gesture_callback(GestureType.PINCH, on_pinch)
    
    print("\nAR Gesture Commands:")
    print("  👉 Point - Select objects")
    print("  ✋ Open Palm - Open menu / Stop")
    print("  ✊ Fist - Grab objects")
    print("  👍 Thumbs Up - Confirm actions")
    print("  🤏 Pinch - Zoom/Scale")
    print("\nPress 'q' in the camera window to quit...")
    
    try:
        # Start the AR gesture recognition
        ar_interface.run_realtime("AR Gesture Control")
    except KeyboardInterrupt:
        print("\nShutting down AR system...")
    finally:
        ar_interface.cleanup()


if __name__ == "__main__":
    simple_ar_example()