"""
AR Interface module for integrating gesture recognition with AR applications.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
from .hand_tracker import HandTracker
from .gesture_classifier import GestureClassifier, GestureType


class ARInterface:
    """Interface for integrating gesture recognition with AR applications."""
    
    def __init__(self, 
                 camera_index: int = 0,
                 frame_width: int = 640,
                 frame_height: int = 480):
        """
        Initialize the AR interface.
        
        Args:
            camera_index: Camera device index
            frame_width: Camera frame width
            frame_height: Camera frame height
        """
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Initialize components
        self.hand_tracker = HandTracker()
        self.gesture_classifier = GestureClassifier()
        
        # Camera setup
        self.cap = None
        self.is_running = False
        
        # Gesture callbacks
        self.gesture_callbacks = {}
        self.current_gesture = GestureType.NONE
        self.gesture_confidence = 0.0
        
        # AR overlay settings
        self.show_landmarks = True
        self.show_gesture_info = True
        
    def start_camera(self) -> bool:
        """
        Start the camera capture.
        
        Returns:
            True if camera started successfully
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            if not self.cap.isOpened():
                return False
                
            self.is_running = True
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop the camera capture."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def register_gesture_callback(self, gesture_type: GestureType, callback: Callable):
        """
        Register a callback function for a specific gesture.
        
        Args:
            gesture_type: The gesture type to respond to
            callback: Function to call when gesture is detected
        """
        self.gesture_callbacks[gesture_type] = callback
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single frame for gesture recognition.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Tuple of (processed_frame, gesture_info)
        """
        # Detect hands
        annotated_frame, hand_landmarks_list = self.hand_tracker.detect_hands(frame)
        
        gesture_info = {
            'gesture': GestureType.NONE,
            'confidence': 0.0,
            'description': 'No hands detected',
            'hand_count': len(hand_landmarks_list)
        }
        
        # Process each detected hand
        if hand_landmarks_list:
            # Use the first detected hand for gesture recognition
            landmarks = hand_landmarks_list[0]['landmarks']
            gesture_result = self.gesture_classifier.classify_gesture(landmarks)
            
            gesture_info.update(gesture_result)
            self.current_gesture = gesture_result['gesture']
            self.gesture_confidence = gesture_result['confidence']
            
            # Trigger gesture callback if confidence is high enough
            if (self.gesture_confidence > 0.7 and 
                self.current_gesture in self.gesture_callbacks):
                self.gesture_callbacks[self.current_gesture](gesture_info)
        
        # Add AR overlays
        if self.show_gesture_info:
            annotated_frame = self._add_gesture_overlay(annotated_frame, gesture_info)
        
        return annotated_frame, gesture_info
    
    def run_realtime(self, window_name: str = "Camio AR - Gesture Recognition"):
        """
        Run real-time gesture recognition with camera feed.
        
        Args:
            window_name: Name of the display window
        """
        if not self.start_camera():
            print("Failed to start camera")
            return
        
        print("Starting gesture recognition. Press 'q' to quit.")
        print("Available gestures: Point, Open Palm, Fist, Thumbs Up, Peace, OK Sign, Pinch")
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame, gesture_info = self.process_frame(frame)
                
                # Display frame
                cv2.imshow(window_name, processed_frame)
                
                # Check for quit command
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.stop_camera()
    
    def _add_gesture_overlay(self, frame: np.ndarray, gesture_info: Dict[str, Any]) -> np.ndarray:
        """
        Add gesture information overlay to the frame.
        
        Args:
            frame: Input frame
            gesture_info: Gesture information dictionary
            
        Returns:
            Frame with overlay
        """
        overlay_frame = frame.copy()
        
        # Add background rectangle for text
        cv2.rectangle(overlay_frame, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(overlay_frame, (10, 10), (400, 120), (255, 255, 255), 2)
        
        # Add gesture information text
        gesture_name = gesture_info['gesture'].value if gesture_info['gesture'] != GestureType.NONE else "None"
        confidence = gesture_info['confidence']
        hand_count = gesture_info['hand_count']
        
        text_lines = [
            f"Gesture: {gesture_name}",
            f"Confidence: {confidence:.2f}",
            f"Hands: {hand_count}",
            f"Description: {gesture_info['description'][:30]}..."
        ]
        
        for i, line in enumerate(text_lines):
            y_pos = 30 + i * 20
            cv2.putText(overlay_frame, line, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add gesture confidence bar
        if confidence > 0:
            bar_width = int(300 * confidence)
            bar_color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
            cv2.rectangle(overlay_frame, (15, 95), (15 + bar_width, 105), bar_color, -1)
            cv2.rectangle(overlay_frame, (15, 95), (315, 105), (255, 255, 255), 1)
        
        return overlay_frame
    
    def get_current_gesture(self) -> Tuple[GestureType, float]:
        """
        Get the currently detected gesture and confidence.
        
        Returns:
            Tuple of (gesture_type, confidence)
        """
        return self.current_gesture, self.gesture_confidence
    
    def set_display_options(self, show_landmarks: bool = True, show_gesture_info: bool = True):
        """
        Configure display options.
        
        Args:
            show_landmarks: Whether to show hand landmarks
            show_gesture_info: Whether to show gesture information
        """
        self.show_landmarks = show_landmarks
        self.show_gesture_info = show_gesture_info
    
    def capture_gesture_data(self, duration_seconds: int = 5) -> List[Dict[str, Any]]:
        """
        Capture gesture data for a specified duration.
        
        Args:
            duration_seconds: Duration to capture data
            
        Returns:
            List of gesture data frames
        """
        if not self.start_camera():
            return []
        
        import time
        start_time = time.time()
        gesture_data = []
        
        try:
            while time.time() - start_time < duration_seconds:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                _, gesture_info = self.process_frame(frame)
                
                gesture_data.append({
                    'timestamp': time.time() - start_time,
                    'gesture_info': gesture_info
                })
                
        finally:
            self.stop_camera()
        
        return gesture_data
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_camera()
        if self.hand_tracker:
            self.hand_tracker.close()