"""
Gesture classification module for recognizing common AR gestures.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from enum import Enum


class GestureType(Enum):
    """Enumeration of supported gesture types."""
    NONE = "none"
    POINT = "point"
    OPEN_PALM = "open_palm"
    FIST = "fist"
    THUMBS_UP = "thumbs_up"
    PEACE = "peace"
    OK_SIGN = "ok_sign"
    PINCH = "pinch"


class GestureClassifier:
    """Classifier for common hand gestures used in AR applications."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the gesture classifier.
        
        Args:
            confidence_threshold: Minimum confidence for gesture recognition
        """
        self.confidence_threshold = confidence_threshold
        
    def classify_gesture(self, landmarks: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Classify a gesture based on hand landmarks.
        
        Args:
            landmarks: List of hand landmark dictionaries
            
        Returns:
            Dictionary containing gesture type and confidence
        """
        if len(landmarks) < 21:
            return {
                'gesture': GestureType.NONE,
                'confidence': 0.0,
                'description': 'No valid hand detected'
            }
        
        # Calculate finger states (extended or bent)
        finger_states = self._get_finger_states(landmarks)
        
        # Analyze gesture based on finger patterns
        gesture_result = self._analyze_gesture_pattern(landmarks, finger_states)
        
        return gesture_result
    
    def _get_finger_states(self, landmarks: List[Dict[str, float]]) -> Dict[str, bool]:
        """
        Determine if each finger is extended or bent.
        
        Args:
            landmarks: List of hand landmark dictionaries
            
        Returns:
            Dictionary with finger extension states
        """
        finger_states = {}
        
        # Finger landmark indices (MediaPipe format)
        fingers = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        for finger_name, indices in fingers.items():
            if finger_name == 'thumb':
                # Special handling for thumb (horizontal movement)
                finger_states[finger_name] = self._is_thumb_extended(landmarks, indices)
            else:
                # For other fingers, check if tip is above MCP joint
                tip_y = landmarks[indices[3]]['y']
                mcp_y = landmarks[indices[0]]['y']
                finger_states[finger_name] = tip_y < mcp_y
        
        return finger_states
    
    def _is_thumb_extended(self, landmarks: List[Dict[str, float]], indices: List[int]) -> bool:
        """
        Check if thumb is extended (special case due to thumb orientation).
        
        Args:
            landmarks: List of hand landmark dictionaries
            indices: Thumb joint indices
            
        Returns:
            True if thumb is extended
        """
        # Compare thumb tip distance from wrist vs thumb MCP distance from wrist
        thumb_tip = landmarks[indices[3]]
        thumb_mcp = landmarks[indices[0]]
        wrist = landmarks[0]
        
        tip_to_wrist = np.sqrt((thumb_tip['x'] - wrist['x'])**2 + (thumb_tip['y'] - wrist['y'])**2)
        mcp_to_wrist = np.sqrt((thumb_mcp['x'] - wrist['x'])**2 + (thumb_mcp['y'] - wrist['y'])**2)
        
        return tip_to_wrist > mcp_to_wrist * 1.2
    
    def _analyze_gesture_pattern(self, landmarks: List[Dict[str, float]], 
                                finger_states: Dict[str, bool]) -> Dict[str, Any]:
        """
        Analyze finger pattern to determine gesture type.
        
        Args:
            landmarks: List of hand landmark dictionaries
            finger_states: Dictionary of finger extension states
            
        Returns:
            Dictionary with gesture classification results
        """
        extended_fingers = sum(finger_states.values())
        
        # Pattern matching for common gestures
        if self._is_pointing_gesture(finger_states):
            return {
                'gesture': GestureType.POINT,
                'confidence': 0.9,
                'description': 'Pointing gesture detected'
            }
        elif self._is_open_palm(finger_states):
            return {
                'gesture': GestureType.OPEN_PALM,
                'confidence': 0.9,
                'description': 'Open palm gesture detected'
            }
        elif self._is_fist(finger_states):
            return {
                'gesture': GestureType.FIST,
                'confidence': 0.9,
                'description': 'Fist gesture detected'
            }
        elif self._is_thumbs_up(finger_states):
            return {
                'gesture': GestureType.THUMBS_UP,
                'confidence': 0.8,
                'description': 'Thumbs up gesture detected'
            }
        elif self._is_peace_sign(finger_states):
            return {
                'gesture': GestureType.PEACE,
                'confidence': 0.8,
                'description': 'Peace sign detected'
            }
        elif self._is_ok_sign(landmarks, finger_states):
            return {
                'gesture': GestureType.OK_SIGN,
                'confidence': 0.8,
                'description': 'OK sign detected'
            }
        elif self._is_pinch_gesture(landmarks):
            return {
                'gesture': GestureType.PINCH,
                'confidence': 0.8,
                'description': 'Pinch gesture detected'
            }
        else:
            return {
                'gesture': GestureType.NONE,
                'confidence': 0.0,
                'description': 'No recognized gesture'
            }
    
    def _is_pointing_gesture(self, finger_states: Dict[str, bool]) -> bool:
        """Check if gesture is pointing (only index finger extended)."""
        return (finger_states['index'] and 
                not finger_states['middle'] and 
                not finger_states['ring'] and 
                not finger_states['pinky'])
    
    def _is_open_palm(self, finger_states: Dict[str, bool]) -> bool:
        """Check if gesture is open palm (all fingers extended)."""
        return all(finger_states.values())
    
    def _is_fist(self, finger_states: Dict[str, bool]) -> bool:
        """Check if gesture is fist (all fingers bent)."""
        return not any(finger_states.values())
    
    def _is_thumbs_up(self, finger_states: Dict[str, bool]) -> bool:
        """Check if gesture is thumbs up (only thumb extended)."""
        return (finger_states['thumb'] and 
                not finger_states['index'] and 
                not finger_states['middle'] and 
                not finger_states['ring'] and 
                not finger_states['pinky'])
    
    def _is_peace_sign(self, finger_states: Dict[str, bool]) -> bool:
        """Check if gesture is peace sign (index and middle extended)."""
        return (finger_states['index'] and 
                finger_states['middle'] and 
                not finger_states['ring'] and 
                not finger_states['pinky'])
    
    def _is_ok_sign(self, landmarks: List[Dict[str, float]], finger_states: Dict[str, bool]) -> bool:
        """Check if gesture is OK sign (thumb and index forming circle)."""
        if not finger_states['middle'] or not finger_states['ring'] or not finger_states['pinky']:
            return False
        
        # Check if thumb and index finger tips are close
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = np.sqrt((thumb_tip['x'] - index_tip['x'])**2 + 
                          (thumb_tip['y'] - index_tip['y'])**2)
        
        return distance < 0.05  # Threshold for "touching"
    
    def _is_pinch_gesture(self, landmarks: List[Dict[str, float]]) -> bool:
        """Check if gesture is pinch (thumb and index finger close together)."""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = np.sqrt((thumb_tip['x'] - index_tip['x'])**2 + 
                          (thumb_tip['y'] - index_tip['y'])**2)
        
        return distance < 0.03  # Smaller threshold for pinch
    
    def get_gesture_description(self, gesture_type: GestureType) -> str:
        """
        Get a human-readable description of the gesture.
        
        Args:
            gesture_type: The gesture type enum
            
        Returns:
            Description string
        """
        descriptions = {
            GestureType.NONE: "No gesture detected",
            GestureType.POINT: "Pointing - Use to select or indicate direction",
            GestureType.OPEN_PALM: "Open palm - Stop or show command",
            GestureType.FIST: "Fist - Grab or power gesture",
            GestureType.THUMBS_UP: "Thumbs up - Approval or confirmation",
            GestureType.PEACE: "Peace sign - Victory or two items",
            GestureType.OK_SIGN: "OK sign - Confirmation or perfect",
            GestureType.PINCH: "Pinch - Precise selection or zoom"
        }
        
        return descriptions.get(gesture_type, "Unknown gesture")