"""
Hand tracking module using MediaPipe for real-time hand detection and landmark extraction.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class HandTracker:
    """Real-time hand tracking using MediaPipe."""
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 max_num_hands: int = 2,
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the hand tracker.
        
        Args:
            static_image_mode: Whether to treat input as static images
            max_num_hands: Maximum number of hands to detect
            model_complexity: Complexity of the hand landmark model (0-1)
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def detect_hands(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Detect hands in the input image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Tuple of (annotated_image, hand_landmarks_list)
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        
        # Process the image
        results = self.hands.process(rgb_image)
        
        # Convert back to BGR
        rgb_image.flags.writeable = True
        annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        hand_landmarks_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract landmark coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                
                hand_landmarks_list.append({
                    'landmarks': landmarks,
                    'raw': hand_landmarks
                })
        
        return annotated_image, hand_landmarks_list
    
    def get_finger_positions(self, landmarks: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Extract finger tip and joint positions from landmarks.
        
        Args:
            landmarks: List of landmark dictionaries
            
        Returns:
            Dictionary with finger positions
        """
        if len(landmarks) < 21:
            return {}
        
        # MediaPipe hand landmark indices
        finger_tips = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }
        
        finger_positions = {}
        for finger, tip_idx in finger_tips.items():
            finger_positions[finger] = landmarks[tip_idx]
        
        return finger_positions
    
    def calculate_distances(self, landmarks: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate distances between key landmarks for gesture recognition.
        
        Args:
            landmarks: List of landmark dictionaries
            
        Returns:
            Dictionary with calculated distances
        """
        if len(landmarks) < 21:
            return {}
        
        # Key landmark indices
        thumb_tip = 4
        index_tip = 8
        middle_tip = 12
        ring_tip = 16
        pinky_tip = 20
        wrist = 0
        
        def euclidean_distance(p1: Dict[str, float], p2: Dict[str, float]) -> float:
            return np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
        
        distances = {
            'thumb_index': euclidean_distance(landmarks[thumb_tip], landmarks[index_tip]),
            'thumb_middle': euclidean_distance(landmarks[thumb_tip], landmarks[middle_tip]),
            'index_middle': euclidean_distance(landmarks[index_tip], landmarks[middle_tip]),
            'thumb_wrist': euclidean_distance(landmarks[thumb_tip], landmarks[wrist]),
            'index_wrist': euclidean_distance(landmarks[index_tip], landmarks[wrist]),
        }
        
        return distances
    
    def close(self):
        """Clean up resources."""
        if self.hands:
            self.hands.close()