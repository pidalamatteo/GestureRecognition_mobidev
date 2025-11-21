"""
Gesture Recognition Module for Camio AR
Provides hand tracking and gesture recognition capabilities for augmented reality applications.
"""

from .hand_tracker import HandTracker
from .gesture_classifier import GestureClassifier
from .ar_interface import ARInterface

__version__ = "1.0.0"
__all__ = ["HandTracker", "GestureClassifier", "ARInterface"]