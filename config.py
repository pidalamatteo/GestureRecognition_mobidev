"""
Configuration settings for Camio AR Gesture Recognition
"""

# Camera settings
CAMERA_CONFIG = {
    'default_camera_index': 0,
    'default_frame_width': 640,
    'default_frame_height': 480,
    'flip_horizontal': True  # Mirror effect for natural interaction
}

# Hand tracking settings
HAND_TRACKING_CONFIG = {
    'static_image_mode': False,
    'max_num_hands': 2,
    'model_complexity': 1,  # 0-1, higher = more accurate but slower
    'min_detection_confidence': 0.7,
    'min_tracking_confidence': 0.5
}

# Gesture recognition settings
GESTURE_CONFIG = {
    'confidence_threshold': 0.7,
    'gesture_hold_time': 0.5,  # seconds to hold gesture for stable detection
    'distance_threshold_pinch': 0.03,
    'distance_threshold_ok': 0.05
}

# Display settings
DISPLAY_CONFIG = {
    'show_landmarks': True,
    'show_gesture_info': True,
    'overlay_color': (255, 255, 255),
    'overlay_background': (0, 0, 0),
    'text_font': 'cv2.FONT_HERSHEY_SIMPLEX',
    'text_scale': 0.5,
    'text_thickness': 1
}

# AR integration settings
AR_CONFIG = {
    'coordinate_system': 'normalized',  # 'normalized' or 'pixel'
    'gesture_smoothing': True,
    'multi_hand_support': True,
    'gesture_callbacks_enabled': True
}