import cv2
import numpy as np
import os

def test_basic_functionality():
    """Test basic face detection functionality"""
    print("Testing basic face detection functionality...")
    
    # Test 1: Check if OpenCV is working
    print("✓ OpenCV version:", cv2.__version__)
    
    # Test 2: Check if cascade classifiers are available
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    if not face_cascade.empty():
        print("✓ Face cascade classifier loaded successfully")
    else:
        print("✗ Failed to load face cascade classifier")
        return False
    
    if not eye_cascade.empty():
        print("✓ Eye cascade classifier loaded successfully")
    else:
        print("✗ Failed to load eye cascade classifier")
    
    if not smile_cascade.empty():
        print("✓ Smile cascade classifier loaded successfully")
    else:
        print("✗ Failed to load smile cascade classifier")
    
    # Test 3: Check if DNN module is available
    try:
        net = cv2.dnn.readNet()
        print("✓ DNN module is available")
    except Exception as e:
        print("✗ DNN module not available:", e)
    
    # Test 4: Check if webcam can be accessed
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✓ Webcam is accessible")
        cap.release()
    else:
        print("✗ Webcam is not accessible")
    
    print("\nBasic functionality test completed!")
    return True

if __name__ == "__main__":
    test_basic_functionality() 