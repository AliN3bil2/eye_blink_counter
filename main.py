import cv2
import numpy as np
from detector.mediapipe_detector import MediapipeDetector
import warnings

if __name__ == "__main__":
    detector = MediapipeDetector()
    cap = cv2.VideoCapture(r"WIN_20231220_17_58_13_Pro.mp4")
    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break
        success, img = cap.read()
        if success:
            img, left_eye_state, right_eye_state = detector.detect(img, visualise=True)
            print(left_eye_state)
            cv2.imshow('img',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            warnings.warn("No image found !!")
    
    cap.release()
    cv2.destroyAllWindows()



