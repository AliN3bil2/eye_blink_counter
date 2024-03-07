import cv2
import numpy as np
from mediapipe_detector import MediapipeDetector
import warnings

if __name__ == "__main__":
    detect = MediapipeDetector()
    cap = cv2.VideoCapture(r"G:\data_for_drawssy\IMG_2597.MOV")
    
    while True:
        #frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):                
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break
        success, img = cap.read()
        if success:
            #img, left_eye_state, right_eye_state, detected_face, store = detect.detect(img, visualise=True)
            img, left_eye_state, right_eye_state, detected_face = detect.detect(img, visualise=True)

            if detected_face is not None :
                num_landmarks = len(detected_face.landmark)
                max_landmark_index = max(max(detect.triangle_points), max(detect.left_eye_id_list), max(detect.right_eye_id_list))
                if num_landmarks >= max_landmark_index + 1:
                     angle = detect.fraction(detected_face, detect.triangle_points)
                     mouth_ratio = detect.mouth(detected_face, detect.mouth_list)
                     mouth_state = detect.detect_mouth_state(mouth_ratio, detect.ywan)
                     mouth_state_message = "ywan" if mouth_state == detect.ywan else "active"
                     blink_count = detect.detect_blinks(detect.previous_eye_state,detect.left_eye_state, detect.right_eye_state)
                     
                     cv2.putText(img, mouth_state_message, (10,80 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 0, 20), 2)
                     print("Left_eye:", left_eye_state, "Right_eye:", right_eye_state, "Angle:", angle, "mouth_state", mouth_state  )
                  
                     
                    
            else:
                print("No face detected or no landmarks found.")
        
            #print(detect.divided_ratio)  
                
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        else:
            warnings.warn("No image found !!")  
    print(blink_count)    
    cap.release()
    cv2.destroyAllWindows()

    cv2.destroyAllWindows()



