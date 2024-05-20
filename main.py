
import cv2
import numpy as np
import warnings

if __name__ == "__main__":
    detect = MediapipeDetector()
    cap = cv2.VideoCapture(r"C:\Users\computec\Downloads\WhatsApp Video 2024-05-01 at 3.21.10 PM.mp4")
    
    frame_count = 2997
    save_every_nth_frame = 4  # Change this value as needed

    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):          #frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))       
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break
        success, img = cap.read()
        if success:
            img, left_eye_state, right_eye_state, detected_face = detect.detect(img, visualise=True)
            if detected_face is not None :
                num_landmarks = len(detected_face.landmark)
                max_landmark_index = max(max(detect.triangle_points), max(detect.left_eye_id_list), max(detect.right_eye_id_list))
                if num_landmarks >= max_landmark_index + 1:
                     angle = detect.fraction(detected_face, detect.triangle_points)

                     mouth_ratio = detect.mouth(detected_face, detect.mouth_list)
                     mouth_state = detect.detect_mouth_state(mouth_ratio, detect.ywan)
                     mouth_state_message = "ywan" if mouth_state == detect.ywan else "active"
                     #count  blinks
                     blink_count = detect.detect_blinks(detect.previous_eye_state,detect.left_eye_state, detect.right_eye_state)

                     right_eye_message = "rightClosed" if detect.right_eye_state == detect.closed else "rightOpen"
                     left_eye_message = "leftClosed" if detect.left_eye_state == detect.closed else "leftOpen"
                     print("Left_eye:", left_eye_state, "Right_eye:", right_eye_state, "Angle:", angle, "mouth_state", mouth_state,frame_count )
                     #left eye 
                     cv2.putText(img, left_eye_message, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 0, 0), 2)  
                     #right eye 
                     cv2.putText(img, right_eye_message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 0, 0), 2)
                     #mouth
                     cv2.putText(img, mouth_state_message, (10,80 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 0, 20), 2)
                     
                '''
                if frame_count % save_every_nth_frame == 0:
                    frame_count += 1
                    cv2.imwrite(f'G:\data_for_drawssy\sample\\frame_{frame_count}.jpg', img)  # Save the frame
                else:
                    frame_count += 1   
                '''
            else:
                print("No face detected or no landmarks found.")
           
            #print(detect.divided_ratio)   
            cv2.imshow('img', img)
            if cv2.waitKey(0) & 0xFF == ord('s'): 
                break
        else:
            warnings.warn("No image found !!")  
    
    #print(detect.divided_ratio) 
    cap.release()
    cv2.destroyAllWindows()
    
# 1 close     0 open 
