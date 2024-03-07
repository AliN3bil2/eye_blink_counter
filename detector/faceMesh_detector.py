from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import warnings 
import os

# need to implement the right eye detcetion
class MediapipeDetector():
    def __init__(self):    
        self.detector = FaceMeshDetector(maxFaces=1)
        self.open = 0 
        self.closed = 1
        self.left_eye_state = self.open # start as open
        self.right_eye_state = self.open # start as open
        self.right_eye_id_list=[362, 263, 249, 390, 373, 374, 380, 381, 385, 388, 466, 387, 386, 384, 398, 382]
        self.right_eye_normalizing_lst=[386, 374, 390,362]
        self.left_eye_id_list=[155, 154, 153, 145, 7, 33, 133, 157, 158, 159, 160, 161, 246, 173]
        self.left_eye_normalizing_lst = [159, 145, 161, 133] # eye points: [top, bottom, right left]
    
    def detect_V2H_per_left_eye(self, detected_face, normalizing_points_idx_lst):        
        l_top = detected_face[normalizing_points_idx_lst[0]]
        l_bottom = detected_face[normalizing_points_idx_lst[1]]
        V_distance = self.detector.findDistance(l_top,l_bottom)                                                              #[16,17,18,19,20] open 16, close 20                                                                                                                # cv2.line(img,l_top,l_bottom,(0,0,0)) 
        l_right = detected_face[normalizing_points_idx_lst[2]]
        l_left = detected_face[normalizing_points_idx_lst[3]]
        H_distance = self.detector.findDistance(l_left,l_right)        
        V2H_ratio_left = (V_distance[0]/H_distance[0])*100
        return V2H_ratio_left
    
    def detect_V2H_per_right_eye(self, detected_face, normalizing_points_idx_lst):        
        r_top = detected_face[normalizing_points_idx_lst[0]]
        r_bottom = detected_face[normalizing_points_idx_lst[1]]
        V_distance = self.detector.findDistance(r_top,r_bottom)                                                              #[16,17,18,19,20] open 16, close 20                                                                                                                # cv2.line(img,l_top,l_bottom,(0,0,0)) 
        r_right = detected_face[normalizing_points_idx_lst[2]]
        r_left = detected_face[normalizing_points_idx_lst[3]]
        H_distance = self.detector.findDistance(r_left,r_right)        
        V2H_ratio_right = (V_distance[0]/H_distance[0])*100
        return V2H_ratio_right
   
   #31,28
    def detect_left_eye_state(self, V2H_ratio_left, previous_state, thresh_from_open_to_closed = 20, 
                         thresh_from_closed_to_open = 25):
        current_state = self.open # init as open
        if previous_state == self.open: # open
            if V2H_ratio_left <= thresh_from_open_to_closed:
                current_state = self.closed # turn eye to close
            else:
                current_state = self.open # eye is still open
        else: #if previous_state == self.closed
            if V2H_ratio_left > thresh_from_closed_to_open:
                current_state = self.open # turn eye to open
            else:
                current_state = self.closed # eye is still closed
        
        return current_state
   

    #edited
    def detect_right_eye_state(self, V2H_ratio_right, previous_state, thresh_from_open_to_closed = 20, 
                         thresh_from_closed_to_open = 25):
        current_state = self.open # init as open
        if previous_state == self.open: # open
            if V2H_ratio_right <= thresh_from_open_to_closed:
                current_state = self.closed # turn eye to close
            else:
                current_state = self.open # eye is still open
        else: #if previous_state == self.closed
            if V2H_ratio_right > thresh_from_closed_to_open:
                current_state = self.open # turn eye to open
            else:
                current_state = self.closed # eye is still closed
        
        return current_state

    def visualise(self, img, detected_face):
        for i in self.left_eye_id_list:     # Visualize key points for the left eye
             cv2.circle(img, detected_face[i], 5, (255, 255, 0), cv2.FILLED)
        for j in self.right_eye_id_list:    # Visualize key points for the right eye
             cv2.circle(img, detected_face[j], 5, (255, 255, 0), cv2.FILLED)
        message = "Left Eye Closed" if self.left_eye_state == self.closed else "Left Eye Open"
        cv2.putText(img, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (25, 25, 25), 2)
        return img
   
    
    def detect(self, img, visualise = False):
        _, faces = self.detector.findFaceMesh(img,draw=False)
        if faces:
            detected_face=faces[0]                  #maximum faces is 1
            if visualise:
                img = self.visualise(img, detected_face)
            left_eye_V2H = self.detect_V2H_per_left_eye(detected_face, self.left_eye_normalizing_lst)
            right_eye_V2H = self.detect_V2H_per_right_eye(detected_face, self.right_eye_normalizing_lst)
            self.left_eye_state = self.detect_left_eye_state(left_eye_V2H, self.left_eye_state)
            self.right_eye_state = self.detect_right_eye_state(right_eye_V2H, self.right_eye_state)
            
            
        else:
            warnings.warn('No face detected !!') 
        
        return img, self.left_eye_state, self.right_eye_state


