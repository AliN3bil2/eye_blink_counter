from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import warnings 


# need to implement the right eye detcetion
class MediapipeDetector():
    def __init__(self):
        self.detector = FaceMeshDetector(maxFaces=1)
        self.open = 0 
        self.closed = 1
        self.left_eye_state = self.open # start as open
        self.right_eye_state = self.open # start as open
        self.left_eye_id_list=[22,23,24,26,110,157,158,159,160,161,130,243]
        self.left_eye_normalizing_lst = [159, 23, 243, 130] # eye points: [top, bottom, right left]
    
    def detect_V2H_per_eye(self, detected_face, normalizing_points_idx_lst):        
        l_top = detected_face[normalizing_points_idx_lst[0]]
        l_bottom = detected_face[normalizing_points_idx_lst[1]]
        V_distance = self.detector.findDistance(l_top,l_bottom)                                                              #[16,17,18,19,20] open 16, close 20                                                                                                                # cv2.line(img,l_top,l_bottom,(0,0,0)) 
        l_right = detected_face[normalizing_points_idx_lst[2]]
        l_left = detected_face[normalizing_points_idx_lst[3]]
        H_distance = self.detector.findDistance(l_left,l_right)        
        V2H_ratio = (V_distance[0]/H_distance[0])*100
        return V2H_ratio
    
    def detect_eye_state(self, V2H_ratio, previous_state, thresh_from_open_to_closed = 20, 
                         thresh_from_closed_to_open = 25):
        current_state = self.open # init as open
        if previous_state == self.open: # open
            if V2H_ratio <= thresh_from_open_to_closed:
                current_state = self.closed # turn eye to close
            else:
                current_state = self.open # eye is still open
        else: #if previous_state == self.closed
            if V2H_ratio > thresh_from_closed_to_open:
                current_state = self.open # turn eye to open
            else:
                current_state = self.closed # eye is still closed
        
        return current_state

    def visualise(self, img, detected_face):
        for i in self.left_eye_id_list:               #build circle on key points [22,23,24]
            cv2.circle(img, detected_face[i], 5, (255,255,0), cv2.FILLED)
        return img

    
    def detect(self, img, visualise = False):
        _, faces = self.detector.findFaceMesh(img,draw=False)
        if faces:
            detected_face=faces[0]                  #maximum faces is 1
            if visualise:
                img = self.visualise(img, detected_face)
            left_eye_V2H = self.detect_V2H_per_eye(detected_face, self.left_eye_normalizing_lst)
            self.left_eye_state = self.detect_eye_state(left_eye_V2H, self.left_eye_state, thresh_from_open_to_closed = 20, 
                         thresh_from_closed_to_open = 25)
        else:
            warnings.warn('No face detected !!') 
        
        return img, self.left_eye_state, self.right_eye_state

