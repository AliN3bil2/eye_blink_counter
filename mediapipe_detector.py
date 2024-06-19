import warnings
import cv2
import numpy as np
import mediapipe as mp
import math
import time
import os  
import pandas as pd

class MediapipeDetector:
    def __init__(self):
        self.previous_eye_state = None
        self.start_time_closed = None
        self.blink_count = 0       
        self.angle_degrees = 45                             #store number of blinks 
        self.alpha = math.cos(self.angle_degrees)          
        self.smoothed_threshold = None
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.divided_ratio=[]
        self.angle_degrees = 45
        self.ywan = 0
        self.active = 1
        self.open = 0
        self.closed = 1
        self.left_eye_state = self.open
        self.right_eye_state = self.open
        self.right_eye_id_list=[362, 263, 249, 390, 373, 374, 380, 381, 385, 388, 466, 387, 386, 384, 398, 382]
        self.right_eye_normalizing_lst=[386,374,263,362]   #385, 380, 390,362    
        self.left_eye_id_list=[155, 154, 153, 145, 7, 33, 133, 157, 158, 159, 160, 161, 246, 173]
        self.left_eye_normalizing_lst = [159,145,33,133] # eye points: [top, bottom, right, left] 
        self.triangle_points = [10, 152]                     # Define triangle points here
        self.mouth_list=[13,15,96,325]                       #top,bottom,left,right
        self.detector = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) # use_depth=True
        
        
   
    

    
    #calculate verrical and hor distance for eye and divide them 
    def detect_V2H_per_eye(self, detected_face, normalizing_points_idx_lst):
        top = detected_face.landmark[normalizing_points_idx_lst[0]]
        bottom = detected_face.landmark[normalizing_points_idx_lst[1]]
        right = detected_face.landmark[normalizing_points_idx_lst[2]]
        left = detected_face.landmark[normalizing_points_idx_lst[3]]
        
        V_distance = math.sqrt((bottom.x - top.x)**2 + (bottom.y - top.y)**2 + (bottom.z - top.z)**2)
        H_distance = math.sqrt((right.x - left.x)**2 + (right.y - left.y)**2 + (right.z - left.z)**2)
        V2H_ratio = (V_distance / H_distance) #* 100
        V2H_ratio #/math.cos(self.angle_degrees)
        #self.divided_ratio.append(V2H_ratio)
        return V2H_ratio
        
    
    
    #calculate to vector and divide it to solve scalable distance
    def mouth(self, detected_face, mouth_list):
        landmark_0 = detected_face.landmark[mouth_list[0]]
        landmark_1 = detected_face.landmark[mouth_list[1]]
        landmark_2 = detected_face.landmark[mouth_list[2]]
        landmark_3 = detected_face.landmark[mouth_list[3]]
        # Calculate distances between landmarks
        x1 = math.sqrt((landmark_1.x - landmark_0.x)**2 + (landmark_1.y - landmark_0.y)**2 + (landmark_1.z - landmark_0.z)**2)
        x2 = math.sqrt((landmark_3.x - landmark_2.x)**2 + (landmark_3.y - landmark_2.y)**2 + (landmark_3.z - landmark_2.z)**2)
        d = x1 / x2
        return d
    
    # determine mouth state (ywan,active)
    #previous_state,threshold_from_ywan_2_not = 0.42774961181825455, threshold_from_not_2_ywan = 0.4951901418998729
    def detect_mouth_state(self, d, previous_state,threshold_from_ywan_2_not = 0.51774961181825455, threshold_from_not_2_ywan = 0.5551901418998729):
        current_mouth_state = self.ywan #ywan
        if previous_state == current_mouth_state:
            if d <= threshold_from_ywan_2_not:
                current_mouth_state = self.active
            else:
                current_mouth_state = self.ywan
        else:
            if d >= threshold_from_not_2_ywan:
                current_mouth_state = self.active            
            else:
                current_mouth_state = self.ywan
        return current_mouth_state
    
    
    
    # function to give the angle of the face depend on 3-point 
    def fraction(self, detected_face, triangle_points):
        
        pt_1 = detected_face.landmark[self.triangle_points[0]].x, detected_face.landmark[self.triangle_points[0]].y, detected_face.landmark[self.triangle_points[0]].z
        pt_2 = detected_face.landmark[self.triangle_points[1]].x, detected_face.landmark[self.triangle_points[1]].y, detected_face.landmark[self.triangle_points[1]].z
        pt_3 = (pt_1[0], pt_1[1], pt_1[2] + 4)  # Adding 4 to the z-coordinate of pt_1
        vector1 = np.array(pt_2) - np.array(pt_1)
        vector2 = np.array(pt_3) - np.array(pt_2)
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        cosine_angle = dot_product / (magnitude1 * magnitude2)
        angle_radians = np.arccos(cosine_angle)
        angle_degrees = np.degrees(angle_radians)
        return angle_degrees
        '''
        x=detected_face.landmark[self.triangle_points[2]].z - detected_face.landmark[self.triangle_points[0]].z
        y=detected_face.landmark[self.triangle_points[0]].y - detected_face.landmark[self.triangle_points[1]].y
        z=detected_face.landmark[self.triangle_points[2]].z - detected_face.landmark[self.triangle_points[1]].z
        
        dot_product=np.dot(x,y)
        magnitud1=np.linalg.norm(x)
        magnitud2=np.linalg.norm(y)
        cosine_angle = dot_product / (magnitud1 * magnitud2)

        angle_radians = np.arccos(cosine_angle)
        angle_degrees = np.degrees(angle_radians)
        return angle_degrees

        '''

# to detect eye state (open, close )                                                                                         20 , 21    for closed eye 

    def detect_eye_state(self, V2H_ratio, angle_degrees, previous_state, thresh_from_open_to_closed=0.33890866361416, thresh_from_closed_to_open=0.3550000000000000,
                         angle_threshold=100):
        current_state = self.open

        # Smooth  based on face angle
        if angle_degrees < angle_threshold:
            thresh_from_open_to_closed = 0.3713890866361416
            if self.smoothed_threshold is None:
                self.smoothed_threshold = thresh_from_open_to_closed 
            else:
                self.smoothed_threshold = self.smooth_threshold(self.smoothed_threshold, 0.3713890866361416)
 

        # Determine eye state 
        if previous_state == current_state:
            if V2H_ratio <= self.smoothed_threshold:
                current_state = self.closed
            else:
                current_state = self.open
        else:
            if V2H_ratio > thresh_from_closed_to_open:
                current_state = self.open
            else:
                current_state = self.closed

        return current_state

    def smooth_threshold(self, current_threshold, new_threshold):
        smoothed_threshold =  new_threshold * self.alpha  + (0.9 - self.alpha) * current_threshold
        return smoothed_threshold
      
 
    # to detect number of blinks      normal 25 per 1m
    
    def detect_blinks(self, previous_eye_state, left_eye_state, right_eye_state):
        if self.previous_eye_state == self.open and (left_eye_state == self.closed and right_eye_state == self.closed):
            self.blink_count += 1
            self.start_time_closed = time.time()    # start calc time when eye is closed 
        
        elif self.start_time_closed and (left_eye_state == self.closed and right_eye_state == self.closed):
            time_closed = time.time() - self.start_time_closed
            if time_closed > 5.0:                   # warning when eye closed more than 5s 
                warning = warnings.warn("Warning: Eyes closed for more than 5 seconds!")

        self.previous_eye_state = self.open if left_eye_state == self.open or right_eye_state == self.open else self.closed
        return self.blink_count    
    
    
    #store sample of data 
    def collect_data(self, left_eye_V2H, right_eye_V2H, left_eye_state, right_eye_state, mouth_ratio, current_mouth_state):
        # file path
        directory = "G:/data_for_drawssy"
        file_name = "data2.csv"
        file_path = os.path.join(directory, file_name)

        with open(file_path, "a") as file:
            file.write(f"{left_eye_V2H},{right_eye_V2H},{mouth_ratio},{left_eye_state},{right_eye_state},{current_mouth_state}\n")


    # Resize the image to 640x480
    def resize_image(self, img, width=640, height=480):
        resized_image = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        return resized_image
    
    #visualization
    def visualise(self, img, detected_face):
        #for idx in self.left_eye_normalizing_lst:
        #    cv2.circle(img, (int(detected_face.landmark[idx].x * img.shape[1]), int(detected_face.landmark[idx].y * img.shape[0])), 1, (255, 255, 0), cv2.FILLED)
        #for idx in self.right_eye_id_list:
        #    cv2.circle(img, (int(detected_face.landmark[idx].x * img.shape[1]), int(detected_face.landmark[idx].y * img.shape[0])), 1, (255, 255, 0), cv2.FILLED)
        #for i in range(len(self.triangle_points)):
        #    cv2.line(img, (int(detected_face.landmark[self.triangle_points[i]].x * img.shape[1]), int(detected_face.landmark[self.triangle_points[i]].y * img.shape[0])), (int(detected_face.landmark[self.triangle_points[(i + 1) % len(self.triangle_points)]].x * img.shape[1]), int(detected_face.landmark[self.triangle_points[(i + 1) % len(self.triangle_points)]].y * img.shape[0])), (0, 255, 0), 2)
        #left_eye_message = "leftClosed" if self.left_eye_state == self.closed else "leftOpen"
        #cv2.putText(img, left_eye_message, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 0, 0), 2)
        #right_eye_message = "rightClosed" if self.right_eye_state == self.closed else "rightOpen"
        #cv2.putText(img, right_eye_message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 0, 20), 2)
        return img
    

    def detect(self, img, visualise=False):
        resized_img = self.resize_image(img)
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        results = self.detector.process(img_rgb)
        
        if results.multi_face_landmarks:
            detected_face = results.multi_face_landmarks[0]
            #self.mp_drawing.draw_landmarks(
            #        img, detected_face, self.mp_face_mesh.FACEMESH_CONTOURS,
            #        landmark_drawing_spec=None, connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),)
            if visualise:
                img = self.visualise(img, detected_face)
                left_eye_V2H = self.detect_V2H_per_eye(detected_face, self.left_eye_normalizing_lst)
                right_eye_V2H = self.detect_V2H_per_eye(detected_face, self.right_eye_normalizing_lst)
                #eye state 


                self.left_eye_state = self.detect_eye_state(left_eye_V2H, self.angle_degrees,self.left_eye_state)                  #right eye state
                self.right_eye_state = self.detect_eye_state(right_eye_V2H,self.angle_degrees, self.right_eye_state)               #left eye state 
                blink_count = self.detect_blinks(self.previous_eye_state, self.left_eye_state, self.right_eye_state)

            return img, self.left_eye_state, self.right_eye_state, detected_face #,store    
        else:
            warnings.warn('No face detected !!')
        return img, self.left_eye_state, self.right_eye_state, None





