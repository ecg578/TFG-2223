import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

pose = mp.solutions.pose.Pose(static_image_mode=True,        
                              min_detection_confidence=0.3,
                              model_complexity=1)

mp_drawing = mp.solutions.drawing_utils 

mp_drawing_styles = mp.solutions.drawing_styles

def drawPose(img, results):
 img_copy = img.copy()

 if results.pose_landmarks:

   style=mp_drawing_styles.get_default_pose_landmarks_style()
   mp_drawing.draw_landmarks(
                   image=img_copy,
                   landmark_list=results.pose_landmarks,
                   landmark_drawing_spec=style,                                        
                   connections=mp.solutions.pose.POSE_CONNECTIONS)

   fig = plt.figure(figsize = [10, 10])
   plt.imshow(img_copy[:,:,::-1])
   plt.show()

img1 = cv2.imread('fotoSQ.png')
results1 = pose.process(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
drawPose(img1,results1)

