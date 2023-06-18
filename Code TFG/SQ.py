import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter import filedialog

from scipy.interpolate import interp1d
from matplotlib import pyplot

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

plt.style.use('ggplot')


# Method to calculate the angle
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

# Method to make a smooth curve
def smooth_curve(x, y, window_size):

    if len(x) != len(y):
        raise ValueError('The x and y arrays must have the same length')

    if window_size < 2:
        return x, y

    w = int(window_size/2)

    smoothed_x = x[w:-w]
    smoothed_y = np.zeros_like(smoothed_x)

    for i in range(w, len(x)-w):
        smoothed_y[i-w] = np.mean(y[i-w:i+w+1])

    return smoothed_x, smoothed_y


# Crear la ventana de la interfaz gráfica de usuario
root = tk.Tk()
root.withdraw()

filetypes = (
    ('Archivos MP4', 'SQ*.mp4'),
    ('Todos los archivos', '*.*')
)

# Abrir el cuadro de diálogo de selección de archivo y obtener la ruta del archivo seleccionado
file_path = filedialog.askopenfilename(filetypes=filetypes)
# VIDEO FEED
pixels_per_cm = 25
cap = cv2.VideoCapture(file_path)  # o el nombre de tu archivo de video

# Get Video duration
fps = int(cap.get(cv2.CAP_PROP_FPS)) # FPS 
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Frame Count
duration = frame_count/fps # Duration
duration = round(duration,2)

#Time list 
time = np.arange(0,duration,1/fps)

# Curl counter variables
counter = 0 
min_ang = 0
max_distance = 0
max_ang = 0
min_ang_hip = 0
max_ang_hip = 0
stage = None
spot = None
control = None

# PutText
scale = 0.7

#Bar path
x = []
y = []

#array angles
angle_knee_array=[]

# Distance between foots
distance_max = []

#Minimun Angles
angle_min = []
angle_min_hip = []
angle_min_neck=[]

# Setup Output Video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
new_width = 550
new_height = int(new_width * height / width)

out = cv2.VideoWriter('output_SQ.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (new_width, new_height))

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is not None:
            frame_ = cv2.resize(frame, (new_width, new_height))
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
        landmarks = results.pose_landmarks.landmark
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get visibility Right Body
            kneeRV = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility
            AnkleRV = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility
            HipRV = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
            ShoulderRV = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
            earRV = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].visibility


            # Get visibility Left Body
            kneeLV = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
            AnkleLV = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility
            HipLV = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility
            ShoulderLV = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
            earLV = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].visibility

            # Get foot position 
            left_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            right_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

            visibilityFoot = (right_foot.visibility+left_foot.visibility) / 2 

            # Verificar si la rodilla derecha está dentro del área visible de la imagen
            if kneeRV > 0.6 and AnkleRV > 0.6 and HipRV > 0.6 and ShoulderRV > 0.6 and earRV > 0.6 and visibilityFoot > 0.6:
                text = "Key points for squat detected"

                # If Key points detected are right, I work with RIGHT BODY
                # Get coordinates right shoulder to get bar position
                x.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
                y.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)

                shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
                ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
                ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y

            elif kneeLV > 0.6 and AnkleLV > 0.6 and HipLV > 0.6 and ShoulderLV > 0.6 and earLV > 0.6 and visibilityFoot > 0.6:
                text = "Key points for squat detected"

                # If Key points detected are left, I work with LEFT BODY
                # Get coordinates left shoulder to get bar position
                x.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x)
                y.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y)

                shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
                ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
                hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
                ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y

            else:
                text = "Key points for squat not detected. Try another position" 
        
            # Calculate neck angle
            angle_neck = calculate_angle(hip,shoulder,ear)
            angle_neck = round(angle_neck,2)

            angle_min_neck.append(angle_neck)

            # Calculate hip angle
            angle_hip = calculate_angle(shoulder, hip, knee)
            angle_hip = round(angle_hip,2)

            # Calculate knee angle 
            angle_knee = calculate_angle(hip, knee, ankle)
            angle_knee = round(angle_knee,2)  
            # Add to array          
            angle_knee_array.append(angle_knee)

            angle_min.append(angle_knee)
            angle_min_hip.append(angle_hip)
            
            # Calculate distance foot
            distance = math.sqrt((left_foot.x - right_foot.x)**2 + (left_foot.y - right_foot.y)**2 + (left_foot.z - right_foot.z)**2)
            distance_cm = distance * pixels_per_cm
            distance_cm = round(distance_cm,2)

            distance_max.append(distance_cm)

            # Visualize angle           
            cv2.putText(image, str(angle_knee), 
                           tuple(np.multiply(knee, [450, 950]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            cv2.putText(image, str(angle_hip), 
                           tuple(np.multiply(hip, [450, 900]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            cv2.putText(image, str(angle_neck), 
                        tuple(np.multiply(ear, [450, 900]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            # Curl counter logic
            if angle_knee > 140:
                stage = "Down"
                control = "Keep going down!"
            if angle_knee < 70 and stage =='Down':
                stage="Up"
                counter +=1
                print(counter)
                control = "Good rep, keep it up!"
                    
                min_ang = min(angle_min)
                max_ang = max(angle_min)

                min_ang_hip = min(angle_min_hip)
                max_ang_hip = max(angle_min_hip)

                min_ang_neck = min(angle_min_neck)
                max_ang_neck = max(angle_min_neck)

                max_distance = max(distance_max)

                print("Knee-joint angle : ", min(angle_min), " _ ", max(angle_min))
                print("Hip-joint angle : ", min(angle_min_hip), " _ ", max(angle_min_hip))
                print("Distance foot(cm) : ", max_distance)
                print("Neck align : ", min(angle_min_neck), " _ ", max(angle_min_neck))
                

            if max_distance > 40:
                spot = "Sumo Squat"
            if max_distance < 40:
                spot="Conventional Squat"  

            cv2.rectangle(image, (20,960), (535,860), (0,0,0), -1)
            if angle_neck < 120:
                neck_status = "Neck is not alligned"
                cv2.putText(image, neck_status, (30,890), cv2.FONT_HERSHEY_SIMPLEX, scale, (50, 50, 255), 2)
            else:
                neck_status = "Neck is alligned"
                cv2.putText(image, neck_status, (30,890), cv2.FONT_HERSHEY_SIMPLEX, scale, (127, 255, 0), 2)

        except:
            pass

        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (20,20), (500,280), (0,0,0), -1)
        cv2.rectangle(image, (20,960), (535,900), (0,0,0), -1)
        
        # Rep data
        cv2.putText(image, "Repetition : " + str(counter), 
                    (30,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2, cv2.LINE_AA)
        
        #Knee angle:
        cv2.putText(image, "Knee-joint angle : " + str(min_ang), 
                    (30,100), 
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2, cv2.LINE_AA)
        
        #Hip angle:
        cv2.putText(image, "Hip-joint angle : " + str(min_ang_hip), 
                    (30,140), 
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2, cv2.LINE_AA)

        #Distance foot:
        cv2.putText(image, "Distance foot : " + str(max_distance), 
                    (30,180), 
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2, cv2.LINE_AA)

        #Type of Squat:
        cv2.putText(image, "Spot : " + str(spot), 
                    (30,220), 
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2, cv2.LINE_AA)  

        #Good or Bad Squat:
        cv2.putText(image, control, 
                    (30,260), 
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2, cv2.LINE_AA) 
        
        # Key Points SQ
        cv2.putText(image, text, 
                    (30,940), 
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2, cv2.LINE_AA)
        
        # Render detections
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0,0,0), thickness=4, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(203,17,17), thickness=4, circle_radius=4) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)
        out.write(image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

window_size = 10
smoothed_x, smoothed_y = smooth_curve(np.array(x), np.array(y), window_size)

# Bar pathing plot
plt.plot(smoothed_y, smoothed_x,  label="Pathing")
plt.title('Pathing of the bar during the squat')
plt.xlabel('Horizontal position of the bar')
plt.ylabel('Vertical position of the bar')

plt.show()

#Angle Knee pathing plot
angle_knee_array = np.resize(angle_knee_array,time.shape)

# Crear gráfica
smoothed_x, smoothed_y = smooth_curve(time, angle_knee_array, window_size)
plt.plot(smoothed_x, smoothed_y) # Graficar una línea recta en el eje X (opcional)
plt.xlabel('Time (seconds)')
plt.title('Time plot')

# Mostrar gráfica
plt.show()

