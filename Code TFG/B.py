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
    ('Archivos MP4', 'B*.mp4'),
    ('Todos los archivos', '*.*')
)

# Abrir el cuadro de diálogo de selección de archivo y obtener la ruta del archivo seleccionado
file_path = filedialog.askopenfilename(filetypes=filetypes)
# VIDEO FEED
pixels_per_cm = 125
#cap = cv2.VideoCapture(file_path)  # o el nombre de tu archivo de video
cap = cv2.VideoCapture(file_path)

# Get Video duration
fps = int(cap.get(cv2.CAP_PROP_FPS)) # FPS 
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Frame Count
duration = frame_count/fps # Duration
duration = round(duration,2)

#Time list 
time = np.arange(0,duration,1/fps)


# PutText
scale = 0.7

# Curl counter variables
counter = 0 
min_ang_elbowL = 0
min_ang_elbowR = 0
max_ang_elbowL = 0
max_ang_elbowR = 0
max_distance = 0
max_ang = 0
stage = None
spot = None
control = None

#Bar path
x = []
y = []

#array angles
angle_elbowR_array = []
angle_elbowL_array = []
average_angle_elbow_array =[]

# Distance between foots
distance_max = []

#Minimun Angles
angle_min_elbowR = []
angle_min_elbowL = []
angle_min_hip = []
angle_min_neck = []

# Setup Output Video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
new_width = 550
new_height = int(new_width * height / width)

out = cv2.VideoWriter('output_B.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (new_width, new_height))

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

            # Get visibility neccesary
            ShoulderRV = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
            ShoulderLV = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility

            shoulderV = (ShoulderRV + ShoulderLV) / 2

            ElbowRV = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility
            ElbowLV = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility

            elbowV = (ElbowRV + ElbowLV) / 2

            WristRV = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility
            WristLV = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility

            wristV = (WristRV + WristLV) / 2

            hipRV = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
            hipLV = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility

            hipV = (hipRV + hipLV) / 2

            kneeRV = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility
            kneeLV = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility

            kneeV = (kneeRV + kneeLV) / 2


            # Verificar si estos puntos están dentro del área visible de la imagen
            if shoulderV > 0.6 and elbowV > 0.6 and wristV > 0.6 and hipV > 0.6 and kneeV > 0.6:
                text = "Key points for Bench Press detected"

                # If Key points are detected, calcule the key points
                shoulderL = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                elbowL = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
                wristL = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
                hipL = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
                kneeL = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y

                shoulderR = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                elbowR = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
                wristR = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
                hipR = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
                kneeR = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y

                # Get coordinates right shoulder to get bar position
                x.append((results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x + results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x) / 2)
                y.append((results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y + results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y) / 2)
            else:
                text = "Key points for Bench Press not detected. Try another position" 

            # Calculate Right elbow angle 
            angle_elbowR = calculate_angle(wristR, elbowR, shoulderR)
            angle_elbowR = round(angle_elbowR,2)  
            # Add to array          
            angle_elbowR_array.append(angle_elbowR)
            angle_min_elbowR.append(angle_elbowR)


            # Calculate Left elbow angle 
            angle_elbowL = calculate_angle(wristL, elbowL, shoulderL)
            angle_elbowL = round(angle_elbowL,2)  
            # Add to array          
            angle_elbowL_array.append(angle_elbowL)
            angle_min_elbowL.append(angle_elbowL)

            
            # The average of both angles
            average_angle_elbow = (angle_elbowR + angle_elbowL) / 2
            average_angle_elbow_array.append(average_angle_elbow)

            # Calculate distance wrist
            distance = math.sqrt((wristL[0] - wristR[0])**2 + (wristL[1] - wristR[1])**2)
            distance_cm = distance * pixels_per_cm
            distance_cm = round(distance_cm,2)

            distance_max.append(distance_cm)

            # Calculate diference between wrists to know if bar is in a correct position
            y_wrist_diff = abs(wristR[1] - wristL[1])

            y_threshold = 0.015


            # Visualize angle           
            cv2.putText(image, str(angle_elbowR), 
                           tuple(np.multiply(elbowR, [450, 950]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            cv2.putText(image, str(angle_elbowL), 
                           tuple(np.multiply(elbowL, [450, 900]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            # Get bar position
            # Curl counter logic
            if elbowL[1] < shoulderL[1] and elbowR[1] < shoulderR[1] :
                stage = "Down"
                control = "Keep going down!"
            if elbowL[1] > shoulderL[1] and elbowR[1] > shoulderR[1] and stage =='Down':
                stage="Up"
                counter +=1
                print(counter)
                control = "Good rep, keep it up!"
                    
                max_ang_elbowL = max(angle_min_elbowL)
                min_ang_elbowL = min(angle_min_elbowL)

                max_ang_elbowR = max(angle_min_elbowR)
                min_ang_elbowR = min(angle_min_elbowL)

                max_distance = max(distance_max)

                print("Right Elbow -joint angle : ", min(angle_min_elbowR), " _ ", max(angle_min_elbowR))
                print("Left Elbow-joint angle : ", min(angle_min_elbowL), " _ ", max(angle_min_elbowL))
                print("Distance wrist(cm) : ", max_distance)


            cv2.rectangle(image, (20,960), (535,860), (0,0,0), -1)
            if distance_cm > 85:
                spot = "Open grip"
                type="Close the grip" 
                cv2.putText(image, type, (30,890), cv2.FONT_HERSHEY_SIMPLEX, scale, (50, 50, 255), 2)  
            elif distance_cm > 55 and distance_cm < 75:
                spot="Conventional grip"  
                type="Focus on pectoral"  
                cv2.putText(image, type, (30,890), cv2.FONT_HERSHEY_SIMPLEX, scale, (127, 255, 0), 2)  
            elif distance_cm < 50:
                spot="Close grip"
                type="Focus on triceps"
                cv2.putText(image, type, (30,890), cv2.FONT_HERSHEY_SIMPLEX, scale, (127, 255, 0), 2)  
            else:
                type = "Mixed grip"
                cv2.putText(image, type, (30,890), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 0), 2)  


            # Evaluar si la barra está recta
            if y_wrist_diff < y_threshold:
                cv2.putText(image, "Straight bar", (250,890), cv2.FONT_HERSHEY_SIMPLEX, scale, (127, 255, 0), 2) 
            else:
                cv2.putText(image, "Bar not straight. STOP", (250,890), cv2.FONT_HERSHEY_SIMPLEX, scale, (50, 50, 255), 2)  

        except AttributeError:
            landmarks = None
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
        cv2.putText(image, "Max Right Elbow-joint angle : " + str(max_ang_elbowR), 
                    (30,100), 
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2, cv2.LINE_AA)
        
        #Hip angle:
        cv2.putText(image, "Max Left Elbow-joint angle : " + str(max_ang_elbowL), 
                    (30,140), 
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2, cv2.LINE_AA)

        #Distance foot:
        cv2.putText(image, "Distance wrist(cm) : " + str(max_distance), 
                    (30,180), 
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2, cv2.LINE_AA)

        #Type of Squat:
        cv2.putText(image, "Spot: " + str(spot), 
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
plt.title('Pathing of the bar during the bench')
plt.xlabel('Horizontal position of the bar')
plt.ylabel('Vertical position of the bar')

plt.show()

#Angle Knee pathing plot
average_angle_elbow_array = np.resize(average_angle_elbow_array,time.shape)

# Crear gráfica
smoothed_x, smoothed_y = smooth_curve(time, average_angle_elbow_array, window_size)
plt.plot(smoothed_x, smoothed_y) # Graficar una línea recta en el eje X (opcional)
plt.xlabel('Time (seconds)')
plt.title('Time plot')

# Mostrar gráfica
plt.show()

