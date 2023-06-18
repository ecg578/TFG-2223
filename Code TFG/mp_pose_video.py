import cv2
import math
import mediapipe as mp

# Inicializar el modelo de Mediapipe Pose y los parámetros del modelo
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Definir la función para calcular el ángulo de la espalda
def calculate_angle(a, b, c):
    angle = math.degrees(math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x))
    if angle < 0:
        angle += 360
    return angle

# Leer el video en el que se está haciendo el ejercicio de peso muerto
cap = cv2.VideoCapture('PM3.mp4')

# Iniciar un bucle para leer cada fotograma del video y detectar la pose del cuerpo humano
while True:
    ret, frame = cap.read()
    if ret == False:
        break
    
    # Convertir la imagen a escala de grises y realizar la detección de la pose
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = pose.process(frame_gray)

    # Obtener la pose de las articulaciones y evaluar si el ejercicio de peso muerto está bien hecho
    if results.pose_landmarks:
        # Acceder a las coordenadas de las articulaciones
        landmarks = results.pose_landmarks.landmark

        # Verificar que se hayan detectado todas las articulaciones
        if len(landmarks) >= 27:

            # Calcular los ángulos de la espalda y de las piernas
            angle_back = calculate_angle(landmarks[11], landmarks[12], landmarks[24])
            angle_legs = calculate_angle(landmarks[23], landmarks[24], landmarks[26])

            # Realizar la evaluación de la pose y mostrar el resultado en pantalla
            # Ejemplo: si el ángulo de la espalda está dentro de un rango específico, se considera que el ejercicio está bien hecho
            if angle_back > 70 and angle_back < 110 and angle_legs > 170 and angle_legs < 190:
                cv2.putText(frame, 'Ejercicio de peso muerto bien hecho', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar el resultado en pantalla
    cv2.imshow('Pose Estimation', frame)

    # Esperar a que se presione la tecla 'q' para detener la ejecución del programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos utilizados
cap.release()
cv2.destroyAllWindows()
