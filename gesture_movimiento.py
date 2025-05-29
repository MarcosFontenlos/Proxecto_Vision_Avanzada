#!/usr/bin/env python3
import cv2
import mediapipe as mp
import sys
import rospy
import math
import time
from geometry_msgs.msg import Twist
from ultralytics import YOLO
import numpy as np
import argparse

class ModeloYolo:
    def __init__(self, ruta_modelo='yolov8n.pt'):
        self.model = YOLO(ruta_modelo)

    def detectar_botellas(self, frame, dibujar=False):
        results = self.model.predict(source=frame, imgsz=640, conf=0.5, verbose=False)
        botellas_detectadas = 0
        botella_centro = None

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 39:
                
                    if dibujar:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if botella_centro is None:
                        x1, y1, x2, y2 = box.xyxy[0]
                        botella_centro = ((x1 + x2) / 2, (y1 + y2) / 2)
                    botellas_detectadas += 1

        return botellas_detectadas > 0, botella_centro

class KalmanFilter1D:
    def __init__(self, process_variance=1e-4, measurement_variance=1e-2):
        self.x = np.array([[0.], [0.]])
        self.P = np.eye(2)
        self.F = np.array([[1., 1.], [0., 1.]])
        self.H = np.array([[1., 0.]])
        self.Q = process_variance * np.eye(2)
        self.R = np.array([[measurement_variance]])

    def update(self, measurement):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = np.array([[measurement]]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x[0, 0]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', action='store_true', help='Usar cámara IP en lugar de cámara local')
    return parser.parse_args()

args = parse_args()


class GestureDetector:
    def __init__(self, cam_index=0):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        if args.ip:
            url_camara_ip = "http://192.168.1.10/axis-cgi/mjpg/video.cgi"
            self.cap = cv2.VideoCapture(url_camara_ip)
        else:
            self.cap = cv2.VideoCapture(cam_index)

        self.pub = rospy.Publisher('/robulab10/cmd_vel', Twist, queue_size=10)
        rospy.init_node('gesture_controller', anonymous=True)

        self.modelo_botellas = ModeloYolo()
        self.modelo_personas = YOLO('yolov8n.pt')

        # Estados activos
        self.estado_actual = "RECONOCIMIENTO"  # Estado neutro de espera/reconocimiento

        # Contadores y métricas para cambios de estado con precisión y frames consecutivos
        self.contador_frames_reconocimiento = 0
        self.contador_frames_parar = 0
        self.frames_reconocimiento_necesarios = 90
        self.frames_parar_necesarios = 20
        self.precision_reconocimiento = 0.8
        self.precision_parar = 0.5

        # Conteos para gesto cerrado (personas), 1-2 dedos (botellas), 5 dedos (parar)
        self.contador_cerrado = 0
        self.contador_botella = 0
        self.contador_parar = 0

        self.contador_perdida_persona = 0
        self.contador_perdida_botella = 0
        self.mejor_botella = None
        self.mejor_distancia = float('inf')
        self.inicio_giro_botella = None

        self.ticks_perdida_max = 5 * 30  # 5 segundos si fps ~30 para considerar perdida

        # Filtros Kalman para suavizar giros en seguimiento
        self.kalman_angular = KalmanFilter1D()
        
        # Umbrales de área para definir "cercanía"
        self.umbral_area_persona_cerca = 30000  # ajusta según tamaño bbox persona
        self.umbral_area_botella_cerca = 8000  # ajusta según tamaño bbox botella

        # Variables para control de búsqueda 360 grados botellas
        self.busqueda_360_en_curso = False
        self.busqueda_360_inicio = 0
        self.busqueda_360_duracion = 10  # segundos para dar la vuelta completa

        self.botella_detectada_durante_busqueda = False  # <-- Añade esta línea

        self.pid_kp = 0.00009 # Ganancia proporcional
        self.pid_ki = 0.000005  # Ganancia integral
        self.pid_kd = 0.00005  # Ganancia derivativa

        self.pid_error_prev = 0
        self.pid_integral = 0

        # Variables para suavizado de comandos
        self.lin_filtrado = 0.0
        self.ang_filtrado = 0.0
        self.alpha_suavizado = 0.3  # 0.0 = muy suave, 1.0 = sin suavizado

        if not self.cap.isOpened():
            print("[ERROR] No se pudo abrir la cámara")
            sys.exit(1)

    def analizar_gesto(self, hand_landmarks):
        # Detectar dedos levantados
        dedos = [
            hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x,
            hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,
            hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y,
            hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y,
            hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y,
        ]
        dedos_levantados = dedos.count(True)
        return dedos_levantados

    def mano_centrada(self, hand_landmarks, frame_width):
        # Definir un margen para considerar centrada la mano en la imagen
        mano_x = hand_landmarks.landmark[9].x * frame_width  # punto medio de la palma
        margen_centro = frame_width * 0.2  # 20% margen
        centro = frame_width / 2
        return abs(mano_x - centro) < margen_centro

    def run(self):
        last_persona_detectada_time = None
        last_botella_detectada_time = None

        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_hands = self.hands.process(frame_rgb)

            gesto_actual = "NINGUNO"
            mano_centrada_flag = False

            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    dedos_levantados = self.analizar_gesto(hand_landmarks)
                    mano_centrada_flag = self.mano_centrada(hand_landmarks, frame.shape[1])
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Clasificar gesto
                    if dedos_levantados == 5:
                        gesto_actual = "PARAR"
                    elif dedos_levantados == 0:
                        gesto_actual = "CERRADO"
                    elif dedos_levantados == 1 or dedos_levantados == 2:
                        gesto_actual = "BOTELLA"
                    else:
                        gesto_actual = "NINGUNO"
                    break
            else:
                gesto_actual = "NINGUNO"

            # Actualizar contadores segun gesto
            if gesto_actual == "PARAR" and mano_centrada_flag:
                self.contador_parar += 1
            else:
                self.contador_parar = max(0, self.contador_parar - 1)

            if gesto_actual == "CERRADO" and mano_centrada_flag:
                self.contador_cerrado += 1
            else:
                self.contador_cerrado = max(0, self.contador_cerrado - 1)

            if gesto_actual == "BOTELLA" and mano_centrada_flag:
                self.contador_botella += 1
            else:
                self.contador_botella = max(0, self.contador_botella - 1)

            # Cambio de estados solo si la mano está centrada
            if mano_centrada_flag:
                # Prioridad PARAR
                if self.contador_parar >= self.frames_parar_necesarios * self.precision_parar:
                    if self.estado_actual != "PARAR":
                        print("[INFO] Estado cambiado a PARAR por gesto")
                    self.estado_actual = "PARAR"
                    self.contador_perdida_persona = 0
                    self.contador_perdida_botella = 0

                # Si estamos en PARAR y no hay gesto parar suficiente, volver a RECONOCIMIENTO
                elif self.estado_actual == "PARAR" and self.contador_parar < self.frames_parar_necesarios * self.precision_parar:
                    self.estado_actual = "RECONOCIMIENTO"
                    print("[INFO] Estado PARAR desactivado")

                # Cambiar a seguimiento persona si gesto cerrado está suficiente tiempo
                elif self.contador_cerrado >= self.frames_reconocimiento_necesarios * self.precision_reconocimiento:
                    if self.estado_actual != "SEGUIMIENTO_PERSONA":
                        print("[INFO] Estado cambiado a SEGUIMIENTO_PERSONA")
                    self.estado_actual = "SEGUIMIENTO_PERSONA"
                    self.contador_perdida_persona = 0
                    self.contador_perdida_botella = 0
                    self.busqueda_360_en_curso = False

                # Cambiar a seguimiento botella si gesto botella está suficiente tiempo
                elif self.contador_botella >= self.frames_reconocimiento_necesarios * self.precision_reconocimiento:
                    if self.estado_actual != "SEGUIMIENTO_BOTELLA":
                        print("[INFO] Estado cambiado a SEGUIMIENTO_BOTELLA")
                        self.estado_actual = "SEGUIMIENTO_BOTELLA"
                        self.busqueda_360_en_curso = True
                        self.inicio_giro_botella = time.time()
                        self.mejor_botella = None
                        self.mejor_distancia = float('inf')


                # Si no hay ningún gesto claro, volver a reconocimiento
                elif self.estado_actual not in ["RECONOCIMIENTO", "PARAR"]:
                    self.estado_actual = "RECONOCIMIENTO"
                    print("[INFO] Estado cambiado a RECONOCIMIENTO")

            # Comportamiento por estado
            twist = Twist()
            if self.estado_actual == "PARAR":
                twist.linear.x = 0
                twist.angular.z = 0
                self.pub.publish(twist)
                cv2.putText(frame, "Estado: PARAR", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            elif self.estado_actual == "RECONOCIMIENTO":
                twist.linear.x = 0
                twist.angular.z = 0
                self.pub.publish(twist)
                cv2.putText(frame, "Estado: RECONOCIMIENTO", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

            elif self.estado_actual == "SEGUIMIENTO_PERSONA":
                # Detectar personas en el frame
                results_persona = self.modelo_personas.predict(source=frame, imgsz=640, conf=0.5, verbose=False)
                persona_centro = None
                persona_area = 0
                if results_persona:
                    for r in results_persona:
                        for box in r.boxes:
                            if int(box.cls[0]) == 0:  # Clase 0 = persona
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                area = (x2 - x1) * (y2 - y1)
                                cx = (x1 + x2) / 2
                                cy = (y1 + y2) / 2
                                persona_centro = (cx, cy)
                                persona_area = area
                                # Dibujar bbox
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                break
                        if persona_centro:
                            break

                if persona_centro:
                    centro_imagen = frame.shape[1] / 2
                    margen = 30  # píxeles de tolerancia para considerar "centrado"
                    velocidad_giro = 0.3  # velocidad angular fija, ajusta según tu robot

                    if persona_centro[0] < centro_imagen - margen:
                        ang = velocidad_giro  # gira a la izquierda
                    elif persona_centro[0] > centro_imagen + margen:
                        ang = -velocidad_giro  # gira a la derecha
                    else:
                        ang = 0.0  # no gira

                    # PID solo para avance/retroceso
                    area_objetivo = 80000
                    tolerancia = 10000
                    error = area_objetivo - persona_area

                    if abs(error) > tolerancia:
                        self.pid_integral += error
                        derivada = error - self.pid_error_prev
                        lin = (self.pid_kp * error) + (self.pid_ki * self.pid_integral) + (self.pid_kd * derivada)
                        lin = max(min(lin, 0.4), -0.2)
                    else:
                        lin = 0.0
                        self.pid_integral = 0

                    self.pid_error_prev = error

                    # Suavizado solo para avance
                    self.lin_filtrado = self.alpha_suavizado * lin + (1 - self.alpha_suavizado) * self.lin_filtrado

                    twist.linear.x = self.lin_filtrado
                    twist.angular.z = ang  # control directo, sin suavizado
                    self.pub.publish(twist)


                else:
                    # Persona no detectada
                    self.contador_perdida_persona += 1
                    if self.contador_perdida_persona > self.ticks_perdida_max:
                        print("[INFO] Persona perdida, volviendo a RECONOCIMIENTO")
                        self.estado_actual = "RECONOCIMIENTO"
                        twist.linear.x = 0
                        twist.angular.z = 0
                        self.pub.publish(twist)
                        self.contador_perdida_persona = 0
                    else:
                        # Mantener último comando o girar despacio para buscar persona
                        twist.linear.x = 0
                        twist.angular.z = 0.1
                        self.pub.publish(twist)
                    cv2.putText(frame, "Estado: SEGUIMIENTO PERSONA (PERDIDO)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,100,0), 2)

            elif self.estado_actual == "SEGUIMIENTO_BOTELLA":
                # Iniciar búsqueda de 360 grados si no está en curso
                if not self.busqueda_360_en_curso:
                    self.busqueda_360_en_curso = True
                    self.busqueda_360_inicio = time.time()
                    self.botella_detectada_durante_busqueda = False  # <- Nuevo flag
                    print("[INFO] Iniciando búsqueda 360 grados de botellas")

                # Detectar botellas
                botellas_encontradas, botella_centro = self.modelo_botellas.detectar_botellas(frame, dibujar=True)

                if botellas_encontradas:
                    self.botella_detectada_durante_busqueda = True
                    self.ultima_botella_centro = botella_centro  # Guardar posición da botella máis recente

                # Comprobar se completou a busca de 360°
                tempo_giro = time.time() - self.busqueda_360_inicio
                if tempo_giro < self.busqueda_360_duracion:
                    # Seguir xirando
                    twist.linear.x = 0
                    twist.angular.z = 2 * math.pi / self.busqueda_360_duracion  # velocidade para 360° en duración
                    self.pub.publish(twist)
                    cv2.putText(frame, "Buscando botella (360°)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                else:
                    # Finalizou a busca de 360°
                    self.busqueda_360_en_curso = False
                    twist.linear.x = 0
                    twist.angular.z = 0
                    self.pub.publish(twist)

                    if self.botella_detectada_durante_busqueda:
                        print("[INFO] Botella detectada durante a busca, iniciando aproximación")
                        # Aproximarse á botella detectada
                        error_x = (frame.shape[1]/2) - self.ultima_botella_centro[0]
                        ang = 0.005 * error_x
                        ang = self.kalman_angular.update(ang)

                        twist.linear.x = 0.1  # Ou 0.0 se queres só xirarse
                        twist.angular.z = ang
                        self.pub.publish(twist)
                        cv2.putText(frame, "Estado: APROXIMACIÓN BOTELLA", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # Se queres cambiar de estado, podes poñer: self.estado_actual = "APROXIMACION"
                    else:
                        print("[INFO] Non se detectou botella, volvendo a RECONOCIMIENTO")
                        self.estado_actual = "RECONOCIMIENTO"
                        self.contador_perdida_botella = 0
                        self.busqueda_360_en_curso = False
                        cv2.putText(frame, "No se encontró botella", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            else:
                # Estado por defecto: parar robot
                twist.linear.x = 0
                twist.angular.z = 0
                self.pub.publish(twist)
                cv2.putText(frame, "Estado: DESCONOCIDO", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # Mostrar los contadores de gestos en pantalla
            cv2.putText(frame, f"Parar: {self.contador_parar}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.putText(frame, f"Cerrado: {self.contador_cerrado}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Botella: {self.contador_botella}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

            cv2.imshow('Camara', frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    gd = GestureDetector()
    gd.run()
