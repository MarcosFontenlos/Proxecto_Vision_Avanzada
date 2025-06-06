#!/usr/bin/env python3
import cv2
import mediapipe as mp
import rospy
import time
import numpy as np
from geometry_msgs.msg import Twist
from ultralytics import YOLO
import csv
import os
from datetime import datetime
import math
import argparse

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
    parser.add_argument('--cam-index', type=int, default=0, help='Índice de cámara USB (ej. 0, 1, 2...)')
    parser.add_argument('--output-csv', type=str, default='metricas_rendimiento.csv', help='Ruta del archivo CSV de salida para las métricas')
    return parser.parse_args()


args = parse_args()

class GestureSegmentDetector:
    def __init__(self, cam_index=0):
        # Inicializar MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                       max_num_hands=1,
                                       min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

        # Inicializar YOLOv8 para segmentación y detección
        self.modelo_segmentacion = YOLO('yolov8n-seg.pt')  # Segmentación
        self.modelo_personas = YOLO('yolov8n.pt')          # Detección de personas

        # Inicializar cámara (igual que en gesture_movimiento.py)
        if args.ip:
            url_camara_ip = "http://192.168.1.10/axis-cgi/mjpg/video.cgi"
            self.cap = cv2.VideoCapture(url_camara_ip)
        else:
            self.cap = cv2.VideoCapture(cam_index)

        if not self.cap.isOpened():
            print("[ERROR] No se pudo abrir la cámara")
            exit()

        # Inicializar ROS (opcional)
        try:
            rospy.init_node('gesture_segment_controller', anonymous=True)
            self.pub = rospy.Publisher('/robulab10/cmd_vel', Twist, queue_size=10)
        except:
            print("[INFO] ROS no inicializado. Modo sin control de robot.")

        # Estados y contadores
        self.estado_actual = "RECONOCIMIENTO"
        self.gesto_actual = "NINGUNO"
        self.mano_centrada_flag = False
        self.contador_parar = 0
        self.contador_cerrado = 0
        self.contador_objeto = 0
        self.frames_necesarios = 15
        self.umbral_confianza = 0.7

        # Variables para búsqueda 360°
        self.busqueda_360_en_curso = False
        self.busqueda_360_inicio = 0
        self.busqueda_360_duracion = 10  # segundos para dar la vuelta completa
        self.ultima_botella_centro = None
        self.botella_detectada_durante_busqueda = False
        self.contador_perdida_botella = 0
        self.ticks_perdida_max = 30  # Mismo valor que para personas

        # Variables para seguimiento de persona
        self.contador_perdida_persona = 0
        self.ticks_perdida_max = 30
        self.lin_filtrado = 0
        self.alpha_suavizado = 0.2

        # Filtros Kalman
        self.kalman_angular = KalmanFilter1D()

        # Control PID
        self.pid_kp = 0.00009 # Ganancia proporcional
        self.pid_ki = 0.000005  # Ganancia integral
        self.pid_kd = 0.00005  # Ganancia derivativa
        self.pid_error_prev = 0
        self.pid_integral = 0

        # Configuración de ventana
        self.window_name = "Gesture + Segmentación"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Métricas
        self.metricas = {
            "perdidas_seguimiento": 0,
            "perdidas_mano": 0,
            "precision_personas": [],
            "precision_mano": [],
            "precision_objeto": [],
            "tiempo_estados": {"PARAR": 0, "SEGUIMIENTO_PERSONA": 0, "SEGUIMIENTO_OBJETO": 0, 'RECONOCIMIENTO': 0 },
            "distancia_objeto": [],
            "colisiones_evitadas": 0
        }
        self.tiempo_inicio_estado = time.time()
        self.archivo_csv = args.output_csv if args.output_csv else "metricas_gesture_movimiento_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"

    def guardar_metricas(self):
        with open(self.archivo_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Métrica", "Valor"])
            for key, value in self.metricas.items():
                if isinstance(value, list):
                    avg = sum(value)/len(value) if len(value) > 0 else 0
                    writer.writerow([key, avg])
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        writer.writerow([f"{key}_{subkey}", subvalue])
                else:
                    writer.writerow([key, value])

    def actualizar_tiempo_estado(self):
        tiempo_actual = time.time()
        duracion = tiempo_actual - self.tiempo_inicio_estado
        self.metricas["tiempo_estados"][self.estado_actual] += duracion
        self.tiempo_inicio_estado = tiempo_actual

    def analizar_gesto(self, hand_landmarks):
        dedos = [
            hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x,  # Pulgar
            hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,   # Índice
            hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y, # Medio
            hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y, # Anular
            hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y, # Meñique
        ]
        return dedos.count(True)

    def mano_centrada(self, hand_landmarks, frame_width):
        mano_x = hand_landmarks.landmark[9].x * frame_width
        margen = frame_width * 0.2
        return abs(mano_x - frame_width/2) < margen

    def detectar_persona(self, frame):
        results = self.modelo_personas(frame, verbose=False)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:  # Clase 0 = persona
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confianza = float(box.conf[0])
                    self.metricas["precision_personas"].append(confianza)
                    area = (x2 - x1) * (y2 - y1)
                    return (x1 + x2)/2, (y1 + y2)/2, area
        return None, None, None

    def detectar_botella(self, frame):
        results = self.modelo_segmentacion(frame, verbose=False)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 39:  # Clase 39 = botella en COCO
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confianza = float(box.conf[0])
                    self.metricas["precision_objeto"].append(confianza)
                    area = (x2 - x1) * (y2 - y1)
                    return (x1 + x2)/2, (y1 + y2)/2, area
        return None, None, None

    def verificar_colision(self, frame):
        results = self.modelo_segmentacion(frame, verbose=False)
        mascara = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for r in results:
            if r.masks is not None:
                for mask in r.masks:
                    mascara = np.maximum(mascara, mask.data.cpu().numpy().astype(np.uint8))
        
        zona_riesgo = mascara[-100:, frame.shape[1]//2 - 50:frame.shape[1]//2 + 50]
        colision = np.sum(zona_riesgo) > 1000
        if colision:
            self.metricas["colisiones_evitadas"] += 1
        return colision

    def run(self):
        try:
            while not rospy.is_shutdown():
                ret, frame = self.cap.read()
                if not ret:
                    continue

                # Procesamiento de gestos
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_hands = self.hands.process(frame_rgb)

                self.gesto_actual = "NINGUNO"
                self.mano_centrada_flag = False

                if results_hands.multi_hand_landmarks:
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        dedos_levantados = self.analizar_gesto(hand_landmarks)
                        self.mano_centrada_flag = self.mano_centrada(hand_landmarks, frame.shape[1])
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        
                        # Registrar precisión de detección de mano
                        if results_hands.multi_handedness:
                            self.metricas["precision_mano"].append(results_hands.multi_handedness[0].classification[0].score)

                        if dedos_levantados == 5:
                            self.gesto_actual = "PARAR"
                        elif dedos_levantados <=1:
                            self.gesto_actual = "CERRADO"
                        elif 1 <= dedos_levantados <= 2:
                            self.gesto_actual = "OBJETO"
                else:
                    self.metricas["perdidas_mano"] += 1

                # Actualizar contadores de gestos
                if self.gesto_actual == "PARAR" and self.mano_centrada_flag:
                    self.contador_parar = min(self.contador_parar + 1, self.frames_necesarios)
                else:
                    self.contador_parar = max(0, self.contador_parar - 1)

                if self.gesto_actual == "CERRADO" and self.mano_centrada_flag:
                    self.contador_cerrado = min(self.contador_cerrado + 1, self.frames_necesarios)
                else:
                    self.contador_cerrado = max(0, self.contador_cerrado - 1)

                if self.gesto_actual == "OBJETO" and self.mano_centrada_flag:
                    self.contador_objeto = min(self.contador_objeto + 1, self.frames_necesarios)
                else:
                    self.contador_objeto = max(0, self.contador_objeto - 1)

                # Cambiar estados
                estado_anterior = self.estado_actual
                
                if self.contador_parar >= self.frames_necesarios * self.umbral_confianza:
                    self.estado_actual = "RECONOCIMIENTO"
                    self.contador_cerrado = 0
                    self.contador_objeto = 0
                    self.contador_perdida_persona = 0
                    self.contador_perdida_botella = 0
                    self.botella_detectada_durante_busqueda = False
                    self.busqueda_360_en_curso = False
                
                elif self.estado_actual == "PARAR":
                    # Permanece en estado PARAR hasta nuevo gesto
                    if self.contador_cerrado >= self.frames_necesarios * self.umbral_confianza:
                        self.estado_actual = "SEGUIMIENTO_PERSONA"
                    elif self.contador_objeto >= self.frames_necesarios * self.umbral_confianza:
                        self.estado_actual = "SEGUIMIENTO_OBJETO"
                        self.busqueda_360_en_curso = True
                        self.busqueda_360_inicio = time.time()
                        self.botella_detectada_durante_busqueda = False
                
                elif self.estado_actual == "RECONOCIMIENTO":
                    if self.contador_cerrado >= self.frames_necesarios * self.umbral_confianza:
                        self.estado_actual = "SEGUIMIENTO_PERSONA"
                    elif self.contador_objeto >= self.frames_necesarios * self.umbral_confianza:
                        self.estado_actual = "SEGUIMIENTO_OBJETO"
                        self.busqueda_360_en_curso = True
                        self.busqueda_360_inicio = time.time()
                        self.botella_detectada_durante_busqueda = False

                if estado_anterior != self.estado_actual:
                    self.actualizar_tiempo_estado()

                # Segmentación con YOLOv8
                resultados_seg = self.modelo_segmentacion(frame)
                frame_segmentado = resultados_seg[0].plot()

                # Lógica de control por estado
                twist = Twist()
                texto_estado = ""
                color = (0, 0, 0)

                if self.estado_actual == "PARAR":
                    twist.linear.x = 0
                    twist.angular.z = 0
                    texto_estado = "PARAR"
                    color = (0, 0, 255)
                    self.estado_actual = "RECONOCIMIENTO"

                elif self.estado_actual == "RECONOCIMIENTO":
                    twist.linear.x = 0
                    twist.angular.z = 0
                    texto_estado = "RECONOCIMIENTO"
                    color = (255, 255, 0)

                elif self.estado_actual == "SEGUIMIENTO_PERSONA":
                    # Detectar personas en el frame (usando YOLOv8n como en gesture_movimiento.py)
                    results_persona = self.modelo_personas(frame, verbose=False)
                    persona_centro = None
                    persona_area = 0
                    
                    for r in results_persona:
                        for box in r.boxes:
                            if int(box.cls[0]) == 0:  # Clase 0 = persona
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                area = (x2 - x1) * (y2 - y1)
                                cx = (x1 + x2) / 2
                                cy = (y1 + y2) / 2
                                persona_centro = (cx, cy)
                                persona_area = area
                                # Dibujar bbox (opcional)
                                cv2.rectangle(frame_segmentado, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                break  # Solo considerar la primera persona detectada
                        if persona_centro:
                            break

                    if persona_centro:
                        centro_imagen = frame.shape[1] / 2
                        margen = 30  # píxeles de tolerancia para considerar "centrado"
                        velocidad_giro = 0.3  # velocidad angular fija (igual que gesture_movimiento.py)

                        if persona_centro[0] < centro_imagen - margen:
                            ang = velocidad_giro  # gira a la izquierda
                        elif persona_centro[0] > centro_imagen + margen:
                            ang = -velocidad_giro  # gira a la derecha
                        else:
                            ang = 0.0  # no gira

                        # Control PID para distancia (igual que gesture_movimiento.py)
                        area_objetivo = 80000
                        tolerancia = 10000
                        error = area_objetivo - persona_area

                        if abs(error) > tolerancia:
                            self.pid_integral += error
                            derivada = error - self.pid_error_prev
                            lin = (self.pid_kp * error) + (self.pid_ki * self.pid_integral) + (self.pid_kd * derivada)
                            lin = max(min(lin, 0.4), -0.2)  # mismos límites
                        else:
                            lin = 0.0
                            self.pid_integral = 0

                        self.pid_error_prev = error

                        # Suavizado (opcional, igual que gesture_movimiento.py)
                        self.lin_filtrado = self.alpha_suavizado * lin + (1 - self.alpha_suavizado) * self.lin_filtrado

                        twist.linear.x = self.lin_filtrado
                        twist.angular.z = ang
                        texto_estado = "SIGUIENDO PERSONA"
                        color = (0, 255, 0)
                    else:
                        # Persona no detectada (comportamiento igual que gesture_movimiento.py)
                        self.contador_perdida_persona += 1
                        if self.contador_perdida_persona > self.ticks_perdida_max:
                            print("[INFO] Persona perdida, volviendo a RECONOCIMIENTO")
                            self.estado_actual = "RECONOCIMIENTO"
                            twist.linear.x = 0
                            twist.angular.z = 0
                            self.contador_perdida_persona = 0
                        else:
                            # Girar despacio para buscar persona
                            twist.linear.x = 0
                            twist.angular.z = 0.1
                        texto_estado = "BUSCANDO PERSONA"
                        self.metricas["perdidas_seguimiento"] += 1  # Mantener métricas
                        color = (0, 255, 0)

                elif self.estado_actual == "SEGUIMIENTO_OBJETO":
                    if self.busqueda_360_en_curso:
                        # [1] Tiempo desde que comenzó la búsqueda 360°
                        tiempo_giro = time.time() - self.busqueda_360_inicio

                        if tiempo_giro < self.busqueda_360_duracion:
                            # [2] Intentar detectar botella en cada frame mientras gira
                            centro_botella, _, area_botella = self.detectar_botella(frame)

                            if centro_botella:
                                # [3] Botella detectada → interrumpir búsqueda y pasar a seguimiento
                                self.botella_detectada_durante_busqueda = True
                                self.ultima_botella_centro = centro_botella
                                self.ultima_area_botella = area_botella
                                self.busqueda_360_en_curso = False
                                self.contador_perdida_botella = 0

                                texto_estado = "BOTELLA DETECTADA (INTERRUMPIR GIRO)"
                                color = (255, 0, 0)
                                continue  # saltar al siguiente bucle para iniciar seguimiento
                            else:
                                # [4] No se detectó botella aún → seguir girando
                                twist.linear.x = 0
                                twist.angular.z = 2 * math.pi / self.busqueda_360_duracion
                                texto_estado = "BUSQUEDA 360°"
                                color = (255, 255, 0)
                        else:
                            # [5] Tiempo agotado → fin de búsqueda
                            self.busqueda_360_en_curso = False
                            if self.botella_detectada_durante_busqueda:
                                texto_estado = "BOTELLA DETECTADA"
                                color = (0, 255, 0)
                            else:
                                self.estado_actual = "RECONOCIMIENTO"
                                texto_estado = "NO SE ENCONTRO BOTELLA"
                                color = (0, 0, 255)

                    else:
                        results_seg = self.modelo_segmentacion(frame, verbose=False)
                        mejor_confianza = 0
                        botella_centro = None
                        botella_area = 0

                        for r in results_seg:
                            for box in r.boxes:
                                clase = int(box.cls[0])
                                if clase == 39:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    confianza = float(box.conf[0])
                                    if confianza > mejor_confianza:
                                        mejor_confianza = confianza
                                        cx = (x1 + x2) / 2
                                        cy = (y1 + y2) / 2
                                        botella_centro = (cx, cy)
                                        botella_area = (x2 - x1) * (y2 - y1)
                                        cv2.rectangle(frame_segmentado, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            if botella_centro:
                                break

                        if botella_centro:
                            centro_imagen = frame.shape[1] / 2
                            margen = 30
                            if botella_centro[0] < centro_imagen - margen:
                                ang = 0.3
                            elif botella_centro[0] > centro_imagen + margen:
                                ang = -0.3
                            else:
                                ang = 0.0

                            area_objetivo = 50000
                            tolerancia = 10000
                            error = area_objetivo - botella_area

                            if abs(error) > tolerancia:
                                self.pid_integral += error
                                derivada = error - self.pid_error_prev
                                lin = (self.pid_kp * error) + (self.pid_ki * self.pid_integral) + (self.pid_kd * derivada)
                                lin = max(min(lin, 0.3), -0.1)
                            else:
                                lin = 0.0
                                self.pid_integral = 0

                            self.pid_error_prev = error
                            self.lin_filtrado = self.alpha_suavizado * lin + (1 - self.alpha_suavizado) * self.lin_filtrado

                            if self.verificar_colision(frame):
                                twist.linear.x = -0.1
                                twist.angular.z = 0.1
                                texto_estado = "OBSTACULO DETECTADO"
                                color = (0, 0, 255)
                            else:
                                twist.linear.x = self.lin_filtrado
                                twist.angular.z = ang
                                texto_estado = f"SIGUIENDO BOTELLA ({mejor_confianza:.2f})"
                                color = (255, 0, 0)

                            self.contador_perdida_botella = 0
                        else:
                            self.contador_perdida_botella += 1
                            if self.contador_perdida_botella > self.ticks_perdida_max:
                                print("[INFO] Botella perdida, volviendo a RECONOCIMIENTO")
                                self.estado_actual = "RECONOCIMIENTO"
                                twist.linear.x = 0
                                twist.angular.z = 0
                                self.contador_perdida_botella = 0
                            else:
                                twist.linear.x = 0
                                twist.angular.z = 0.2
                            texto_estado = "BUSCANDO BOTELLA"
                            color = (255, 165, 0)


                # Publicar comando de movimiento
                try:
                    self.pub.publish(twist)
                except:
                    pass

                # Mostrar información
                cv2.putText(frame_segmentado, f"Estado: {texto_estado}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame_segmentado, f"Gesto: {self.gesto_actual}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame_segmentado, f"Mano: {'SI' if self.mano_centrada_flag else 'NO'}", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                cv2.imshow(self.window_name, frame_segmentado)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        finally:
            self.actualizar_tiempo_estado()
            self.guardar_metricas()
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = GestureSegmentDetector(cam_index=args.cam_index)
    detector.run()
