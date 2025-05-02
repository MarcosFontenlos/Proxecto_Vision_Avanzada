import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import mediapipe as mp

class GestureDetector(Node):
    def __init__(self):
        super().__init__('gesture_node')
        self.publisher_ = self.create_publisher(String, 'gestos_mano', 10)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=2,
                                         min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        self.timer = self.create_timer(0.1, self.detect_callback)
        cv2.namedWindow('Detección Gestos', cv2.WINDOW_NORMAL)
        self.get_logger().info('Nodo de detección de gestos iniciado.')

    def detect_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('No se pudo leer de la cámara')
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        gesture = "ninguno"
        if results.multi_hand_landmarks:
            gesture = "mano detectada"
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        msg = String()
        msg.data = gesture
        self.publisher_.publish(msg)

        cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Detección Gestos', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = GestureDetector()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
