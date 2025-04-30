import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')  # O un modelo personalizado con detección de manos
        self.cap = cv2.VideoCapture(0)
        self.hand_roi_pub = self.create_publisher(Image, '/hand_rois', 10)
        self.timer = self.create_timer(0.1, self.detect_callback)

    def detect_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('No se pudo leer la cámara')
            return

        results = self.model(frame)[0]
        hand_boxes = []

        # Detección de manos (asumiendo que 'hand' es una clase en tu modelo)
        for box in results.boxes:
            cls = int(box.cls[0])
            label = self.model.names[cls]
            if label == 'hand':  # Ajusta según las clases de tu modelo
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                hand_boxes.append((x1, y1, x2, y2))

        # Publicar cada ROI de mano
        for (x1, y1, x2, y2) in hand_boxes:
            hand_roi = frame[y1:y2, x1:x2]  # Recortar la región de la mano
            if hand_roi.size > 0:  # Asegurar que la ROI no está vacía
                roi_msg = self.bridge.cv2_to_imgmsg(hand_roi, encoding='bgr8')
                self.hand_roi_pub.publish(roi_msg)

        # Visualización (opcional)
        cv2.imshow('YOLO Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()