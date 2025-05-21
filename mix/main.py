import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from cv2 import aruco
from ID1 import ArucoPIDController
from ID2 import ArucoPWMController
from message_filters import Subscriber, ApproximateTimeSynchronizer

class DualArucoMain(Node):
    def __init__(self):
        super().__init__('dual_aruco_main')
        self.bridge = CvBridge()
        self.controller_id1 = ArucoPIDController(self)
        self.controller_id2 = ArucoPWMController(self)
        self.controller_id1.set_target_pixel()

        # 同步兩相機 topic
        self.sub1 = Subscriber(self, Image, '/camera1/image_raw')
        self.sub2 = Subscriber(self, Image, '/camera2/image_raw')
        self.ts = ApproximateTimeSynchronizer([self.sub1, self.sub2], queue_size=5, slop=0.1)
        self.ts.registerCallback(self.image_callback)

        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
        self.aruco_params = aruco.DetectorParameters()

    def image_callback(self, img_msg1, img_msg2):
        # 轉為 OpenCV 格式
        frame1 = self.bridge.imgmsg_to_cv2(img_msg1, desired_encoding='bgr8')
        frame2 = self.bridge.imgmsg_to_cv2(img_msg2, desired_encoding='bgr8')

        # 合併影像（橫向拼接）
        combined = cv2.hconcat([frame1, frame2])  # 或 cv2.vconcat(...)

        # ArUco 偵測
        gray = cv2.cvtColor(combined, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        # 傳給兩個控制器
        self.controller_id1.process(combined, corners, ids)
        self.controller_id2.process(combined, corners, ids)

        cv2.imshow("Combined Aruco Frame", combined)
        cv2.waitKey(1)

    def cleanup(self):
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = DualArucoMain()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("✅ 關閉中止")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
