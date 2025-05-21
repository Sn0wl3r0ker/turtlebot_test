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

        # 新增影像暫存變數
        self.image1 = None
        self.image2 = None

        # 分別訂閱兩個相機
        self.sub1 = self.create_subscription(
            Image, '/camera1/image_raw', self.image1_callback, 10)
        self.sub2 = self.create_subscription(
            Image, '/camera2/image_raw', self.image2_callback, 10)
        # self.sub1 = Subscriber(self, Image, '/camera1/image_raw')
        # self.sub2 = Subscriber(self, Image, '/camera2/image_raw')
        # self.ts = ApproximateTimeSynchronizer([self.sub1, self.sub2], queue_size=5, slop=0.1)
        
        # print(f"[INFO] before ts.registerCallback")
        # self.ts.registerCallback(self.image_callback)
        # print(f"[INFO] after ts.registerCallback")
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
        self.aruco_params = aruco.DetectorParameters()
    def image1_callback(self, img_msg1):
        print("收到 camera1 影像訊息")
        # 將影像訊息轉成 OpenCV 格式並存入 self.image1
        try:
            self.image1 = self.bridge.imgmsg_to_cv2(img_msg1, desired_encoding='bgr8')
        except Exception as e:
            print("❌ camera1 cv_bridge 轉換失敗:", e)
            self.image1 = None
        self.try_show_combined()
    def image2_callback(self, img_msg2):
        print("收到 camera2 影像訊息")
        # 將影像訊息轉成 OpenCV 格式並存入 self.image2
        try:
            self.image2 = self.bridge.imgmsg_to_cv2(img_msg2, desired_encoding='bgr8')
        except Exception as e:
            print("❌ camera2 cv_bridge 轉換失敗:", e)
            self.image2 = None
        self.try_show_combined()
    # def image_callback(self, img_msg1, img_msg2):
    #     # 轉為 OpenCV 格式
    #     print("收到兩個影像訊息")
    #     try:
    #         frame1 = self.bridge.imgmsg_to_cv2(img_msg1, desired_encoding='bgr8')
    #         frame2 = self.bridge.imgmsg_to_cv2(img_msg2, desired_encoding='bgr8')
    #         print("影像1 shape:", frame1.shape, "dtype:", frame1.dtype)
    #         print("影像2 shape:", frame2.shape, "dtype:", frame2.dtype)
    #     except Exception as e:
    #         print("❌ cv_bridge 轉換失敗:", e)
    #         return

    #     # 檢查尺寸是否一致
    #     if frame1.shape != frame2.shape:
    #         print(f"❗ frame1.shape={frame1.shape}, frame2.shape={frame2.shape}，無法合併")
    #         return

    #     try:
    #         combined = cv2.hconcat([frame1, frame2])
    #         print("合併後影像 shape:", combined.shape)
    #     except Exception as e:
    #         print("❌ 合併影像失敗:", e)
    #         return
    def try_show_combined(self):
        if self.image1 is not None and self.image2 is not None:
            if self.image1.shape == self.image2.shape:
                combined = cv2.hconcat([self.image1, self.image2])
                print("合併後影像 shape:", combined.shape)
                cv2.imshow("Combined Aruco Frame", combined)
                cv2.waitKey(1)
            else:
                print(f"❗ frame1.shape={self.image1.shape}, frame2.shape={self.image2.shape}，無法合併")
        # 後續處理
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
