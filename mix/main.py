import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from cv2 import aruco
from ID1 import ArucoPIDController
from ID2 import ArucoPWMController
import numpy as np

class DualArucoMain(Node):
    def __init__(self):
        super().__init__('dual_aruco_main')
        self.bridge = CvBridge()
        self.controller_id1 = ArucoPIDController(self)
        self.controller_id2 = ArucoPWMController(self)
        self.controller_id1.set_target_pixel()
        self.image1 = None
        self.image2 = None
        self.sub1 = self.create_subscription(
            Image, '/camera1/image_raw', self.image1_callback, 10)
        self.sub2 = self.create_subscription(
            Image, '/camera2/image_raw', self.image2_callback, 10)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
        self.aruco_params = aruco.DetectorParameters()

    def image1_callback(self, img_msg1):
        try:
            self.image1 = self.bridge.imgmsg_to_cv2(img_msg1, desired_encoding='bgr8')
        except Exception as e:
            print("❌ camera1 cv_bridge 轉換失敗:", e)
            self.image1 = None
        self.try_show_combined()

    def image2_callback(self, img_msg2):
        try:
            self.image2 = self.bridge.imgmsg_to_cv2(img_msg2, desired_encoding='bgr8')
        except Exception as e:
            print("❌ camera2 cv_bridge 轉換失敗:", e)
            self.image2 = None
        self.try_show_combined()

    def try_show_combined(self):
        if self.image1 is not None and self.image2 is not None:
            if self.image1.shape == self.image2.shape:
                h, w, c = self.image1.shape
                combined = cv2.hconcat([self.image1.copy(), self.image2.copy()])
                # --- 處理左邊相機（ID1）---
                gray1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
                corners1, ids1, _ = aruco.detectMarkers(gray1, self.aruco_dict, parameters=self.aruco_params)
                if ids1 is not None:
                    for i, marker_id in enumerate(ids1.flatten()):
                        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                            corners1[i], self.controller_id1.marker_length, self.controller_id1.camera_matrix, self.controller_id1.dist_coeffs)
                        tvec = tvec[0][0]
                        proj, _ = cv2.projectPoints(
                            np.array([[tvec]], dtype=np.float32), rvec, tvec,
                            self.controller_id1.camera_matrix, self.controller_id1.dist_coeffs)
                        if (
                            proj is not None and proj.shape == (1, 1, 2) and
                            np.isfinite(proj[0][0][0]) and np.isfinite(proj[0][0][1])
                        ):
                            px, py = int(proj[0][0][0]), int(proj[0][0][1])
                            if 0 <= px < w and 0 <= py < h:
                                cv2.circle(combined, (px, py), 8, (255, 0, 0), 2)
                                cv2.putText(combined, f"ID1", (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # --- 處理右邊相機（ID2）---
                gray2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
                corners2, ids2, _ = aruco.detectMarkers(gray2, self.aruco_dict, parameters=self.aruco_params)
                if ids2 is not None:
                    for i, marker_id in enumerate(ids2.flatten()):
                        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                            corners2[i], self.controller_id2.marker_length, self.controller_id2.camera_matrix, self.controller_id2.dist_coeffs)
                        tvec = tvec[0][0]
                        proj, _ = cv2.projectPoints(
                            np.array([[tvec]], dtype=np.float32), rvec, tvec,
                            self.controller_id2.camera_matrix, self.controller_id2.dist_coeffs)
                        if (
                            proj is not None and proj.shape == (1, 1, 2) and
                            np.isfinite(proj[0][0][0]) and np.isfinite(proj[0][0][1])
                        ):
                            px, py = int(proj[0][0][0]) + w, int(proj[0][0][1])
                            if 0 <= px < 2*w and 0 <= py < h:
                                cv2.circle(combined, (px, py), 8, (0, 255, 0), 2)
                                cv2.putText(combined, f"ID2", (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # --- 控制邏輯 ---
                # 可根據需求呼叫 self.controller_id1.process(self.image1, ...) 與 self.controller_id2.process(self.image2, ...)
                # 例如：
                # self.controller_id1.process(self.image1, corners1, ids1, offset_x=0)
                # self.controller_id2.process(self.image2, corners2, ids2, offset_x=0)
                cv2.imshow("Aruco Shared Frame", combined)
                cv2.waitKey(1)
            else:
                print(f"❗ frame1.shape={self.image1.shape}, frame2.shape={self.image2.shape}，無法合併")

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
