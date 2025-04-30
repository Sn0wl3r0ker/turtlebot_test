import rclpy
from rclpy.node import Node
import cv2
from cv2 import aruco
from ID1 import ArucoPIDController
from ID2 import ArucoPWMController

class DualArucoMain(Node):
    def __init__(self):
        super().__init__('dual_aruco_main')

        # 建立兩個控制器實例（共用 ROS Node）
        self.controller_id1 = ArucoPIDController(self)
        self.controller_id2 = ArucoPWMController(self)

        # 要求使用者輸入 ID1 的目標像素位置
        self.controller_id1.set_target_pixel()

        # 設定攝影機與 ArUco 偵測參數
        self.cap = cv2.VideoCapture(2)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
        self.aruco_params = aruco.DetectorParameters()

        # 每 0.15 秒執行一次控制迴圈
        self.timer = self.create_timer(0.15, self.control_loop)

    def control_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            print("❌ 無法讀取攝影機畫面")
            return

        # 進行 ArUco 偵測
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        # 將 frame, corners, ids 分別傳入兩個控制器處理
        self.controller_id1.process(frame, corners, ids)
        self.controller_id2.process(frame, corners, ids)

        # 顯示畫面（包含目標點與追蹤點標示）
        cv2.imshow("Aruco Shared Frame", frame)
        cv2.waitKey(1)

    def cleanup(self):
        self.cap.release()
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
