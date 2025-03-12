import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import math
from cv2 import aruco

class ArucoPIDController(Node):
    def __init__(self):
        super().__init__('aruco_pid_controller')
        
        # 訂閱 /cmd_vel 來發送控制訊號
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 5)
        self.timer = self.create_timer(0.15, self.control_loop)  # 每 0.15 秒執行一次

        # PID 參數
        self.Kp_linear = 0.2
        self.Ki_linear = 0.0001
        self.Kd_linear = 0.01

        self.Kp_angular = 0.1
        self.Ki_angular = 0.00001
        self.Kd_angular = 0.005

        self.prev_err_dis = 0.0
        self.prev_err_theta = 0.0
        self.integral_dis = 0.0
        self.integral_theta = 0.0

        # 設定攝影機
        self.cap = cv2.VideoCapture(0)  # 攝影機索引
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()

        # 設定目標 ArUco 標記 ID
        self.robot_id = 1
        self.target_id = 2

    def detect_aruco(self):
        """ 透過攝影機影像偵測 ArUco 標記 """
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        positions = {}
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                c = corners[i][0]
                center_x = int(np.mean(c[:, 0]))
                center_y = int(np.mean(c[:, 1]))
                positions[marker_id] = (center_x, center_y)
        
        return positions.get(self.robot_id, None), positions.get(self.target_id, None), frame

    def compute_control(self, robot_pos, target_pos):
        """ 計算 PID 控制輸出 """
        if robot_pos is None or target_pos is None:
            return None

        robot_x, robot_y = robot_pos
        target_x, target_y = target_pos

        # 計算位置誤差
        err_dis = math.sqrt((target_x - robot_x) ** 2 + (target_y - robot_y) ** 2)

        # 計算角度誤差
        target_angle = math.atan2(target_y - robot_y, target_x - robot_x)
        err_theta = target_angle  # 假設機器人方向與攝影機對齊，簡化角度計算

        # PID 控制
        self.integral_dis += err_dis
        self.integral_theta += err_theta

        derivative_dis = err_dis - self.prev_err_dis
        derivative_theta = err_theta - self.prev_err_theta

        self.prev_err_dis = err_dis
        self.prev_err_theta = err_theta

        # 計算控制輸出
        linear_speed = (self.Kp_linear * err_dis) + (self.Ki_linear * self.integral_dis) + (self.Kd_linear * derivative_dis)
        angular_speed = (self.Kp_angular * err_theta) + (self.Ki_angular * self.integral_theta) + (self.Kd_angular * derivative_theta)

        # 限制速度範圍
        linear_speed = max(min(linear_speed, 0.2), -0.2)
        angular_speed = max(min(angular_speed, 1.0), -1.0)

        return linear_speed, angular_speed

    def control_loop(self):
        """ 主要控制迴圈 """
        robot_pos, target_pos, frame = self.detect_aruco()
        
        if frame is not None:
            cv2.imshow("Aruco Detection", frame)
            cv2.waitKey(1)

        if robot_pos is None or target_pos is None:
            return

        control_output = self.compute_control(robot_pos, target_pos)
        if control_output is None:
            return

        linear_speed, angular_speed = control_output

        # 發送控制訊號
        cmd = Twist()
        cmd.linear.x = linear_speed
        cmd.angular.z = angular_speed
        if abs(self.err_dis) < 0.02 and abs(self.err_theta) < 0.05:  # 誤差很小時
            cmd = Twist()  # 建立空的 Twist（速度為 0）
            self.cmd_pub.publish(cmd)  # 立即清除運動
        else:
            self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoPIDController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
