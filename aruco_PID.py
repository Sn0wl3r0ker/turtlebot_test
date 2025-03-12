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
        
        # 發布 /cmd_vel 來控制機器人
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 5)
        self.timer = self.create_timer(0.15, self.control_loop)  # 每 0.15 秒執行一次

        # PID 參數
        self.Kp_linear = 0.5
        self.Ki_linear = 0.001
        self.Kd_linear = 0.01

        self.Kp_angular = 0.5
        self.Ki_angular = 0.0001
        self.Kd_angular = 0.01

        self.prev_err_dis = 0.0
        self.prev_err_theta = 0.0
        self.integral_dis = 0.0
        self.integral_theta = 0.0

        # 設定攝影機
        self.cap = cv2.VideoCapture(0)  # 攝影機索引
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
        self.aruco_params = aruco.DetectorParameters()

        # 設定目標 ArUco 標記 ID
        self.robot_id = 1
        self.target_id = 2

        # ArUco 標記實際大小（單位：公尺）
        self.marker_length = 0.06  # 5cm 標記

        # 相機內參數（Camera Calibration）
        self.camera_matrix = np.array([[960.42974, 0.0, 628.25951],
                                       [0.0, 960.58843, 339.99534],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array([0.01736, -0.076006, 0.002602, 0.000286, 0.0])

    def detect_aruco(self):
        """ 偵測 ArUco 標記並顯示 ID、3D 坐標（X, Y, Z） """
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        positions = {}
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                c = corners[i][0]

                # 估算 ArUco 標記的 3D 姿態（Position & Rotation）
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                    corners[i], self.marker_length, self.camera_matrix, self.dist_coeffs
                )

                # 轉換平移向量 (tvec) 為 3D 坐標
                x, y, z = tvec[0][0]

                # 顯示 Marker ID 和 3D 座標
                print(f"Marker ID: {marker_id}, X: {x:.3f}m, Y: {y:.3f}m, Z: {z:.3f}m")

                # 在影像上標示 ArUco 標記
                cv2.putText(frame, f"ID: {marker_id}", (int(c[:, 0].mean()), int(c[:, 1].mean()) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"X: {x:.3f}m, Y: {y:.3f}m, Z: {z:.3f}m", 
                            (int(c[:, 0].mean()), int(c[:, 1].mean()) + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                positions[marker_id] = (x, y, z)

        return positions.get(self.robot_id, None), positions.get(self.target_id, None), frame

    def compute_control(self, robot_pos, target_pos):
        """ 計算 PID 控制輸出 """
        if robot_pos is None or target_pos is None:
            return None

        robot_x, robot_y, robot_z = robot_pos
        target_x, target_y, target_z = target_pos

        # 設定 TurtleBot3 正前方方向（選擇 X 軸或 Y 軸）
        # 這裡假設機器人前方是 Z 軸，因此主要考慮 X 軸方向
        err_dis = target_x - robot_x  # 以 X 軸作為前進方向

        # 計算角度誤差（機器人轉向目標標記）
        target_angle = math.atan2(target_y - robot_y, target_x - robot_x)
        err_theta = target_angle

        # PID 控制計算
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

        # 當機器人接近目標時，停止
        if abs(self.prev_err_dis) < 0.02 and abs(self.prev_err_theta) < 0.05:
            print("🎯 到達目標，停止機器人")
            cmd = Twist()  # 停止機器人
        
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoPIDController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
