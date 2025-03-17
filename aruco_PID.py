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

        self.Kp_angular = 0.3
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
        self.marker_length = 0.06  # 6cm 標記

        # 相機內參數（Camera Calibration）
        self.camera_matrix = np.array([[641.2391308,    0.,         316.90188846],
                                       [0.,         639.76069811, 227.92853594],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array([8.27101136e-03,  2.35184440e-01,  4.10730291e-03,  3.48728526e-04, -1.40848823e+00])

    def detect_aruco(self):
        """ 偵測 ArUco 標記並計算機器人朝向 """
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        positions = {}
        robot_orientation = None
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                c = corners[i][0]

                # 估算 ArUco 標記的 3D 姿態
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                    corners[i], self.marker_length, self.camera_matrix, self.dist_coeffs
                )

                # 取得 ArUco 標記的 3D 座標
                x, y, z = tvec[0][0]
                positions[marker_id] = (x, y, z)

                # 如果是機器人標記，計算朝向（Y 軸方向）
                if marker_id == self.robot_id:
                    # 將旋轉向量轉換為旋轉矩陣
                    rmat, _ = cv2.Rodrigues(rvec)
                    # Y 軸在標記坐標系中是 [0, 1, 0]，轉換到相機坐標系
                    y_axis = rmat @ np.array([0, 1, 0])
                    # 計算 Y 軸在 XY 平面上的投影角度（弧度）
                    robot_orientation = math.atan2(y_axis[0], y_axis[1])

                # 繪製 3D 坐標軸
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length * 0.5)

                # 在控制台上打印座標
                print(f"Marker ID: {marker_id}, X: {x:.3f}m, Y: {y:.3f}m, Z: {z:.3f}m")

        return positions.get(self.robot_id, None), positions.get(self.target_id, None), robot_orientation, frame

    def compute_control(self, robot_pos, target_pos, robot_orientation):
        """ 計算 PID 控制輸出，考慮機器人朝向 """
        if robot_pos is None or target_pos is None or robot_orientation is None:
            return None

        robot_x, robot_y, robot_z = robot_pos
        target_x, target_y, target_z = target_pos

        # Y 軸為前進方向，計算距離誤差
        err_dis = target_y - robot_y

        # 計算目標方向（從機器人到目標）
        target_angle = math.atan2(target_x - robot_x, target_y - robot_y)

        # 角度誤差 = 目標方向 - 機器人朝向
        err_theta = target_angle - robot_orientation

        # 將角度誤差正規化到 [-pi, pi]
        err_theta = (err_theta + math.pi) % (2 * math.pi) - math.pi

        # PID 控制計算
        self.integral_dis += err_dis
        self.integral_theta += err_theta

        derivative_dis = err_dis - self.prev_err_dis
        derivative_theta = err_theta - self.prev_err_theta

        self.prev_err_dis = err_dis
        self.prev_err_theta = err_theta

        # 角速度控制
        angular_speed = (self.Kp_angular * err_theta) + (self.Ki_angular * self.integral_theta) + (self.Kd_angular * derivative_theta)
        angular_speed = max(min(angular_speed, 1.0), -1.0)  # 限制最大角速度

        print(f"🔄 角度誤差: {err_theta:.3f} rad, angular.z: {angular_speed:.3f}")

        # 若角度誤差較大，先轉向
        if abs(err_theta) > 0.1:
            linear_speed = 0.0
        else:
            linear_speed = (self.Kp_linear * err_dis) + (self.Ki_linear * self.integral_dis) + (self.Kd_linear * derivative_dis)
            linear_speed = max(min(linear_speed, 0.2), -0.2)  # 限制線速度

        return linear_speed, angular_speed

    def control_loop(self):
        """ 主要控制迴圈 """
        robot_pos, target_pos, robot_orientation, frame = self.detect_aruco()
        
        if frame is not None:
            cv2.imshow("Aruco Detection", frame)
            cv2.waitKey(1)

        if robot_pos is None or target_pos is None or robot_orientation is None:
            return

        control_output = self.compute_control(robot_pos, target_pos, robot_orientation)
        if control_output is None:
            return

        linear_speed, angular_speed = control_output

        print(f"📢 發送給 /cmd_vel -> linear.x: {linear_speed:.3f}, angular.z: {angular_speed:.3f}")

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
    node.cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()