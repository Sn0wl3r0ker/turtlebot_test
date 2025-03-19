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
        self.Kp_linear = 0.4
        self.Ki_linear = 0.00005
        self.Kd_linear = 0.05

        self.Kp_angular = 1.0
        self.Ki_angular = 0.0001
        self.Kd_angular = 0.005

        self.prev_err_dis = 0.0
        self.prev_err_theta = 0.0
        self.integral_dis = 0.0
        self.integral_theta = 0.0

        # 設定攝影機
        self.cap = cv2.VideoCapture(2)  # 攝影機索引
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
        self.aruco_params = aruco.DetectorParameters()

        # 設定機器人 ArUco 標記 ID
        self.robot_id = 1

        # ArUco 標記實際大小（單位：公尺）
        self.marker_length = 0.06  # 6cm 標記

        # 相機內參數（Camera Calibration）
        self.camera_matrix = np.array([[641.2391308, 0., 316.90188846],
                                       [0., 639.76069811, 227.92853594],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array([8.27101136e-03, 2.35184440e-01, 4.10730291e-03, 3.48728526e-04, -1.40848823e+00])

        # 畫面解析度
        self.image_width = 640
        self.image_height = 480

        # 假設 Z 軸距離固定（單位：公尺）
        self.fixed_z = 1.13

        # 初始目標像素座標（預設值）
        self.target_pixel_x = 320.0  # 畫面中心
        self.target_pixel_y = 240.0  # 畫面中心

    def set_target_pixel(self):
        """ 讓使用者輸入目標像素座標 """
        try:
            self.target_pixel_x = float(input(f"輸入目標 X 像素座標 (0-{self.image_width}): "))
            self.target_pixel_y = float(input(f"輸入目標 Y 像素座標 (0-{self.image_height}): "))
            
            # 檢查輸入是否在有效範圍內
            if not (0 <= self.target_pixel_x <= self.image_width and 0 <= self.target_pixel_y <= self.image_height):
                print("輸入超出範圍，使用預設值 (320, 240)")
                self.target_pixel_x = 320.0
                self.target_pixel_y = 240.0
        except ValueError:
            print("輸入無效，使用預設值 (320, 240)")
            self.target_pixel_x = 320.0
            self.target_pixel_y = 240.0

    def pixel_to_camera(self, pixel_x, pixel_y, z_depth):
        """ 將像素座標轉換為相機座標系中的 3D 座標 """
        fx = self.camera_matrix[0, 0]  # 焦距 X
        fy = self.camera_matrix[1, 1]  # 焦距 Y
        cx = self.camera_matrix[0, 2]  # 光心 X
        cy = self.camera_matrix[1, 2]  # 光心 Y

        # 根據相機模型反算 X, Y（假設 Z 已知）
        x = (pixel_x - cx) * z_depth / fx
        y = (pixel_y - cy) * z_depth / fy
        z = z_depth

        return x, y, z

    def detect_aruco(self):
        """ 偵測 ArUco 標記並計算機器人位置與朝向 """
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        robot_pos = None
        robot_orientation = None
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == self.robot_id:
                    c = corners[i][0]

                    # 估算 ArUco 標記的 3D 姿態
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                        corners[i], self.marker_length, self.camera_matrix, self.dist_coeffs
                    )

                    # 取得機器人的 3D 座標
                    x, y, z = tvec[0][0]
                    robot_pos = (x, y, z)

                    # 計算機器人朝向（Y 軸方向）
                    rmat, _ = cv2.Rodrigues(rvec)
                    y_axis = rmat @ np.array([0, 1, 0])
                    robot_orientation = math.atan2(y_axis[0], y_axis[1])

                    # 繪製 3D 坐標軸
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length * 0.5)

                    print(f"Robot ID: {marker_id}, X: {x:.3f}m, Y: {y:.3f}m, Z: {z:.3f}m")

        # 在影像上標記目標位置
        cv2.circle(frame, (int(self.target_pixel_x), int(self.target_pixel_y)), 10, (0, 255, 0), 2)
        cv2.putText(frame, "Target", (int(self.target_pixel_x) + 15, int(self.target_pixel_y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return robot_pos, robot_orientation, frame

    def compute_control(self, robot_pos, robot_orientation):
        """ 計算 PID 控制輸出，移動到相對於 ID1 的目標座標 """
        if robot_pos is None or robot_orientation is None:
            return None

        robot_x, robot_y, robot_z = robot_pos

        # 將目標像素座標轉換為相機座標系
        target_x, target_y, target_z = self.pixel_to_camera(self.target_pixel_x, self.target_pixel_y, self.fixed_z)

        # 計算相對於 ID1 的相對座標
        relative_x = target_x - robot_x
        relative_y = target_y - robot_y

        # 計算距離誤差（僅考慮 X 和 Y，因為 Z 固定）
        err_dis = math.sqrt(relative_x**2 + relative_y**2)

        # 計算目標方向（相對於 ID1）
        target_angle = math.atan2(relative_x, relative_y)

        # 角度誤差 = 目標方向 - 機器人朝向
        err_theta = target_angle - robot_orientation
        err_theta = (err_theta + math.pi) % (2 * math.pi) - math.pi  # 正規化到 [-pi, pi]

        # PID 控制計算
        self.integral_dis += err_dis
        self.integral_theta += err_theta

        # 限制積分項
        if err_dis < 0.26:
            self.integral_dis = min(max(self.integral_dis, -10.0), 10.0)

        derivative_dis = err_dis - self.prev_err_dis
        derivative_theta = err_theta - self.prev_err_theta

        self.prev_err_dis = err_dis
        self.prev_err_theta = err_theta

        # 角速度控制
        angular_speed = (self.Kp_angular * err_theta) + (self.Ki_angular * self.integral_theta) + (self.Kd_angular * derivative_theta)
        angular_speed = max(min(angular_speed, 1.0), -1.0)

        # 線速度控制
        base_linear_speed = (self.Kp_linear * err_dis) + (self.Ki_linear * self.integral_dis) + (self.Kd_linear * derivative_dis)
        linear_speed = -base_linear_speed * math.cos(err_theta)
        linear_speed = max(min(linear_speed, 0.2), -0.2)

        print(f"📏 距離誤差: {err_dis:.3f} m, base_linear: {base_linear_speed:.3f}")
        print(f"🔄 角度誤差: {err_theta:.3f} rad, angular.z: {angular_speed:.3f}")
        print(f"🎯 目標相對座標: X: {relative_x:.3f}m, Y: {relative_y:.3f}m")

        # 若角度誤差較大，先轉向
        if abs(err_theta) > 0.3:
            linear_speed = 0.0

        return linear_speed, angular_speed

    def control_loop(self):
        """ 主要控制迴圈 """
        robot_pos, robot_orientation, frame = self.detect_aruco()
        
        if frame is not None:
            cv2.imshow("Aruco Detection", frame)
            cv2.waitKey(1)

        if robot_pos is None or robot_orientation is None:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            return

        control_output = self.compute_control(robot_pos, robot_orientation)
        if control_output is None:
            return

        linear_speed, angular_speed = control_output

        print(f"📢 發送給 /cmd_vel -> linear.x: {linear_speed:.3f}, angular.z: {angular_speed:.3f}")

        # 發送控制訊號
        cmd = Twist()
        cmd.linear.x = linear_speed
        cmd.angular.z = angular_speed

        # 當機器人接近目標時，停止並重置積分
        if abs(self.prev_err_dis) < 0.05:
            print("🎯 到達目標，停止機器人")
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.integral_dis = 0.0
            self.integral_theta = 0.0

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoPIDController()
    
    # 在程式啟動時讓使用者輸入目標座標
    print("請輸入目標像素座標：")
    node.set_target_pixel()
    
    rclpy.spin(node)
    node.cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()