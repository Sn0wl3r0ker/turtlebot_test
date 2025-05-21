from geometry_msgs.msg import Twist
import numpy as np
import math
import cv2
from cv2 import aruco

class ArucoPIDController:
    def __init__(self, node):
        self.node = node
        self.cmd_pub = node.create_publisher(Twist, '/cmd_vel', 5)

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

        # 相機參數
        self.camera_matrix = np.array([[641.2391308, 0., 316.90188846],
                                       [0., 639.76069811, 227.92853594],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array([8.27101136e-03, 2.35184440e-01, 4.10730291e-03, 3.48728526e-04, -1.40848823e+00])

        self.marker_length = 0.06
        self.robot_id = 1
        self.fixed_z = 1.13

        self.target_pixel = (320.0, 240.0)

    def set_target_pixel(self):
        try:
            x = float(input("輸入目標 X 像素座標 (0–640): "))
            y = float(input("輸入目標 Y 像素座標 (0–480): "))
            if not (0 <= x <= 640 and 0 <= y <= 480):
                raise ValueError
            self.target_pixel = (x, y)
            print(f"[INFO] 目標座標設為: {self.target_pixel}")
        except ValueError:
            print("⚠️ 無效輸入，使用預設值 (320, 240)")
            self.target_pixel = (320.0, 240.0)

    def pixel_to_camera(self, pixel_x, pixel_y, z_depth):
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        x = (pixel_x - cx) * z_depth / fx
        y = (pixel_y - cy) * z_depth / fy
        return x, y, z_depth

    def process(self, frame, corners, ids):
        robot_pos = None
        robot_orientation = None

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == self.robot_id:
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                        corners[i], self.marker_length, self.camera_matrix, self.dist_coeffs)
                    tvec = tvec[0][0]
                    rmat, _ = cv2.Rodrigues(rvec)
                    y_axis = rmat @ np.array([0, 1, 0])
                    theta = math.atan2(y_axis[0], y_axis[1])
                    robot_pos = tvec
                    robot_orientation = theta

                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec.reshape(3,1), self.marker_length * 0.5)

        # 畫出目標像素點
        cv2.circle(frame, (int(self.target_pixel[0]), int(self.target_pixel[1])), 10, (0, 0, 255), 2)
        cv2.putText(frame, "Target1", (int(self.target_pixel[0]) + 10, int(self.target_pixel[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 若沒偵測到則停止
        if robot_pos is None or robot_orientation is None:
            self.publish_stop()
            return None

        # 控制計算
        tx, ty, tz = self.pixel_to_camera(*self.target_pixel, self.fixed_z)
        dx, dy = tx - robot_pos[0], ty - robot_pos[1]
        err_dis = math.sqrt(dx ** 2 + dy ** 2)
        target_angle = math.atan2(dx, dy)
        err_theta = (target_angle - robot_orientation + math.pi) % (2 * math.pi) - math.pi

        # PID
        self.integral_dis += err_dis
        self.integral_theta += err_theta
        d_dis = err_dis - self.prev_err_dis
        d_theta = err_theta - self.prev_err_theta
        self.prev_err_dis = err_dis
        self.prev_err_theta = err_theta

        linear = -(self.Kp_linear * err_dis + self.Ki_linear * self.integral_dis + self.Kd_linear * d_dis) * math.cos(err_theta)
        angular = self.Kp_angular * err_theta + self.Ki_angular * self.integral_theta + self.Kd_angular * d_theta

        if abs(err_theta) > 0.3:
            linear = 0.0
        if err_dis < 0.05:
            linear = angular = 0.0
            self.integral_dis = self.integral_theta = 0.0

        twist = Twist()
        twist.linear.x = max(min(linear, 0.2), -0.2)
        twist.angular.z = max(min(angular, 1.0), -1.0)
        self.cmd_pub.publish(twist)
        return robot_pos

    def publish_stop(self):
        stop = Twist()
        stop.linear.x = 0.0
        stop.angular.z = 0.0
        self.cmd_pub.publish(stop)
