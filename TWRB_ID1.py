import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray
import cv2
import numpy as np
import math
from cv2 import aruco

class ArucoFollowController(Node):
    def __init__(self):
        super().__init__('aruco_follow_controller')

        self.pwm_pub = self.create_publisher(Int16MultiArray, '/set_pwm', 1)
        self.timer = self.create_timer(0.1, self.control_loop)

        self.Kp_linear = 40.0
        self.Ki_linear = 0.5
        self.Kd_linear = 5.0

        self.Kp_angular = 50.0
        self.Ki_angular = 0.0
        self.Kd_angular = 10.0

        self.prev_err_dis = 0.0
        self.prev_err_theta = 0.0
        self.integral_dis = 0.0
        self.integral_theta = 0.0

        self.prev_left_pwm = 0
        self.prev_right_pwm = 0
        self.max_pwm_step = 5
        self.max_pwm_value = 50
        self.min_pwm_threshold = 30  # 補償大一點，確保馬達能動

        self.cap = cv2.VideoCapture(2)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
        self.aruco_params = aruco.DetectorParameters()

        self.follower_id = 2
        self.leader_id = 1
        self.marker_length = 0.06

        self.camera_matrix = np.array([[641.2391308, 0., 316.90188846],
                                       [0., 639.76069811, 227.92853594],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array([8.27101136e-03, 2.35184440e-01, 4.10730291e-03, 3.48728526e-04, -1.40848823e+00])

        self.target_pos = None
        self.target_reached = False
        self.align_done = False

    def detect_aruco(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None, frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        follower_pos = None
        follower_ori = None
        leader_pos = None
        leader_ori = None

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                    corners[i], self.marker_length, self.camera_matrix, self.dist_coeffs
                )
                x, y, z = tvec[0][0]

                rmat, _ = cv2.Rodrigues(rvec)
                y_axis = rmat @ np.array([0, 1, 0])
                theta = math.atan2(y_axis[0], y_axis[1])

                if marker_id == self.follower_id:
                    follower_pos = (x, y, z)
                    follower_ori = theta
                elif marker_id == self.leader_id:
                    leader_pos = (x, y, z)
                    leader_ori = theta
                    self.leader_ori = theta  # 記錄 leader y 軸朝向
                    print(f"📌 ID1 ArUco 相對位置: x={x:.2f}, y={y:.2f}, z={z:.2f}")

                    # 計算 ID2 目標位置：往 ID1 背後移動 0.15m
                    opposite_y = y - 0.15 * math.cos(theta)
                    opposite_x = x - 0.15 * math.sin(theta)
                    self.target_pos = (opposite_x, opposite_y)
                    print(f"🎯 ID2 target: x={opposite_x:.2f}, y={opposite_y:.2f}")

                    # --- 畫紅色正方形框出目標位置 ---
                    target_point = np.array([[opposite_x, opposite_y, 0.0]], dtype=np.float32)
                    image_points, _ = cv2.projectPoints(
                        target_point,
                        np.array([[0.0], [0.0], [0.0]]),
                        np.array([[0.0], [0.0], [0.0]]),
                        self.camera_matrix,
                        self.dist_coeffs
                    )
                    px, py = int(image_points[0][0][0]), int(image_points[0][0][1])
                    cv2.rectangle(frame, (px - 5, py - 5), (px + 5, py + 5), (0, 0, 255), 2)
                    cv2.putText(frame, "ID2 target", (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                aruco.drawDetectedMarkers(frame, [corners[i]])
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length * 0.5)

        return follower_pos, follower_ori, leader_pos, leader_ori, frame

    def apply_deadzone(self, pwm):
        if pwm == 0:
            return 0
        # 只要不是0，強制補償到最小可動值
        return int(math.copysign(max(abs(pwm), self.min_pwm_threshold), pwm))

    def compute_pwm(self, follower_pos, follower_ori):
        if follower_pos is None or follower_ori is None or self.target_pos is None:
            return [0, 0]

        dx = self.target_pos[0] - follower_pos[0]
        dy = self.target_pos[1] - follower_pos[1]
        distance = math.sqrt(dx**2 + dy**2)

        # 目標方向
        target_theta = math.atan2(dy, dx)
        err_theta_to_target = (target_theta - follower_ori + math.pi) % (2 * math.pi) - math.pi

        # 目標到達後，對齊 y 軸
        if hasattr(self, 'final_align') and self.final_align:
            # 只做第二次轉向，對齊 y 軸
            if abs(self.leader_yaw_to_world) > 0.05:
                self.integral_theta += self.leader_yaw_to_world
                derivative_theta = self.leader_yaw_to_world - self.prev_err_theta
                self.prev_err_theta = self.leader_yaw_to_world
                linear_pwm = 0
                angular_pwm = self.Kp_angular * self.leader_yaw_to_world + self.Ki_angular * self.integral_theta + self.Kd_angular * derivative_theta
            else:
                print("✅ 已完成最終對齊")
                return [0, 0]
        # 階段 1: 先轉向面向 target
        elif not self.target_reached:
            if abs(err_theta_to_target) > 0.05:
                self.integral_theta += err_theta_to_target
                derivative_theta = err_theta_to_target - self.prev_err_theta
                self.prev_err_theta = err_theta_to_target
                linear_pwm = 0
                angular_pwm = self.Kp_angular * err_theta_to_target + self.Ki_angular * self.integral_theta + self.Kd_angular * derivative_theta
            elif distance > 0.02:
                # 轉向完成，開始前進
                self.integral_dis += distance
                derivative_dis = distance - self.prev_err_dis
                self.prev_err_dis = distance
                linear_pwm = self.Kp_linear * distance + self.Ki_linear * self.integral_dis + self.Kd_linear * derivative_dis
                angular_pwm = 0
            else:
                print("✅ 已抵達目標位置，準備最終對齊")
                self.target_reached = True
                self.final_align = True
                # 計算 y 軸對齊誤差
                if hasattr(self, 'leader_ori') and self.leader_ori is not None:
                    self.leader_yaw_to_world = (self.leader_ori - follower_ori + math.pi) % (2 * math.pi) - math.pi
                else:
                    self.leader_yaw_to_world = 0
                return [0, 0]
        else:
            return [0, 0]

        left_pwm = int(np.clip(linear_pwm - angular_pwm, -self.max_pwm_value, self.max_pwm_value))
        right_pwm = int(np.clip(linear_pwm + angular_pwm, -self.max_pwm_value, self.max_pwm_value))
        left_pwm = int(np.clip(left_pwm, self.prev_left_pwm - self.max_pwm_step, self.prev_left_pwm + self.max_pwm_step))
        right_pwm = int(np.clip(right_pwm, self.prev_right_pwm - self.max_pwm_step, self.prev_right_pwm + self.max_pwm_step))
        alpha = 0.5
        left_pwm = int(alpha * left_pwm + (1 - alpha) * self.prev_left_pwm)
        right_pwm = int(alpha * right_pwm + (1 - alpha) * self.prev_right_pwm)
        self.prev_left_pwm = left_pwm
        self.prev_right_pwm = right_pwm
        left_pwm = self.apply_deadzone(left_pwm)
        right_pwm = self.apply_deadzone(right_pwm)
        return [left_pwm, right_pwm]

    def control_loop(self):
        follower_pos, follower_ori, leader_pos, leader_ori, frame = self.detect_aruco()
        if frame is not None:
            cv2.imshow("Aruco Follow", frame)
            cv2.waitKey(1)

        pwm_values = self.compute_pwm(follower_pos, follower_ori)
        pwm_msg = Int16MultiArray()
        pwm_msg.data = pwm_values
        self.pwm_pub.publish(pwm_msg)
        print(f"🔧 PWM: {pwm_values}")

def main(args=None):
    rclpy.init(args=args)
    node = ArucoFollowController()
    rclpy.spin(node)
    node.cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
