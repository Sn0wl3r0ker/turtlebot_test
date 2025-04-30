from std_msgs.msg import Int16MultiArray
import numpy as np
import math
import cv2
from cv2 import aruco

class ArucoPWMController:
    def __init__(self, node):
        self.node = node
        self.pwm_pub = node.create_publisher(Int16MultiArray, '/set_pwm', 5)

        self.Kp_linear = 20.0
        self.Ki_linear = 0.1
        self.Kd_linear = 20.0

        self.Kp_angular = 10.0
        self.Ki_angular = 0.01
        self.Kd_angular = 30.0

        self.prev_err_dis = 0.0
        self.prev_err_theta = 0.0
        self.integral_dis = 0.0
        self.integral_theta = 0.0

        self.max_pwm = 50
        self.deadzone = 30

        self.marker_length = 0.06
        self.camera_matrix = np.array([[641.2391308, 0., 316.90188846],
                                       [0., 639.76069811, 227.92853594],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array([8.27101136e-03, 2.35184440e-01, 4.10730291e-03, 3.48728526e-04, -1.40848823e+00])

        self.robot_id = 2
        self.target_id = 1
        self.aligning = False
        self.target_orientation = None
        self.stop_sent = False  # ✅ 加入對齊停止旗標

    def process(self, frame, corners, ids):
        robot_pos, robot_orient, target_pos = None, None, None

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                    corners[i], self.marker_length, self.camera_matrix, self.dist_coeffs)
                tvec = tvec[0][0]
                rmat, _ = cv2.Rodrigues(rvec)
                y_axis = rmat @ np.array([0, 1, 0])
                theta = math.atan2(y_axis[0], y_axis[1])

                if marker_id == self.robot_id:
                    robot_pos = tvec
                    robot_orient = theta
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec.reshape(3, 1), self.marker_length * 0.5)

                elif marker_id == self.target_id:
                    offset_vec = rmat @ np.array([0, -0.3, 0])
                    target_pos = tuple(tvec + offset_vec)
                    self.target_orientation = theta

                    image_point = np.array([[target_pos]], dtype=np.float32)
                    proj, _ = cv2.projectPoints(image_point, np.zeros((3,1)), np.zeros((3,1)), self.camera_matrix, self.dist_coeffs)
                    px, py = int(proj[0][0][0]), int(proj[0][0][1])
                    cv2.circle(frame, (px, py), 8, (0, 255, 0), 2)
                    cv2.putText(frame, "Target2", (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if target_pos is None:
            print("⚠️ 無法偵測 ID1 目標")
            return

        pwm_values = self.compute_pwm(robot_pos, robot_orient, target_pos)
        pwm_msg = Int16MultiArray()
        pwm_msg.data = pwm_values
        self.pwm_pub.publish(pwm_msg)

    def compute_pwm(self, robot_pos, robot_orient, target_pos):
        if robot_pos is None or robot_orient is None or target_pos is None:
            return [0, 0]

        dx = target_pos[0] - robot_pos[0]
        dy = target_pos[1] - robot_pos[1]
        err_dis = math.sqrt(dx ** 2 + dy ** 2)

        if err_dis < 0.05:
            if self.target_orientation is not None:
                err_theta = (self.target_orientation - robot_orient + math.pi) % (2 * math.pi) - math.pi
                self.integral_theta += err_theta
                derivative_theta = err_theta - self.prev_err_theta
                self.prev_err_theta = err_theta

                angular_output = (self.Kp_angular * err_theta +
                                  self.Ki_angular * self.integral_theta +
                                  self.Kd_angular * derivative_theta)
                linear_output = 0.0

                if abs(err_theta) < 0.05:
                    if not self.stop_sent:
                        print("✅ ID2 對齊完成，傳送一次停止PWM")
                        self.stop_sent = True
                        return [0, 0]
                    else:
                        return [0, 0]
            else:
                return [0, 0]
        else:
            self.stop_sent = False  # ✅ 若有移動，重新允許送停止PWM

            target_angle = math.atan2(dx, dy)
            err_theta = (target_angle - robot_orient + math.pi) % (2 * math.pi) - math.pi

            self.integral_dis += err_dis
            self.integral_theta += err_theta

            derivative_dis = err_dis - self.prev_err_dis
            derivative_theta = err_theta - self.prev_err_theta
            self.prev_err_dis = err_dis
            self.prev_err_theta = err_theta

            linear_output = self.Kp_linear * err_dis + self.Ki_linear * self.integral_dis + self.Kd_linear * derivative_dis
            angular_output = self.Kp_angular * err_theta + self.Ki_angular * self.integral_theta + self.Kd_angular * derivative_theta

            if abs(err_theta) > 0.3:
                linear_output = 0.0

        left_pwm = linear_output - angular_output
        right_pwm = linear_output + angular_output

        left_pwm = int(max(min(left_pwm, self.max_pwm), -self.max_pwm))
        right_pwm = int(max(min(right_pwm, self.max_pwm), -self.max_pwm))

        if 0 < abs(left_pwm) < self.deadzone:
            left_pwm = int(math.copysign(self.deadzone, left_pwm))
        if 0 < abs(right_pwm) < self.deadzone:
            right_pwm = int(math.copysign(self.deadzone, right_pwm))

        return [left_pwm, right_pwm]
