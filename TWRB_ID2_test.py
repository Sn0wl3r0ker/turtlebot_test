import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray
import cv2
import numpy as np
import math
from cv2 import aruco

class ArucoPWMController(Node):
    def __init__(self):
        super().__init__('aruco_pwm_controller')

        self.pwm_pub = self.create_publisher(Int16MultiArray, '/set_pwm', 5)
        self.timer = self.create_timer(0.15, self.control_loop)

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

        self.cap = cv2.VideoCapture(2)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
        self.aruco_params = aruco.DetectorParameters()

        self.robot_id = 2
        self.target_id = 1
        self.marker_length = 0.06

        self.camera_matrix = np.array([[641.2391308, 0., 316.90188846],
                                       [0., 639.76069811, 227.92853594],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array([8.27101136e-03, 2.35184440e-01, 4.10730291e-03, 3.48728526e-04, -1.40848823e+00])

        self.aligning = False
        self.alignment_done = False
        self.target_orientation = None

    def detect_aruco(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None, frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        robot_pos = None
        robot_orientation = None
        target_world = None

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                    corners[i], self.marker_length, self.camera_matrix, self.dist_coeffs
                )
                tvec = tvec[0][0]
                rmat, _ = cv2.Rodrigues(rvec)

                if marker_id == self.robot_id:
                    robot_pos = tuple(tvec)
                    y_axis = rmat @ np.array([0, 1, 0])
                    robot_orientation = math.atan2(y_axis[0], y_axis[1])
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec.reshape(3, 1), self.marker_length * 0.5)

                elif marker_id == self.target_id:
                    offset = -0.3
                    offset_vec = rmat @ np.array([0, offset, 0])
                    target_world = tuple(tvec + offset_vec)

                    # 顯示 target2 畫面座標
                    target_point = np.array([[target_world]], dtype=np.float32)
                    image_points, _ = cv2.projectPoints(
                        target_point,
                        np.array([[0.0], [0.0], [0.0]]),
                        np.array([[0.0], [0.0], [0.0]]),
                        self.camera_matrix,
                        self.dist_coeffs
                    )
                    px, py = int(image_points[0][0][0]), int(image_points[0][0][1])
                    cv2.circle(frame, (px, py), 8, (0, 255, 0), 2)
                    cv2.putText(frame, "Target2", (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    print(f"🎯 ID1 target world position: {target_world}, pixel: ({px}, {py})")

                    # 繪製 ArUco 1 的三維軸
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec.reshape(3, 1), self.marker_length * 0.5)

                    # 記錄 ID1 的朝向供對齊使用
                    y_axis = rmat @ np.array([0, 1, 0])
                    self.target_orientation = math.atan2(y_axis[0], y_axis[1])

        return robot_pos, robot_orientation, target_world, frame

    def compute_pwm(self, robot_pos, robot_orientation, target_pos):
        if robot_pos is None or robot_orientation is None or target_pos is None:
            return [0, 0]

        if self.alignment_done:
            return [0, 0]

        robot_x, robot_y, _ = robot_pos
        target_x, target_y, _ = target_pos

        dx = target_x - robot_x
        dy = target_y - robot_y
        err_dis = math.sqrt(dx ** 2 + dy ** 2)

        if abs(err_dis) < 0.05:
            if not self.aligning:
                print("🎯 到達目標，開始對齊 Y 軸")
                self.aligning = True
            if self.target_orientation is not None:
                err_theta = (self.target_orientation - robot_orientation + math.pi) % (2 * math.pi) - math.pi
                self.integral_theta += err_theta
                derivative_theta = err_theta - self.prev_err_theta
                self.prev_err_theta = err_theta

                angular_output = (self.Kp_angular * err_theta +
                                  self.Ki_angular * self.integral_theta +
                                  self.Kd_angular * derivative_theta)
                linear_output = 0.0

                if abs(err_theta) < 0.05:
                    print("✅ 已完成 Y 軸對齊並停止")

                    return [0, 0]
            else:
                return [0, 0]
        else:
            self.aligning = False
            target_angle = math.atan2(dx, dy)
            err_theta = (target_angle - robot_orientation + math.pi) % (2 * math.pi) - math.pi

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

        print(f"📏 err_dis: {err_dis:.3f}, 🔄 err_theta: {err_theta:.3f}, PWM: [{left_pwm}, {right_pwm}]")
        return [left_pwm, right_pwm]

    def control_loop(self):
        robot_pos, robot_orientation, target_pos, frame = self.detect_aruco()
        if frame is not None:
            cv2.imshow("Aruco Detection", frame)
            cv2.waitKey(1)

        if target_pos is None:
            print("⚠️ 無法偵測 ID1 目標")
            return

        pwm_values = self.compute_pwm(robot_pos, robot_orientation, target_pos)
        pwm_msg = Int16MultiArray()
        pwm_msg.data = pwm_values
        self.pwm_pub.publish(pwm_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoPWMController()
    rclpy.spin(node)
    node.cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()