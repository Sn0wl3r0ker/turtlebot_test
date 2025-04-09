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

        self.Kp_linear = 50.0
        self.Ki_linear = 0.01
        self.Kd_linear = 5.0

        self.Kp_angular = 100.0
        self.Ki_angular = 0.01
        self.Kd_angular = 2.0

        self.prev_err_dis = 0.0
        self.prev_err_theta = 0.0
        self.integral_dis = 0.0
        self.integral_theta = 0.0

        self.cap = cv2.VideoCapture(2)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
        self.aruco_params = aruco.DetectorParameters()

        self.robot_id = 2
        self.marker_length = 0.06

        self.camera_matrix = np.array([[641.2391308, 0., 316.90188846],
                                       [0., 639.76069811, 227.92853594],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array([8.27101136e-03, 2.35184440e-01, 4.10730291e-03, 3.48728526e-04, -1.40848823e+00])

        self.image_width = 640
        self.image_height = 480
        self.fixed_z = 1.1
        self.target_pixel_x = 320.0
        self.target_pixel_y = 240.0

    def set_target_pixel(self):
        try:
            self.target_pixel_x = float(input(f"輸入目標 X 像素座標 (0-{self.image_width}): "))
            self.target_pixel_y = float(input(f"輸入目標 Y 像素座標 (0-{self.image_height}): "))
        except ValueError:
            print("輸入無效，使用預設值 (320, 240)")
            self.target_pixel_x = 320.0
            self.target_pixel_y = 240.0

    def pixel_to_camera(self, pixel_x, pixel_y, z_depth):
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        x = (pixel_x - cx) * z_depth / fx
        y = (pixel_y - cy) * z_depth / fy
        z = z_depth
        return x, y, z

    def detect_aruco(self):
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
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                        corners[i], self.marker_length, self.camera_matrix, self.dist_coeffs
                    )
                    x, y, z = tvec[0][0]
                    robot_pos = (x, y, z)

                    rmat, _ = cv2.Rodrigues(rvec)
                    y_axis = rmat @ np.array([0, 1, 0])
                    robot_orientation = math.atan2(y_axis[0], y_axis[1])

                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length * 0.5)

        cv2.circle(frame, (int(self.target_pixel_x), int(self.target_pixel_y)), 10, (0, 255, 0), 2)
        cv2.putText(frame, "Target", (int(self.target_pixel_x) + 15, int(self.target_pixel_y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return robot_pos, robot_orientation, frame

    def compute_pwm(self, robot_pos, robot_orientation):
        if robot_pos is None or robot_orientation is None:
            return [0, 0]

        robot_x, robot_y, _ = robot_pos
        target_x, target_y, _ = self.pixel_to_camera(self.target_pixel_x, self.target_pixel_y, self.fixed_z)

        relative_x = target_x - robot_x
        relative_y = target_y - robot_y
        err_dis = math.sqrt(relative_x**2 + relative_y**2)
        target_angle = math.atan2(relative_x, relative_y)
        err_theta = (target_angle - robot_orientation + math.pi) % (2 * math.pi) - math.pi

        self.integral_dis += err_dis
        self.integral_theta += err_theta

        derivative_dis = err_dis - self.prev_err_dis
        derivative_theta = err_theta - self.prev_err_theta

        self.prev_err_dis = err_dis
        self.prev_err_theta = err_theta

        linear_pwm = self.Kp_linear * err_dis + self.Ki_linear * self.integral_dis + self.Kd_linear * derivative_dis
        angular_pwm = self.Kp_angular * err_theta + self.Ki_angular * self.integral_theta + self.Kd_angular * derivative_theta

        if abs(err_theta) > 0.3:
            linear_pwm = 0

        left_pwm = int(max(min(linear_pwm - angular_pwm, 255), -255))
        right_pwm = int(max(min(linear_pwm + angular_pwm, 255), -255))

        if err_dis < 0.05:
            print("🎯 已到達目標，停止")
            self.integral_dis = 0.0
            self.integral_theta = 0.0
            return [0, 0]

        return [left_pwm, right_pwm]

    def control_loop(self):
        robot_pos, robot_orientation, frame = self.detect_aruco()
        if frame is not None:
            cv2.imshow("Aruco Tracking", frame)
            cv2.waitKey(1)

        pwm_values = self.compute_pwm(robot_pos, robot_orientation)
        pwm_msg = Int16MultiArray()
        pwm_msg.data = pwm_values
        self.pwm_pub.publish(pwm_msg)
        print(f"🔧 PWM: {pwm_values}")

def main(args=None):
    rclpy.init(args=args)
    node = ArucoPWMController()
    print("請輸入目標像素座標：")
    node.set_target_pixel()
    rclpy.spin(node)
    node.cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
