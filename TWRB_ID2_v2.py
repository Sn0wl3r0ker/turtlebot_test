import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from cv2 import aruco

class ArucoPWMController(Node):
    def __init__(self):
        super().__init__('aruco_pwm_controller')

        self.bridge = CvBridge()
        self.image1 = None
        self.image2 = None
        self.merged_frame = None

        self.subscription1 = self.create_subscription(
            Image,
            '/camera1/image_raw',
            self.image1_callback,
            10)

        self.subscription2 = self.create_subscription(
            Image,
            '/camera2/image_raw',
            self.image2_callback,
            10)

        self.pwm_pub = self.create_publisher(Int16MultiArray, '/set_pwm', 5)
        self.timer = self.create_timer(0.15, self.control_loop)

        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
        self.aruco_params = aruco.DetectorParameters()

        self.robot_id = 2
        self.target_id = 1

        # 控制參數
        self.Kp_linear = 35.0
        self.Ki_linear = 0.05
        self.Kd_linear = 10.0
        self.Kp_angular = 25.0
        self.Ki_angular = 0.0
        self.Kd_angular = 8.0
        self.prev_err_dis = 0.0
        self.prev_err_theta = 0.0
        self.integral_dis = 0.0
        self.integral_theta = 0.0
        self.max_pwm = 50
        self.deadzone = 30

    def image1_callback(self, msg):
        self.image1 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.update_merged_frame()

    def image2_callback(self, msg):
        self.image2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.update_merged_frame()

    def update_merged_frame(self):
        if self.image1 is not None and self.image2 is not None:
            self.merged_frame = np.hstack((self.image1, self.image2))

    def detect_aruco(self):
        if self.merged_frame is None:
            return None, None, None

        gray = cv2.cvtColor(self.merged_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        robot_pos = None
        target_pos = None

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                c = corners[i][0]
                center_x = int(np.mean(c[:, 0]))
                center_y = int(np.mean(c[:, 1]))

                if marker_id == self.robot_id:
                    robot_pos = (center_x, center_y)
                    cv2.circle(self.merged_frame, (center_x, center_y), 8, (255, 0, 0), -1)
                    cv2.putText(self.merged_frame, "Robot", (center_x + 5, center_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                elif marker_id == self.target_id:
                    target_pos = (center_x, center_y)
                    cv2.circle(self.merged_frame, (center_x, center_y), 8, (0, 255, 0), -1)
                    cv2.putText(self.merged_frame, "Target", (center_x + 5, center_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return robot_pos, target_pos

    def compute_pwm(self, robot_pos, target_pos):
        if robot_pos is None or target_pos is None:
            return [0, 0]

        dx = target_pos[0] - robot_pos[0]
        dy = target_pos[1] - robot_pos[1]
        err_dis = math.sqrt(dx ** 2 + dy ** 2)
        err_theta = math.atan2(dy, dx)

        if err_dis < 10:  # pixel threshold
            print("✅ 已到達目標，停止")
            return [0, 0]

        self.integral_dis += err_dis
        self.integral_theta += err_theta
        derivative_dis = err_dis - self.prev_err_dis
        derivative_theta = err_theta - self.prev_err_theta
        self.prev_err_dis = err_dis
        self.prev_err_theta = err_theta

        linear_output = self.Kp_linear * err_dis + self.Ki_linear * self.integral_dis + self.Kd_linear * derivative_dis
        angular_output = self.Kp_angular * err_theta + self.Ki_angular * self.integral_theta + self.Kd_angular * derivative_theta

        left_pwm = linear_output - angular_output
        right_pwm = linear_output + angular_output

        left_pwm = int(max(min(left_pwm, self.max_pwm), -self.max_pwm))
        right_pwm = int(max(min(right_pwm, self.max_pwm), -self.max_pwm))

        if 0 < abs(left_pwm) < self.deadzone:
            left_pwm = int(math.copysign(self.deadzone, left_pwm))
        if 0 < abs(right_pwm) < self.deadzone:
            right_pwm = int(math.copysign(self.deadzone, right_pwm))

        print(f"📏 err_dis: {err_dis:.2f}, 🔄 err_theta: {err_theta:.2f}, PWM: [{left_pwm}, {right_pwm}]")
        return [left_pwm, right_pwm]

    def control_loop(self):
        robot_pos, target_pos = self.detect_aruco()
        if self.merged_frame is not None:
            cv2.imshow("Merged View", self.merged_frame)
            cv2.waitKey(1)

        if robot_pos is None or target_pos is None:
            print("⚠️ 無法偵測機器人或目標")
            return

        pwm_values = self.compute_pwm(robot_pos, target_pos)
        pwm_msg = Int16MultiArray()
        pwm_msg.data = pwm_values
        self.pwm_pub.publish(pwm_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoPWMController()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
