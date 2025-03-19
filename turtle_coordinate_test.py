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
        self.cap = cv2.VideoCapture(2)  # 攝影機索引
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.aruco_params = aruco.DetectorParameters()

        # 設定 ArUco 標記 ID
        self.robot_id = 1  # TurtleBot ID
        
        # Homography 矩陣 (手動標定後填入)
        self.H = np.array([[1.2, 0.0, -300], 
                           [0.0, 1.2, -200], 
                           [0.0, 0.0, 1]])
        
        # 使用者輸入目標座標
        self.target_px, self.target_py = self.get_target_coordinates()
        self.target_x, self.target_y = self.pixel_to_world(self.target_px, self.target_py)

    def get_target_coordinates(self):
        x = int(input("請輸入目標 X 座標 (像素): "))
        y = int(input("請輸入目標 Y 座標 (像素): "))
        return x, y

    def pixel_to_world(self, px, py):
        """ 轉換影像座標 (像素) 到世界座標 """
        pixel_point = np.array([px, py, 1]).reshape(3, 1)
        world_point = np.dot(self.H, pixel_point)
        world_x, world_y = world_point[0][0] / world_point[2][0], world_point[1][0] / world_point[2][0]
        return world_x, world_y

    def detect_aruco(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        positions = {}
        orientations = {}

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                c = corners[i][0]
                center_x = int(np.mean(c[:, 0]))
                center_y = int(np.mean(c[:, 1]))
                positions[marker_id] = self.pixel_to_world(center_x, center_y)

                dx = c[0, 0] - c[1, 0]
                dy = c[0, 1] - c[1, 1]
                angle = math.atan2(dy, dx)
                orientations[marker_id] = angle
                
                # 標記 ID 1 並顯示座標
                if marker_id == self.robot_id:
                    cv2.polylines(frame, [c.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, f"ID: {marker_id} ({center_x}, {center_y})", (center_x, center_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return (positions.get(self.robot_id, None), 
                self.target_x, self.target_y,
                orientations.get(self.robot_id, None),
                frame)

    def control_loop(self):
        robot_pos, _, _, robot_angle, frame = self.detect_aruco()
        
        if frame is not None:
            cv2.drawMarker(frame, (self.target_px, self.target_py), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            cv2.imshow("Aruco Detection", frame)
            cv2.waitKey(1)

        if robot_pos is None or robot_angle is None:
            return

        control_output = self.compute_control(robot_pos, (self.target_x, self.target_y), robot_angle)
        if control_output is None:
            return

        linear_speed, angular_speed, err_dis, err_theta = control_output

        cmd = Twist()
        if abs(err_dis) < 5 and abs(err_theta) < 0.05:
            self.get_logger().info("目標已到達，停止移動")
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            cmd.linear.x = linear_speed
            cmd.angular.z = angular_speed

        self.cmd_pub.publish(cmd)

    def compute_control(self, robot_pos, target_pos, robot_angle):
        robot_x, robot_y = robot_pos
        err_dis = math.sqrt((target_pos[0] - robot_x) ** 2 + (target_pos[1] - robot_y) ** 2)
        target_angle = math.atan2(target_pos[1] - robot_y, target_pos[0] - robot_x)
        err_theta = math.atan2(math.sin(target_angle - robot_angle), math.cos(target_angle - robot_angle))

        self.integral_dis += err_dis
        self.integral_theta += err_theta

        derivative_dis = err_dis - self.prev_err_dis
        derivative_theta = err_theta - self.prev_err_theta

        self.prev_err_dis = err_dis
        self.prev_err_theta = err_theta

        linear_speed = -((self.Kp_linear * err_dis) + (self.Ki_linear * self.integral_dis) + (self.Kd_linear * derivative_dis))
        angular_speed = (self.Kp_angular * err_theta) + (self.Ki_angular * self.integral_theta) + (self.Kd_angular * derivative_theta)

        linear_speed = max(min(linear_speed, 0.2), -0.2)
        angular_speed = max(min(angular_speed, 1.0), -1.0)

        return linear_speed, angular_speed, err_dis, err_theta


def main(args=None):
    rclpy.init(args=args)
    node = ArucoPIDController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
