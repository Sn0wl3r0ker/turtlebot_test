#!/usr/bin/env python3
import rospy
from std_msgs.msg import Int16MultiArray
import cv2
import numpy as np
import math
from cv2 import aruco

class Follower:
    def __init__(self):
        rospy.init_node('aruco_follower', anonymous=True)
        self.pub = rospy.Publisher('/set_pwm', Int16MultiArray, queue_size=1)

        self.cap = cv2.VideoCapture(2)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
        self.aruco_params = aruco.DetectorParameters()

        self.target_id = 1  # 追蹤第一台車
        self.marker_length = 0.06  # 根據你的設定

        self.camera_matrix = np.array([[641.2391308, 0., 316.90188846],
                                       [0., 639.76069811, 227.92853594],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array([8.27101136e-03, 2.35184440e-01, 4.10730291e-03, 3.48728526e-04, -1.40848823e+00])

        self.Kp_dis = 40
        self.Kp_angle = 25
        self.desired_distance = 0.5  # 距離維持 50cm

        self.loop()

    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id == self.target_id:
                        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                            corners[i], self.marker_length, self.camera_matrix, self.dist_coeffs
                        )
                        x, y, z = tvec[0][0]

                        distance_error = math.sqrt(x**2 + z**2) - self.desired_distance
                        angle_error = math.atan2(x, z)

                        linear = self.Kp_dis * distance_error
                        angular = self.Kp_angle * angle_error

                        left_pwm = int(linear - angular)
                        right_pwm = int(linear + angular)

                        # 限制 PWM
                        left_pwm = max(min(left_pwm, 60), -60)
                        right_pwm = max(min(right_pwm, 60), -60)

                        # Deadzone 補償
                        def apply_deadzone(pwm):
                            if abs(pwm) < 25:
                                return 0
                            return pwm
                        left_pwm = apply_deadzone(left_pwm)
                        right_pwm = apply_deadzone(right_pwm)

                        msg = Int16MultiArray()
                        msg.data = [left_pwm, right_pwm]
                        self.pub.publish(msg)

                        cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)
                        break  # 找到就不再處理其他 ID

            cv2.imshow("Follower View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            rate.sleep()

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        Follower()
    except rospy.ROSInterruptException:
        pass
