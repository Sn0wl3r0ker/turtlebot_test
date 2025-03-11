import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

# 初始化 ROS 節點
rospy.init_node('aruco_pid_controller')
bridge = CvBridge()
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

# 設定 ArUco 字典與參數
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

# 開啟相機
cap = cv2.VideoCapture(2)

# PID 參數 (調高 Kp 以增加靈敏度)
Kp_x = 0.1
Ki_x = 0.001
Kd_x = 0.01

Kp_y = 0.1  # 角速度控制的 Kp 調高，使轉向更靈敏
Ki_y = 0.001
Kd_y = 0.01

# 誤差累積與前一次誤差
error_sum_x = 0
error_sum_y = 0
last_error_x = 0
last_error_y = 0

# 影像像素轉換為真實世界比例（需根據測試調整，假設 1 公尺 = 500 像素）
pixels_per_meter = 500.0

# 設定最小與最大速度
min_speed = 0.1  # 調低最小速度閾值，確保機器人能動
max_speed = 0.3   # 防止速度過快

# 允許使用者輸入目標座標
target_x = float(input("輸入目標 X 座標: "))
target_y = float(input("輸入目標 Y 座標: "))

while not rospy.is_shutdown():
    ret, frame = cap.read()
    if not ret:
        break

    # 偵測 ArUco 標記
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        bot_position = None

        for i in range(len(ids)):
            marker_id = ids[i][0]
            marker_corners = corners[i]
            marker_center = np.mean(marker_corners[0], axis=0)  # 計算標記中心點

            print("Marker ID: {}, Position: {}".format(marker_id, marker_center))

            if marker_id == 1:  # 將 ID 1 視為機器人
                bot_position = marker_center

        if bot_position is not None:
            # 計算誤差（目標 - 當前位置）
            error_x = (target_x - bot_position[0]) / pixels_per_meter  # 轉換為公尺
            error_y = (target_y - bot_position[1]) / pixels_per_meter

            # PID 控制計算
            error_sum_x += error_x
            error_sum_y += error_y

            d_error_x = error_x - last_error_x
            d_error_y = error_y - last_error_y

            control_x = Kp_x * error_x + Ki_x * error_sum_x + Kd_x * d_error_x
            control_y = Kp_y * error_y + Ki_y * error_sum_y + Kd_y * d_error_y

            last_error_x = error_x
            last_error_y = error_y

            # 設定最小速度閾值，避免機器人停住
            control_x = max(min(control_x, max_speed), -max_speed)
            control_y = max(min(control_y, max_speed), -max_speed)

            if abs(control_x) < min_speed and control_x != 0:
                control_x = min_speed * np.sign(control_x)
            if abs(control_y) < min_speed and control_y != 0:
                control_y = min_speed * np.sign(control_y)

            # 放大控制量（可調整）
            control_x *= 2
            control_y *= 2

            # 設定並發布速度指令
            move_cmd = Twist()
            move_cmd.linear.x = control_y  # 前進後退
            move_cmd.angular.z = -control_x  # 左右旋轉（負號因應 ROS 左正右負的 convention）

            pub.publish(move_cmd)

    # 繪製藍色十字在目標座標處
    target_position = (int(target_x), int(target_y))
    cv2.drawMarker(frame, target_position, (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    # 顯示畫面
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()