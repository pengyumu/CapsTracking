import cv2
import numpy as np

class KalmanFilter(object):
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        # Define sampling time
        self.dt = dt

        # Define the control input variables
        self.u = np.matrix([[u_x], [u_y]])

        # Initial State
        self.x = np.matrix([[0], [0], [0], [0]])

        # Define the State Transition Matrix A
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        # Define the Control Input Matrix B
        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0, (self.dt**2)/2],
                            [self.dt, 0],
                            [0, self.dt]])

        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        # Initial Process Noise Covariance
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2

        # Initial Measurement Noise Covariance
        self.R = np.matrix([[x_std_meas**2, 0],
                            [0, y_std_meas**2]]) * 0.1

        # Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])

    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0:2]

    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))
        I = np.eye(self.H.shape[1])
        self.P = (I - (K * self.H)) * self.P
        return self.x[0:2]

def detect_pen_caps(frame, color_name, color_range):
    height, width = frame.shape[:2]
    roi_height = int(height * 0.6)
    frame = frame[:roi_height, :]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if color_name == 'red':
        mask1 = cv2.inRange(hsv, color_ranges['red'][0], color_ranges['red'][1])
        mask2 = cv2.inRange(hsv, color_ranges['red2'][0], color_ranges['red2'][1])
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, color_range[0], color_range[1])

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pen_caps = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:
            x, y, w, h = cv2.boundingRect(cnt)
            pen_caps.append((x + w // 2, y + h // 2))
    return pen_caps

def track_and_visualize(video_path, output_path, color_ranges):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    cv2.namedWindow("Pen Cap Tracking", cv2.WINDOW_NORMAL)

    kf_filters = {
        'black': KalmanFilter(dt=0.1, u_x=0.5, u_y=0.5, std_acc=5.0, x_std_meas=0.1, y_std_meas=0.1),
        'blue': KalmanFilter(dt=0.1, u_x=0.5, u_y=0.5, std_acc=5.0, x_std_meas=0.1, y_std_meas=0.1),
        'red': KalmanFilter(dt=0.1, u_x=0.5, u_y=0.5, std_acc=5.0, x_std_meas=0.1, y_std_meas=0.1),
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for color, range_ in color_ranges.items():
            
            positions = detect_pen_caps(frame, color, range_)
            if color == 'red2':
                color = 'red'

            visual_color = visualization_colors.get(color, (0, 255, 0))  # 获取可视化颜色，未定义颜色使用默认绿色
            KF = kf_filters[color]
            for x, y in positions:
                cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), -1)
                cv2.putText(frame, color, (int(x), int(y) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                predict_x, predict_y = map(int, KF.predict())
                cv2.rectangle(frame, (predict_x - 15, predict_y - 15), 
                                     (predict_x + 15, predict_y + 15), visual_color, 2)
                cv2.putText(frame, f"Predicted {color} Position", (predict_x + 15, predict_y), 0, 0.5, visual_color, 2)


                update_x, update_y = map(int, KF.update(np.matrix([[x], [y]])))
                cv2.rectangle(frame, (update_x - 15, update_y - 15), 
                                     (update_x + 15, update_y + 15), visual_color, 2)
        
        out.write(frame)
        cv2.imshow("Pen Cap Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # 释放写入器
    cv2.destroyAllWindows()

color_ranges = {
    'black': ((0, 0, 0), (180, 255, 50)),
    'blue': ((100, 100, 100), (140, 255, 255)),
    'red': ((0, 70, 50), (10, 255, 255)),
    'red2': ((170, 70, 50), (180, 255, 255))
}

visualization_colors = {
    'black': (255, 255, 255),
    'blue': (0, 255, 255), 
    'red': (255, 165, 0)
}

video_path = './pens.mp4'
output_path = './pens_tracked.mp4'
track_and_visualize(video_path, output_path, color_ranges)
