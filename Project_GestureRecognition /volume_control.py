import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class VolumeGestureController:
    def __init__(self, cam_width=640, cam_height=480):
        self.wCam = cam_width
        self.hCam = cam_height
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, self.wCam)
        self.cam.set(4, self.hCam)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Volume setup
        self.volume = self._initialize_volume_interface()
        vol_range = self.volume.GetVolumeRange()
        self.minVol = vol_range[0]
        self.maxVol = vol_range[1]

    def _initialize_volume_interface(self):
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))

    def _get_landmark_positions(self, hand_landmarks, image):
        lm_list = []
        for id, lm in enumerate(hand_landmarks.landmark):
            h, w, _ = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append((id, cx, cy))
        return lm_list

    def _draw_hand_landmarks(self, image, hand_landmarks):
        self.mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )

    def run(self):
        while self.cam.isOpened():
            success, image = self.cam.read()
            if not success:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self._draw_hand_landmarks(image, hand_landmarks)
                    lm_list = self._get_landmark_positions(hand_landmarks, image)

                    if lm_list:
                        x1, y1 = lm_list[4][1], lm_list[4][2]  # Thumb tip
                        x2, y2 = lm_list[8][1], lm_list[8][2]  # Index finger tip

                        # Draw gesture guide
                        cv2.circle(image, (x1, y1), 10, (255, 255, 255), cv2.FILLED)
                        cv2.circle(image, (x2, y2), 10, (255, 255, 255), cv2.FILLED)
                        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        length = math.hypot(x2 - x1, y2 - y1)
                        if length < 50:
                            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                        # Convert length to volume
                        vol = np.interp(length, [50, 220], [self.minVol, self.maxVol])
                        vol_bar = np.interp(length, [50, 220], [400, 150])
                        vol_per = np.interp(length, [50, 220], [0, 100])

                        self.volume.SetMasterVolumeLevel(vol, None)

                        # Draw volume bar and percentage
                        cv2.rectangle(image, (50, 150), (85, 400), (255, 0, 0), 3)
                        cv2.rectangle(image, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
                        cv2.putText(image, f'{int(vol_per)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

            cv2.imshow('Volume Control via Gesture', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    controller = VolumeGestureController()
    controller.run()
