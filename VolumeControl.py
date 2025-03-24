import cv2
import mediapipe as mp
import time
import numpy as np
import HandTrackingModule as htm
import math
import threading
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class VolumeControl:
    def __init__(self, camera_index=0, width=640, height=480,
                 detection_confidence=0.7, text_color=(255, 255, 255),
                 fps_display=True, use_multithreading=True, smoothness=5):
        """
        Initialize the Hand Tracking and Volume Control system.

        :param camera_index: Index of the camera to use.
        :param width: Camera frame width.
        :param height: Camera frame height.
        :param detection_confidence: Confidence threshold for hand detection.
        :param text_color: Color of on-screen text (BGR format).
        :param fps_display: Whether to display FPS on screen.
        :param use_multithreading: Use threading for smoother UI updates.
        :param smoothness: Controls how smoothly volume changes (higher = smoother).
        """

        # Video Capture
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(3, width)
        self.cap.set(4, height)

        # Hand Detector
        self.detector = htm.handDetector(detectionCon=detection_confidence)

        # Get system volume control
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))

        # Volume Variables
        self.vol = -15
        self.volbar = 400
        self.volper = 0
        self.pTime = time.perf_counter()

        # User preferences
        self.text_color = text_color
        self.fps_display = fps_display
        self.use_multithreading = use_multithreading
        self.smoothness = smoothness
        self.prev_vol = self.vol  # Used for smoothing

    def process_frame(self):
        """Processes a single frame, detects hands, and updates UI & volume."""
        success, img = self.cap.read()
        if not success:
            return None  # No frame captured

        img = cv2.flip(img, 1)
        img = self.detector.findHands(img)
        landmarkCoordinates = self.detector.findPosition(img, draw=True, id=(4, 8))

        if landmarkCoordinates:
            x1, y1 = landmarkCoordinates[0][4][1], landmarkCoordinates[0][4][2]
            x2, y2 = landmarkCoordinates[0][8][1], landmarkCoordinates[0][8][2]
            midx, midy = (x1 + x2) // 2, (y1 + y2) // 2
            length = math.hypot(x2 - x1, y2 - y1)

            if self.use_multithreading:
                threading.Thread(target=self.update_volume, args=(length,)).start()
            else:
                self.update_volume(length)

            self.draw_ui(img, x1, y1, x2, y2, midx, midy, length)

        # Calculate FPS
        if self.fps_display:
            cTime = time.perf_counter()
            fps = 1 / (cTime - self.pTime)
            self.pTime = cTime
            cv2.putText(img, f"FPS: {int(fps)}", (40, 70), cv2.FONT_HERSHEY_PLAIN, 1, self.text_color, 2)

        return img  # Return the processed frame

    def update_volume(self, length):
        """Update system volume based on hand distance with smoothing."""
        target_vol = np.interp(length, [50, 250], [-63.5, 0])
        self.vol = (self.prev_vol * (self.smoothness - 1) + target_vol) / self.smoothness
        self.prev_vol = self.vol  # Store for next update

        self.volbar = np.interp(length, [50, 250], [400, 150])
        self.volper = np.interp(length, [50, 250], [0, 100])

        self.volume.SetMasterVolumeLevel(self.vol, None)

    def draw_ui(self, img, x1, y1, x2, y2, midx, midy, length):
        """Draws UI elements (lines, volume bars, indicators)."""
        cv2.line(img, (x1, y1), (x2, y2), (30, 144, 255), 2, cv2.LINE_AA)  # Dodger Blue line

        # Dynamic color based on volume
        if self.vol >= -5:
            color = (0, 255, 0)  # Green
        elif self.vol <= -60:
            color = (255, 0, 0)  # Red
        else:
            color = (255, 165, 0)  # Orange

        overlay = img.copy()
        cv2.circle(overlay, (midx, midy), 10, color, -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)  # Transparency effect

        # Volume Bar
        cv2.rectangle(img, (50, 150), (85, 400), (30, 144, 255), 2, cv2.LINE_AA)  # Dodger Blue border
        cv2.rectangle(img, (50, int(self.volbar)), (85, 400), (30, 144, 255), cv2.FILLED)

        # Circular Progress for Volume
        cv2.ellipse(img, (320, 415), (50, 50), 0, 0, int(self.volper * 3.6), (30, 144, 255), 3, cv2.LINE_AA)

        # Display Volume %
        cv2.putText(img, f"{int(self.volper)} %", (290, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def release(self):
        """Releases the camera and closes all windows."""
        self.cap.release()
        cv2.destroyAllWindows()
