import VolumeControl as vc
import cv2


hand_volume = vc.VolumeControl()

while True:
        frame = hand_volume.process_frame()
        if frame is None:
            break  # Stop if no frame is captured

        cv2.imshow("Volume Control", frame)

        if cv2.waitKey(1) & 0xFF == 27: # Esc button
            break

hand_volume.release()