
import cv2
import threading

class CameraStream:
    def __init__(self, camera_id=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(camera_id)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIHGT, height)

        self.running = False
        self.frame = None
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            with self.lock:
                self.frame = frame

    def get_latest_frame(self):
        
        with self.lock():
            if self.frame is None:
                return None
            
            return self.frame.copy()
        
    def stop(self):

        self.running = False
        self.cap.release()


