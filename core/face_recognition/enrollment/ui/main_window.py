
import customtkinter
from core.face_recognition.enrollment.ui.right_panel import RightPanel
from core.face_recognition.enrollment.ui.webcam_panel import WebcamPanel

class MainWindow(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Face Enrollment")

        self.geometry("1600x900")

        customtkinter.set_appearance_mode("dark")
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.webcam_panel = WebcamPanel(self)
        self.webcam_panel.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

        self.right_panel = RightPanel(self)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)