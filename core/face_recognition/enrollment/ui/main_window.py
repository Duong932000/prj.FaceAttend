
import customtkinter as ctk

from core.face_recognition.enrollment.ui.webcam_panel import WebcamPanel
from core.face_recognition.enrollment.ui.progress_panel import ProgressPanel
from core.face_recognition.enrollment.ui.control_panel import ControlPanel


class MainWindow(ctk.CTk):

    def __init__(self, poses, start_callback):
        super().__init__()

        self.title("Face Dataset Enrollment")

        self.geometry("1500x900")

        self.grid_columnconfigure(0, weight=3)

        self.grid_columnconfigure(1, weight=1)

        self.grid_rowconfigure(0, weight=1)

        self.webcam_panel = WebcamPanel(self)

        self.webcam_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        right_panel = ctk.CTkFrame(self)

        right_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.control_panel = ControlPanel(right_panel, start_callback=start_callback)

        self.control_panel.pack(fill="x", pady=10)

        self.progress_panel = ProgressPanel(right_panel, poses=poses)

        self.progress_panel.pack(fill="both", expand=True, pady=10)