import customtkinter


class RightPanel(customtkinter.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        title = customtkinter.CTkLabel(self, text="Face Enrollment", font=("Arial", 28, "bold"))

        title.pack(pady=30)

        self.pose_label = customtkinter.CTkLabel(self, text="POSE: FRONT", font=("Arial", 20))
        self.pose_label.pack(pady=10)

        self.status_label = customtkinter.CTkLabel(self, text="WAITING", font=("Arial", 20))
        self.status_label.pack(pady=10)

        self.progress_bar = customtkinter.CTkProgressBar(self)

        self.progress_bar.pack(fill="x", padx=20, pady=20)
        self.progress_bar.set(0)

        self.upload_button = customtkinter.CTkButton(self, text="Upload Images")
        self.upload_button.pack(fill="x", padx=20, pady=10)