
import customtkinter as ctk

class ProgressPanel(ctk.CTkFrame):

    def __init__(self, master, poses):
        super().__init__(master)

        self.pose_labels = {}

        title = ctk.CTkLabel(self, text="Dataset Coverage", font=("Arial", 18, "bold"))

        title.pack(pady=10)

        for pose in poses:
            label = ctk.CTkLabel(self, text=f"{pose}: 0", anchor="w")
            label.pack(fill="x", padx=10, pady=5)
            self.pose_labels[pose] = label

    def update_progress(self, pose,count, total):
        self.pose_labels[pose].configure(text=f"{pose}: {count}/{total}")