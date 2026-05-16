import cv2
import customtkinter as ctk

from PIL import Image
from PIL import ImageTk

class WebcamPanel(ctk.CTkFrame):
    def __init__(self, master, width=960, height=720):
        super().__init__(master)
        self.width = width
        self.height = height
        self.label = ctk.CTkLabel(self, text="")
        self.label.pack()

    def update_frame(self, frame):

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = image.resize((self.width, self.height))

        photo = ImageTk.PhotoImage(image=image)

        self.label.configure(image=photo)

        self.label.image = photo