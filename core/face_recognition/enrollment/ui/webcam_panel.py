
import cv2
import customtkinter
from PIL import Image
from PIL import ImageTk

class WebcamPanel(customtkinter.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        self.label = customtkinter.CTkLabel(self, text="")
        self.label.pack(fill="both", expand=True)

    def update_frame(self, frame):

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(frame_rgb)
        image = image.resize((960, 720))

        photo = ImageTk.PhotoImage(image=image)

        self.label.configure(image=photo)
        self.label.image = photo