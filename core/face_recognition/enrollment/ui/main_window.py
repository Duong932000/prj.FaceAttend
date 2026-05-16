#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      ui.py
# .:ccccccccccccc;KMMc;cc;xMMc;ccccccc:.    Description  :      
# ,cccccccccccccc;MMM.;cc;;WW:;cccccccc,    --------------
# :cccccccccccccc;MMM.;cccccccccccccccc:
# :ccccccc;oxOOOo;MMM000k.;cccccccccccc:
# cccccc;0MMKxdd:;MMMkddc.;cccccccccccc;
# ccccc;XMO';cccc;MMM.;cccccccccccccccc'
# ccccc;MMo;ccccc;MMW.;ccccccccccccccc;
# ccccc;0MNc.ccc.xMMd;ccccccccccccccc;
# cccccc;dNMWXXXWM0:;cccccccccccccc:,
# cccccccc;.:odl:.;cccccccccccccc:,.
# ccccccccccccccccccccccccccccc:'.
# :ccccccccccccccccccccccc:;,..
#  ':cccccccccccccccc::;,.
#########################################################


import cv2
import customtkinter
from PIL import Image
from PIL import ImageTk
from core.face_recognition.enrollment.ui.addons import exit_ui

# Custom appearance of GUI
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

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

class MenuPanel(customtkinter.CTkFrame):
    def __init__(self, master, start_detection_callback, stop_detection_callback, start_enrollment_callback, **kwargs):

        super().__init__(master, *kwargs)

        self.start_detection_callback = start_detection_callback
        self.stop_detection_callback = stop_detection_callback
        self.start_enrollment_callback = start_enrollment_callback

        # title
        title = customtkinter.CTkLabel(self,
                                       text="Face Enrollment",
                                       font=customtkinter.CTkFont("Century Gothic",
                                                                size=30,
                                                                weight="bold"))
        title.pack(pady=(30, 25))
        
        # Name entry
        self.name_entry = customtkinter.CTkEntry(self,
                                                 placeholder_text="Enter Person Name",
                                                 height=45)
        self.name_entry.pack(fill="x",padx=20, pady=10)

        # Start detection button: open webcam to face detect
        self.start_detection_button \
            = customtkinter.CTkButton(self,
                                     text="Start Face Detection",
                                     height=45,
                                     font=("Arial", 18),
                                     command=self.start_detection_callback)
        self.start_detection_button.pack(fill="x", padx=20, pady=10)

        # Stop detection button: stop threading
        self.stop_detection_button \
            = customtkinter.CTkButton(self,
                                      text="Stop Face Detection",
                                      height=45,
                                      font=("Arial", 18),
                                      command=self.stop_detection_callback)
        self.stop_detection_button.pack(fill="x", padx=20, pady=10)

        # start process to collect face
        self.start_enrollment_button \
            = customtkinter.CTkButton(self,
                                      text="Start Enrollment",
                                      height=45,
                                      font=("Arial", 18),
                                      command=self.on_start_enrollment)
        self.start_enrollment_button.pack(fill="x", padx=20, pady=10)

        # label of pose
        self.pose_label \
            = customtkinter.CTkLabel(self, text="POSE: FRONT", font=("Arial", 20))
        self.pose_label.pack(pady=(50, 10))

        self.status_label \
            = customtkinter.CTkLabel(self, text="DETECTION OFF", font=("Arial", 22, "bold"))
        self.status_label.pack(pady=10)

    def on_start_enrollment(self):

        person_name = self.name_entry.get().strip()
        if len(person_name) == 0:
            return

        self.start_enrollment_callback(person_name)

class MainWindow(customtkinter.CTk):

    width_dashboard = 1300
    height_dashboard = 700

    def __init__(self,
                 start_detection_callback,
                 stop_detection_callback,
                 start_enrollment_callback):

        super().__init__()

        self.start_detection_callback = start_detection_callback,
        self.stop_detection_callback = stop_detection_callback,
        self.start_enrollment_callback = start_enrollment_callback

        self.GUI_InitialSetupResources_Displayer()
        self.GUI_ControlPanelSetup_Displayer()

    def GUI_InitialSetupResources_Displayer(self):

        self.title("FaceID Enrollment")

        self.geometry(f"{self.width_dashboard}x{self.height_dashboard}")
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.resizable(True, True)

        # Create 'WM_DELETE_WINDOW' when close button
        self.protocol("WM_DELETE_WINDOW", lambda: exit_ui(self))

    def GUI_ControlPanelSetup_Displayer(self):
        
        # webcam panel configure
        self.webcam_panel = WebcamPanel(self)
        self.webcam_panel.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

        # Menu panel configure
        self.menu_panel \
            = MenuPanel(self,
                         start_detection_callback=self.start_detection_callback,
                         stop_detection_callback=self.stop_detection_callback,
                         start_enrollment_callback=self.start_enrollment_callback)

        self.menu_panel.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)