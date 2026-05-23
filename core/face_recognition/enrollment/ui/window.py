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

import sys
import cv2
import time
import customtkinter
from PIL import Image
from CTkMessagebox import CTkMessagebox
from core.face_recognition.enrollment.ui.configure import asset_resources
from core.face_recognition.utils.load_configs import load_enrollment_config
from core.face_recognition.enrollment.camera.camera_stream import CameraStream
from core.face_recognition.enrollment.processor.face_processor import FaceProcessor

# Custom appearance of GUI
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

# load config
enroll_camera_cfg, \
_, _, _, \
enroll_pose_cfg, \
enroll_pipeline_cfg, _, \
enroll_color_detection_cfg, \
_, \
enroll_app_mode_cfg = load_enrollment_config()

class FaceCollectionOptions(customtkinter.CTkFrame):
    def __init__(self, master, height=300, command=None, **kwargs):
        super().__init__(master, height=height, **kwargs)

        self.command = command

        self.required_options = enroll_pose_cfg["required_poses"]
        self.optional_options = enroll_pose_cfg["optional_poses"]

        self.checkbox_vars = {}
        self.checkbox_list = []

        self.select_all_var = customtkinter.BooleanVar(value=True)
        self.select_all_checkbox \
            = customtkinter.CTkCheckBox(self,
                                        text="Select all Required Options",
                                        variable=self.select_all_var,
                                        command=self.selectall,
                                        font=customtkinter.CTkFont("Century Gothic", size=16, slant="italic"))
        self.select_all_checkbox.pack(anchor="w",padx=10,pady=(5, 10))

        self.render_options()

    def render_options(self):

        sections = {
            " - Required Poses": self.required_options,
            " - Optional Poses": self.optional_options
        }

        for section_name, options in sections.items():
            if len(options) == 0:
                continue
            
            # section label
            section_label \
                = customtkinter.CTkLabel(self,
                                        text=section_name,
                                        font=customtkinter.CTkFont(size=15, weight="bold"))
            section_label.pack(anchor="w", padx=10, pady=(10, 5))

            # render checkbox
            for option in options:
                is_required = option in self.required_options

                option_var = customtkinter.BooleanVar(value=is_required)
                option_checkbox \
                    = customtkinter.CTkCheckBox(self,
                                                text=option.capitalize(),
                                                variable=option_var,
                                                command=self.specific_options)
                option_checkbox.pack(anchor="w", padx=40,pady=5)

                self.checkbox_vars[option] = option_var
                self.checkbox_list.append(option_checkbox)

    def selectall(self):

        state = self.select_all_var.get()

        for option in self.required_options:
            self.checkbox_vars[option].set(state)

        if self.command:
            self.command()

    def specific_options(self):

        required_states = [
            self.checkbox_vars[option].get()
            for option in self.required_options
        ]

        self.select_all_var.set(all(required_states))

        if self.command:
            self.command()

    def get_selected_opions(self):

        return [
            self.checkbox_list[i].cget("text")
            for i, var in enumerate(self.checkbox_vars)
            if var.get()
        ]

class EnrollmentPipeline(customtkinter.CTkFrame):
    def __init__(self,master, options=enroll_pipeline_cfg, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        self.options = options
        self.checkbox_vars = {}
        self.checkbox_widgets = {}

        self.render_options()

    def render_options(self):

        for _, option in enumerate(self.options):
            option_var = customtkinter.BooleanVar(value=True)
            option_checkbox = customtkinter.CTkCheckBox(self, text=f"↓ {option}", variable=option_var)
            option_checkbox.pack(anchor="w", padx=10, pady=5)
            self.checkbox_vars[option] = option_var
            self.checkbox_widgets[option] = option_checkbox

    def get_selected_options(self):

        return {
            option: var.get()
            for option, var in self.checkbox_vars.items()
        }

class EnrollmentProgressTextbox(customtkinter.CTkFrame):
    def __init__(self,master, textbox_width=300, textbox_height=150, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        self.textbox \
            = customtkinter.CTkTextbox(self,
                                       width=textbox_width,
                                       height=textbox_height,
                                       wrap="word",
                                       corner_radius=5,
                                       font=customtkinter.CTkFont(size=13))
        self.textbox.pack(fill="both", expand=True)
        self.textbox.configure(state="disabled")

    def append_log(self, message):

        self.textbox.configure(state="normal")
        self.textbox.insert("end", f"{message}\n")
        self.textbox.see("end")
        self.textbox.configure(state="disabled")

    def clear_log(self):

        self.textbox.configure(state="normal")
        self.textbox.delete("1.0", "end")
        self.textbox.configure(state="disabled")

class MainWindow(customtkinter.CTk):

    width_dashboard = 1300
    height_dashboard = 800

    def __init__(self):
        super().__init__()

        # app state
        self.app_mode = enroll_app_mode_cfg["idle"]

        # fps counter
        self.prev_frame_time = time.time()
        self.current_fps = 0

        # init UI
        self.GUI_InitialSetupResources_Displayer()

        # setup widgets for UI 
        self.GUI_WidgetsPanelSetup_Displayer()

        self.GUI_FaceProcessor_Displayer()

    # ---------------- INIT SETUP RESOURCE ---------------- #
    # ------------------------------------------------------#
    def GUI_InitialSetupResources_Displayer(self):
        
        # config for common resources of UI
        self.commonSetupResources()

        # config for images as an icon
        self.imageSetupResources()

    def commonSetupResources(self):

        # Window title
        self.title("FaceID Enrollment")
        self.geometry(f"{self.width_dashboard}x{self.height_dashboard}")
        self.resizable(True, True)

        # root grid
        self.grid_columnconfigure(0, weight=0)   # menu panel
        self.grid_columnconfigure(1, weight=3)   # camera panel
        self.grid_columnconfigure(2, weight=1)   # processing log panel
        self.grid_columnconfigure(3, weight=1)   # controll_ panel
        self.grid_rowconfigure(0, weight=1)

        # menu panel
        self.menu_panel = customtkinter.CTkFrame(self, width=220, corner_radius=20)
        self.menu_panel.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="ns")
        self.menu_panel.grid_rowconfigure(10, weight=1)

        # camera panel
        self.camera_panel = customtkinter.CTkFrame(self, corner_radius=20)
        self.camera_panel.grid(row=0, column=1, padx=5, pady=10, sticky="nsew")
        self.camera_panel.grid_rowconfigure(0, weight=1)
        self.camera_panel.grid_columnconfigure(0, weight=1)

        # processing log panel
        self.processing_panel = customtkinter.CTkFrame(self, corner_radius=20)
        self.processing_panel.grid(row=0, column=2, padx=(5, 10), pady=10, sticky="nsew")
        self.processing_panel.grid_rowconfigure(0, weight=1)
        self.processing_panel.grid_columnconfigure(0, weight=1)

        # processing panel
        self.control_panel = customtkinter.CTkFrame(self, width=400, corner_radius=20)
        self.control_panel.grid(row=0, column=3, padx=(5, 10), pady=10, sticky="nsew")
        self.control_panel.grid_rowconfigure(0, weight=1)
        self.control_panel.grid_columnconfigure(0, weight=1)

        self.protocol("WM_DELETE_WINDOW", self.closeApp)

    def imageSetupResources(self):

        # Image configuration
        self.registration_img = customtkinter.CTkImage(
            Image.open(asset_resources("registration.png")), size=(25, 25))

        self.advance_img = customtkinter.CTkImage(
            Image.open(asset_resources("advance.png")), size=(25, 25))

        self.logo_img = customtkinter.CTkImage(
            Image.open(asset_resources("logo.png")), size=(50, 50))

    def GUI_WidgetsPanelSetup_Displayer(self):

        # component 1: Menu Panel
        self.menuPanelConfigure()

        # component 2: Camera Panel
        self.cameraPanelConfigure()

        # component 3: Processing log Panel
        self.processingLogPanelConfigure()

        # component 4: Control Panel
        self.controlPanelConfigure()

        # Frame Selection
        self.frameSelection("Enrollment")

    def menuPanelConfigure(self):

        # logo
        self.logo_label \
            = customtkinter.CTkButton(self.menu_panel,
                                      text="FaceID\nEnrollment",
                                      command=self.enrollFrameEvent,
                                      corner_radius=10,
                                      height=60,
                                      anchor="w",
                                      state="disabled",
                                      fg_color="transparent",
                                      text_color=("gray10", "gray90"),
                                      hover_color=("gray70", "gray30"),
                                      font=customtkinter.CTkFont(size=22, weight="bold"),
                                      image=self.logo_img)
        self.logo_label.grid(row=0, column=0, padx=15, pady=15, sticky="ew")

        # Enrollment tab
        self.enrollment_button \
            = customtkinter.CTkButton(self.menu_panel,
                                      text=" Enrollment",
                                      image=self.registration_img,
                                      anchor="w",
                                      height=50,
                                      corner_radius=10,
                                      font=customtkinter.CTkFont(size=16, slant="italic"),
                                      command=self.enrollFrameEvent)
        self.enrollment_button.grid(row=1, column=0, padx=15, pady=2, sticky="ew")

        # Advance tab
        self.advance_button \
            = customtkinter.CTkButton(self.menu_panel,
                                      text=" Advance",
                                      image=self.advance_img,
                                      anchor="w",
                                      height=50,
                                      corner_radius=10,
                                      font=customtkinter.CTkFont(size=16, slant="italic"),
                                      command=self.advanceFrameEvent)
        self.advance_button.grid(row=2, column=0, padx=15, pady=2, sticky="ew")

        # system status frame
        self.system_status_frame = customtkinter.CTkFrame(self.menu_panel, corner_radius=15)
        self.system_status_frame.grid(row=11, column=0, padx=15, pady=20, sticky="ew")

        # system status title
        self.status_title \
            = customtkinter.CTkLabel(self.system_status_frame,
                                    text="System Status",
                                    font=customtkinter.CTkFont(size=16, weight="bold"))
        self.status_title.pack(anchor="w", padx=10, pady=(10, 5))

        # camera status
        self.camera_status \
            = customtkinter.CTkLabel(self.system_status_frame,
                                     text="- Camera Ready")
        self.camera_status.pack(anchor="w", padx=10)

        # engine status
        self.engine_status \
            = customtkinter.CTkLabel(self.system_status_frame,
                                     text="- Engine Loaded")
        self.engine_status.pack(anchor="w", padx=10)

        # storage status
        self.storage_status \
            = customtkinter.CTkLabel(self.system_status_frame,
                                     text="- Storage Ready")
        self.storage_status.pack(anchor="w", padx=10, pady=(0, 10))

    def cameraPanelConfigure(self):

        # camera frame
        self.camera_display_frame \
            = customtkinter.CTkFrame(self.camera_panel, corner_radius=20)
        self.camera_display_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        self.camera_display_frame.grid_rowconfigure(0, weight=1)
        self.camera_display_frame.grid_columnconfigure(0, weight=1)

        self.camera_label = customtkinter.CTkLabel(self.camera_display_frame, text="")
        self.camera_label.grid(row=0, column=0, sticky="nsew")

    def processingLogPanelConfigure(self):
        
        # process label
        self.enroll_progress_label \
            = customtkinter.CTkLabel(self.processing_panel,
                                     text="✓ Enrollment Progress",
                                     font=customtkinter.CTkFont(size=18, weight="bold"))
        self.enroll_progress_label.pack(anchor="w", padx=5, pady=(10, 5))

        # Clear log
        self.clear_log_button \
            = customtkinter.CTkButton(self.processing_panel,
                                      text="Clear log",
                                      command=self.clearProgressLog,
                                      font=customtkinter.CTkFont(size=14, slant="italic"))
        self.clear_log_button.pack(anchor="w", padx=5, pady=(5, 10))

        # Enrollment progress textbox
        self.processing_log_textbox = EnrollmentProgressTextbox(self.processing_panel)
        self.processing_log_textbox.pack(fill="both", expand=True, padx=10, pady=(10, 10))

    def controlPanelConfigure(self):

        # tabview configure
        self.control_tabview = customtkinter.CTkTabview(self.control_panel)
        self.control_tabview.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        self.control_tabview.add("Webcam")
        self.control_tabview.add("Upload")

        # Show Webcam tab
        self.webcam_tab = self.control_tabview.tab("Webcam")
        self.webcamTabConfigure()

        # Show Upload tab
        self.upload_tab = self.control_tabview.tab("Upload")
        self.uploadTabConfigure()

    def webcamTabConfigure(self):

        # ----- FRAME CONFIGURE ----- #
        # define scroll frame
        self.webcame_scroll_frame \
            = customtkinter.CTkScrollableFrame(self.webcam_tab, corner_radius=10)
        self.webcame_scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.detection_button_frame \
            = customtkinter.CTkFrame(self.webcame_scroll_frame, fg_color="transparent")
        self.detection_button_frame.pack(fill="x", padx=10, pady=(20, 10))
        self.detection_button_frame.grid_columnconfigure(0, weight=1)
        self.detection_button_frame.grid_columnconfigure(1, weight=1)

        # ----- FACE DETECTION ----- #
        # Start face detection
        self.start_detection_button \
            = customtkinter.CTkButton(self.detection_button_frame,
                                      text="Start Face Detection",
                                      command=self.startFaceDetection,
                                      height=35,
                                      font=customtkinter.CTkFont(size=11, weight="bold"))
        self.start_detection_button.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        # Stop face detection
        self.stop_detection_button \
            = customtkinter.CTkButton(self.detection_button_frame,
                                      text="Stop Face Detection",
                                      command=self.stopFaceDetection,
                                      height=35,
                                      fg_color="#8B0000",
                                      hover_color="#5A0000",
                                      font=customtkinter.CTkFont(size=11, weight="bold"))
        self.stop_detection_button.grid(row=0, column=1, padx=(5, 0), sticky="ew")


        # ----- USER INFORMATION ----- #
        # User information label
        self.userinfo_label \
            = customtkinter.CTkLabel(self.webcame_scroll_frame,
                                     text="✓ User Information:",
                                     font=customtkinter.CTkFont(size=18, weight="bold"))
        self.userinfo_label.pack(anchor="w", padx=10, pady=(15, 5))

        # Name entry
        self.name_entry \
            = customtkinter.CTkEntry(self.webcame_scroll_frame,
                                    placeholder_text="Enter name")
        self.name_entry.pack(fill="x", padx=10, pady=5)


        # ----- OUTPUT DIRECTORY ----- #
        # Dataset output path label
        self.dataset_output_label \
            = customtkinter.CTkLabel(self.webcame_scroll_frame,
                                     text="✓ Output Directory:",
                                     font=customtkinter.CTkFont(size=18, weight="bold"))
        self.dataset_output_label.pack(anchor="w", padx=10, pady=(15, 5))

        # Dataset output path entry
        self.dataset_output_entry \
            = customtkinter.CTkEntry(self.webcame_scroll_frame,
                                     placeholder_text="*default: ./output/face_recognition/datasets")
        self.dataset_output_entry.pack(fill="x", padx=10, pady=5)


        # ----- FACE COLLECTION OPTIONS ----- #
        # Face Collection Options
        self.face_collect_option_label \
            = customtkinter.CTkLabel(self.webcame_scroll_frame,
                                     text="✓ Face Collection Options",
                                     font=customtkinter.CTkFont( size=18, weight="bold"))
        self.face_collect_option_label.pack(anchor="w", padx=10, pady=(10, 5))

        # Show collection options
        self.collection_options \
            = FaceCollectionOptions(self.webcame_scroll_frame, width=250, height=370)
        self.collection_options.pack(fill="x", expand=False, padx=10, pady=2)
 

        # ----- ENROLLMENT PIPELINE OPTIONS ----- #
        # enrollment pipeline label
        self.pipeline_label \
            = customtkinter.CTkLabel(self.webcame_scroll_frame,
                                     text="✓ Enrollment Pipeline",
                                     font=customtkinter.CTkFont(size=18,weight="bold"))
        self.pipeline_label.pack(anchor="w", padx=10, pady=(10, 5))

        # pipeline options
        self.pipeline_options = EnrollmentPipeline(self.webcame_scroll_frame)
        self.pipeline_options.pack(fill="x", padx=10, pady=(0, 5))



        # ----- START ENROLLMENT ----- #
        self.start_enroll_button \
            = customtkinter.CTkButton(self.webcame_scroll_frame,
                                      text="Enroll New Face",
                                      command=self.startEnrollment,
                                      width=40,
                                      height=20,
                                      font=customtkinter.CTkFont(size=32))
        self.start_enroll_button.pack(fill="x", padx=10, pady=(25, 10))

    def uploadTabConfigure(self):

        self.upload_label \
            = customtkinter.CTkLabel(self.upload_tab,
                                     text="Upload Face Images",
                                     font=customtkinter.CTkFont(size=18, weight="bold"))
        self.upload_label.pack(anchor="w", padx=10, pady=(20, 20))

        self.upload_button \
            = customtkinter.CTkButton(self.upload_tab,text="Select Images")
        self.upload_button.pack(fill="x", padx=10, pady=5)

        self.upload_folder_button \
            = customtkinter.CTkButton(self.upload_tab, text="Select Folder")
        self.upload_folder_button.pack(fill="x", padx=10, pady=5)

        self.align_button \
            = customtkinter.CTkButton(self.upload_tab, text="Start Align Process", height=45)
        self.align_button.pack(fill="x", padx=10, pady=(20, 10))

    def frameSelection(self, frame_name):

        if frame_name == "Enrollment":
            self.enrollment_button.configure(fg_color=("gray75", "gray25"))
            self.advance_button.configure(fg_color="transparent")
        elif frame_name == "Advance":
            self.advance_button.configure(fg_color=("gray75", "gray25"))
            self.enrollment_button.configure(fg_color="transparent")

    def enrollFrameEvent(self):

        self.frameSelection("Enrollment")

    def advanceFrameEvent(self):

        self.frameSelection("Advance")

    def closeApp(self):

        msg_ExitSystem \
            = CTkMessagebox(master=self,
                            title="Exit",
                            message="Do you want to exit the FaceID Enrollment System",
                            icon="question",
                            option_1="Cancel",
                            option_2="Exit")
        if msg_ExitSystem.get() == "Exit":
            if hasattr(self, "face_processor"):
                self.face_processor.stop()

            if hasattr(self, "camera_stream"):
                self.camera_stream.stop()

            self.destroy()
            sys.exit()

    # ---------------- FACE PROCESSOR SETUP RESOURCE ---------------- #
    # ----------------------------------------------------------------#
    def GUI_FaceProcessor_Displayer(self):

        # camera init
        self.cameraInitialize()

        # face processor init
        self.facialProcessor()

    def cameraInitialize(self):
        
        # camera streaming
        self.camera_stream = CameraStream(enroll_camera_cfg)
        self.camera_stream.start()

        # update frame for camera
        self.cameraUpdateFrame()

    def facialProcessor(self):

        # face processing
        self.face_processor = FaceProcessor(self.camera_stream)
        self.face_processor.start()

    def cameraUpdateFrame(self):

        frame = self.camera_stream.get_latest_frame()
        result = self.face_processor.get_latest_result() if hasattr(self, "face_processor") else None

        if frame is not None:
            panel_width = self.camera_display_frame.winfo_width()
            panel_height = self.camera_display_frame.winfo_height()

            if panel_width > 10 and panel_height > 10:
                cv2.putText(frame,
                            f"Camera FPS: {self.camera_stream.capture_fps:.1f}",
                            (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            tuple(enroll_color_detection_cfg["bbox"]["green"]),
                            2,)

                if hasattr(self, "face_processor"):
                    cv2.putText(frame,
                                f"Inference FPS: {self.face_processor.inference_fps:.1f}",
                                (20, 65),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                tuple(enroll_color_detection_cfg["bbox"]["green"]),
                                2,)

                if self.app_mode in [enroll_app_mode_cfg["detection"],
                                     enroll_app_mode_cfg["enrollment"]] \
                                and result is not None and result["face_detected"]:
                    face = result["face"]
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    score = float(face.det_score)
                    yaw, pitch, _ = face.pose

                    box_color = tuple(enroll_color_detection_cfg["bbox"]["green"])
                    if self.app_mode == enroll_app_mode_cfg["enrollment"]:
                        if not result["stable"]:
                            box_color = tuple(enroll_color_detection_cfg["bbox"]["orange"])
                        if not result["pose_valid"]:
                            box_color = tuple(enroll_color_detection_cfg["bbox"]["red"])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(frame, f"Confidence: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                    cv2.putText(frame, f"Yaw:{yaw:.1f} Pitch:{pitch:.1f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                if self.app_mode == enroll_app_mode_cfg["enrollment"]:
                    self.drawEnrollmentGuide(frame)
                    cv2.putText(frame,
                                f"Target Pose: {self.face_processor.target_pose.upper()}",
                                (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                tuple(enroll_color_detection_cfg["bbox"]["green"]),
                                2,)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (panel_width, panel_height))
                pil_image = Image.fromarray(frame_resized)
                ctk_image = customtkinter.CTkImage(light_image=pil_image, dark_image=pil_image, size=(panel_width, panel_height))
                self.camera_label.configure(image=ctk_image)
                self.camera_label.image = ctk_image

        self.after(33, self.cameraUpdateFrame)

    def startFaceDetection(self):

        self.app_mode = "detection"

        self.face_processor.enable_detection()

        self.processing_log_textbox.append_log("[INFO] Face detection started")

        self.faceDetectionStatus()

    def stopFaceDetection(self):

        self.app_mode = enroll_app_mode_cfg["idle"]
        self.face_processor.disable_detection()
        self.camera_status.configure(text="- Camera Ready")
        self.engine_status.configure(text="- Engine Ready")
        self.storage_status.configure(text="- Storage Ready")
        self.processing_log_textbox.append_log("[INFO] Face detection stopped")

    def startEnrollment(self):

        self.app_mode = enroll_app_mode_cfg["enrollment"]

        self.face_processor.enable_detection()

        self.processing_log_textbox.append_log("[INFO] Enrollment started")

        self.faceDetectionStatus()

    def faceDetectionStatus(self):

        result = self.face_processor.get_latest_result()
        if result is not None:
            if result["face_detected"]:
                self.camera_status.configure(text="- Face Detected")
                # detection mode
                if self.app_mode == enroll_app_mode_cfg["detection"]:
                    self.engine_status.configure(text="- Detection Running")
                    self.storage_status.configure(text="- Monitoring Pose")

                # enrollment mode
                elif self.app_mode == enroll_app_mode_cfg["enrollment"]:
                    if result["stable"]:
                        self.engine_status.configure(text="- Stable Face")
                    else:
                        self.engine_status.configure(text="- Unstable Face")
                    if result["pose_valid"]:
                        self.storage_status.configure(text="- Pose Valid")
                    else:
                        self.storage_status.configure(text="- Invalid Pose")
            # no face
            else:
                self.camera_status.configure(text="- No Face Detected")
                self.engine_status.configure(text="- Waiting")
                self.storage_status.configure(text="- Waiting")
        
        self.after(100, self.faceDetectionStatus)

    def drawEnrollmentGuide(self, frame):

        height, width, _ = frame.shape

        center_x = width // 2

        center_y = height // 2

        axes = (int(width * 0.18),int(height * 0.32))

        cv2.ellipse(frame, (center_x, center_y), axes, 0, 0,360, (0, 255, 0), 3)

        cv2.putText(frame, "Place face inside oval",
                    (
                        center_x - 140,
                        center_y + axes[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

    def clearProgressLog(self):

        self.processing_log_textbox.clear_log()
