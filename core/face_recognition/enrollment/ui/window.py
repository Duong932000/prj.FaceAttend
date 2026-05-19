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
from core.face_recognition.enrollment.ui.addons import exit_ui
from core.face_recognition.enrollment.ui.configure import asset_resources
from core.face_recognition.enrollment.camera.camera_stream import CameraStream

# Custom appearance of GUI
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

class FaceCollectionOptions(customtkinter.CTkScrollableFrame):

    DEFAULT_OPTIONS = ["Front", "Left", "Right", "Up", "Down", "Mask", "Glasses"]

    def __init__(self, master, height=300, options=DEFAULT_OPTIONS, command=None, **kwargs):
        super().__init__(master, height=300, orientation="vertical", **kwargs)

        self.options = options
        self.command = command

        self.checkbox_vars = []
        self.checkbox_list = []

        self.select_all_var = customtkinter.BooleanVar(value=True)

        self.select_all_checkbox \
            = customtkinter.CTkCheckBox(self,
                                        text="Select All",
                                        variable=self.select_all_var,
                                        command=self.collection_SelectAllOptions_Handle,
                                        font=customtkinter.CTkFont(
                                            "Century Gothic",
                                            size=16,
                                            slant="italic"
                                        ))
        self.select_all_checkbox.pack(anchor="w",padx=10,pady=(5, 10))

        self.collection_RenderOptions_Handle()

        self.after(1, self.collection_SelectAllOptions_Handle)

    def collection_RenderOptions_Handle(self):

        # clear old checkboxes
        for checkbox in self.checkbox_list:
            checkbox.destroy()

        self.checkbox_list.clear()
        self.checkbox_vars.clear()

        for index, option in enumerate(self.options):
            option_var = customtkinter.BooleanVar(value=False)
            option_checkbox \
                = customtkinter.CTkCheckBox(self,
                                            text=option,
                                            variable=option_var,
                                            command=self.collection_SpecificOption_Handle)
            option_checkbox.pack(anchor="w", padx=40, pady=5)

            self.checkbox_vars.append(option_var)
            self.checkbox_list.append(option_checkbox)

    def collection_SelectAllOptions_Handle(self):

        state = self.select_all_var.get()

        for var in self.checkbox_vars:
            var.set(state)

        if self.command:
            self.command()

    def collection_SpecificOption_Handle(self):

        # if any unchecked -> disable select all
        if not all(var.get() for var in self.checkbox_vars):
            self.select_all_var.set(False)

        # if all checked -> enable select all
        elif all(var.get() for var in self.checkbox_vars):
            self.select_all_var.set(True)

        if self.command:
            self.command()

    def collection_GetSelectedOptions_Handle(self):

        return [
            self.checkbox_list[i].cget("text")
            for i, var in enumerate(self.checkbox_vars)
            if var.get()
        ]

class EnrollmentPipeline(customtkinter.CTkFrame):

    DEFAULT_PIPELINE_OPTIONS = [
        "Align Face",
        "Generate Embedding",
        "Build FAISS Index",
        "Generate dataset_report.json"
    ]

    def __init__(self,master, options=DEFAULT_PIPELINE_OPTIONS, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        self.options = options
        self.checkbox_vars = {}
        self.checkbox_widgets = {}
        self.pipeline_RenderOptions_Handle()

    def pipeline_RenderOptions_Handle(self):

        for index, option in enumerate(self.options):
            option_var = customtkinter.BooleanVar(value=True)
            option_checkbox = customtkinter.CTkCheckBox(self, text=option, variable=option_var)
            option_checkbox.pack(anchor="w", padx=10, pady=5)
            self.checkbox_vars[option] = option_var
            self.checkbox_widgets[option] = option_checkbox

    def pipeline_GetSelectedOptions_Handle(self):

        return {
            option: var.get()
            for option, var in self.checkbox_vars.items()
        }

class EnrollmentProgressTextbox(customtkinter.CTkFrame):

    def __init__(self,
                 master,
                 textbox_width=300,
                 textbox_height=150,
                 **kwargs):

        super().__init__(
            master,
            fg_color="transparent",
            **kwargs
        )

        self.textbox \
            = customtkinter.CTkTextbox(
                self,
                width=textbox_width,
                height=textbox_height,
                wrap="word",
                corner_radius=5,
                font=customtkinter.CTkFont(size=13)
            )

        self.textbox.pack(
            fill="both",
            expand=True
        )

        self.textbox.configure(state="disabled")

    def progress_AppendLog_Handle(self, message):

        self.textbox.configure(state="normal")

        self.textbox.insert(
            "end",
            f"{message}\n"
        )

        self.textbox.see("end")

        self.textbox.configure(state="disabled")

    def progress_ClearLog_Handle(self):

        self.textbox.configure(state="normal")

        self.textbox.delete("1.0", "end")

        self.textbox.configure(state="disabled")

class MainWindow(customtkinter.CTk):

    width_dashboard = 1300
    height_dashboard = 800

    def __init__(self):
        super().__init__()

        # init UI
        self.GUI_InitialSetupResources_Displayer()

        # setup widgets for UI 
        self.GUI_WidgetsPanelSetup_Displayer()

        self.GUI_ControlPanelProcess_Displayer()

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
        self.controll_panel = customtkinter.CTkFrame(self, width=400, corner_radius=20)
        self.controll_panel.grid(row=0, column=3, padx=(5, 10), pady=10, sticky="nsew")
        self.controll_panel.grid_rowconfigure(0, weight=1)
        self.controll_panel.grid_columnconfigure(0, weight=1)

        self.protocol("WM_DELETE_WINDOW",self.closeApplication)

    def imageSetupResources(self):
        
        # image configure
        self.imageConfigure()

        # icon as an image
        self.assetsCompress()

    def imageConfigure(self):

        self.registration_img = customtkinter.CTkImage(
            Image.open(asset_resources("registration.png")), size=(25, 25))

        self.advance_img = customtkinter.CTkImage(
            Image.open(asset_resources("advance.png")), size=(25, 25))

        self.logo_img = customtkinter.CTkImage(
            Image.open(asset_resources("logo.png")), size=(50, 50))

    def assetsCompress(self):

        self.assets_compress = {
            "registration": self.registration_img,
            "advance": self.advance_img,
        }

    def GUI_WidgetsPanelSetup_Displayer(self):

        # component 1: Menu Panel
        self.menuPanelConfigure()

        # component 2: Camera Panel
        self.cameraPanelConfigure()

        # component 3: Processing log Panel
        self.processingLogPanelConfigure()

        # component 4: Control Panel
        self.ControlPanelConfigure()

        # Frame Selection
        self.frameSelection("Enrollment")

    def menuPanelConfigure(self):

        # logo
        self.logo_label \
            = customtkinter.CTkButton(self.menu_panel,
                                      text="FaceID\nEnrollment",
                                      command=self.enrollmentFrameEvent_Observer,
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
                                      command=self.enrollmentFrameEvent_Observer)
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
                                      command=self.advanceFrameEvent_Observer)
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
        self.progress_label \
            = customtkinter.CTkLabel(self.processing_panel,
                                     text="✓ Enrollment Progress",
                                     font=customtkinter.CTkFont(size=18, weight="bold"))
        self.progress_label.pack(anchor="w", padx=5, pady=(10, 5))

        # Enrollment progress textbox
        self.processing_log_textbox = EnrollmentProgressTextbox(self.processing_panel)
        self.processing_log_textbox.pack(fill="both", expand=True, padx=10, pady=(10, 10))

    def ControlPanelConfigure(self):

        # tabview configure
        self.control_tabview = customtkinter.CTkTabview(self.controll_panel)
        self.control_tabview.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        self.control_tabview.add("Webcam")
        self.control_tabview.add("Upload")

        # Show Webcam tab
        self.webcam_tab = self.control_tabview.tab("Webcam")
        self.webcamTabConfigure()

        # show Upload tab
        self.upload_tab = self.control_tabview.tab("Upload")
        self.uploadTabConfigure()

    def webcamTabConfigure(self):
        
        # ----- USER INFORMATION ----- #
        # User information label
        self.userinfo_label \
            = customtkinter.CTkLabel(self.webcam_tab,
                                     text="✓ User Information:",
                                     font=customtkinter.CTkFont(size=18, weight="bold"))
        self.userinfo_label.pack(anchor="w", padx=10, pady=(15, 10))

        # Name entry
        self.name_entry \
            = customtkinter.CTkEntry(self.webcam_tab,
                                    placeholder_text="Enter name")
        self.name_entry.pack(fill="x", padx=10, pady=5)


        # ----- OUTPUT DIRECTORY ----- #
        # Dataset output path label
        self.dataset_output_label \
            = customtkinter.CTkLabel(self.webcam_tab,
                                     text="✓ Output Directory:",
                                     font=customtkinter.CTkFont(size=18, weight="bold"))
        self.dataset_output_label.pack(anchor="w", padx=10, pady=(15, 10))

        # Dataset output path entry
        self.dataset_output_entry \
            = customtkinter.CTkEntry(self.webcam_tab,
                                     placeholder_text="*default: ./output/face_recognition/datasets")
        self.dataset_output_entry.pack(fill="x", padx=10, pady=5)


        # ----- FACE COLLECTION OPTIONS ----- #
        # Face Collection Options
        self.option_label \
            = customtkinter.CTkLabel(self.webcam_tab,
                                     text="✓ Face Collection Options",
                                     font=customtkinter.CTkFont( size=18, weight="bold"))
        self.option_label.pack(anchor="w", padx=10, pady=(10, 5))

        # Show collection options
        self.collection_options \
            = FaceCollectionOptions(self.webcam_tab, width=250, height=140)
        self.collection_options.pack(fill="x", expand=False, padx=10, pady=5)
 

        # ----- ENROLLMENT PIPELINE OPTIONS ----- #
        # enrollment pipeline label
        self.pipeline_label \
            = customtkinter.CTkLabel(self.webcam_tab,
                                     text="✓ Enrollment Pipeline",
                                     font=customtkinter.CTkFont(size=18,weight="bold"))
        self.pipeline_label.pack(anchor="w", padx=10, pady=(15, 5))

        # pipeline options
        self.pipeline_options = EnrollmentPipeline(self.webcam_tab)
        self.pipeline_options.pack(fill="x", padx=10, pady=(0, 10))


        # ----- START ENROLLMENT ----- #
        # Start Enrollment button
        self.start_button \
            = customtkinter.CTkButton(self.webcam_tab,
                                      text="Start Enrollment",
                                      command=self.runningEnrollment,
                                      width=40,
                                      height=20,
                                      font=customtkinter.CTkFont(size=32))
        self.start_button.pack(fill="x", padx=10, pady=(25, 10))

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

    def enrollmentFrameEvent_Observer(self):

        self.frameSelection("Enrollment")

    def advanceFrameEvent_Observer(self):

        self.frameSelection("Advance")

    def GUI_ControlPanelProcess_Displayer(self):

        self.cameraInitialize()

    def cameraInitialize(self):

        camera_cfg = {
            "camera_id": 0,
            "width": 960,
            "height": 540
        }

        self.camera_stream = CameraStream(camera_cfg)

        self.camera_stream.start()

        self.camera_UpdateFrame_Handle()

    def camera_UpdateFrame_Handle(self):

        frame = self.camera_stream.get_latest_frame()

        if frame is not None:

            # ==========================================
            # GET CURRENT CAMERA PANEL SIZE
            # ==========================================
            panel_width = self.camera_display_frame.winfo_width()
            panel_height = self.camera_display_frame.winfo_height()

            # avoid startup size issue
            if panel_width > 10 and panel_height > 10:

                # ======================================
                # CONVERT BGR -> RGB
                # ======================================
                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # ======================================
                # RESIZE TO FIT PANEL
                # ======================================
                frame_resized = cv2.resize(
                    frame_rgb,
                    (panel_width, panel_height)
                )

                # ======================================
                # PIL IMAGE
                # ======================================
                pil_image = Image.fromarray(frame_resized)

                # ======================================
                # CTK IMAGE
                # ======================================
                ctk_image = customtkinter.CTkImage(
                    light_image=pil_image,
                    dark_image=pil_image,
                    size=(panel_width, panel_height)
                )

                self.camera_label.configure(
                    image=ctk_image
                )

                self.camera_label.image = ctk_image

        self.after(33, self.camera_UpdateFrame_Handle)

    def closeApplication(self):

        if hasattr(self, "camera_stream"):
            self.camera_stream.stop()

        self.destroy()

    def runningEnrollment(self):

        pass
