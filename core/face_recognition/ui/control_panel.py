
import customtkinter as ctk

class ControlPanel(ctk.CTkFrame):

    def __init__(self, master, start_callback):
        super().__init__(master)
        self.start_callback = start_callback

        title = ctk.CTkLabel(self, text="Enrollment", font=("Arial", 18, "bold"))

        title.pack(pady=10)

        self.name_entry = ctk.CTkEntry(self, placeholder_text="Enter person name", width=240)

        self.name_entry.pack(pady=10)

        self.status_label = ctk.CTkLabel(self, text="Ready")

        self.status_label.pack(pady=10)

        self.start_button = ctk.CTkButton(self, text="Start Enrollment", command=self.start_enrollment)

        self.start_button.pack(pady=20)

    def start_enrollment(self):

        person_name = self.name_entry.get().strip()
        if len(person_name) == 0:
            self.status_label.configure(text="Please enter name")
            return

        self.start_callback(person_name)

    def update_status(self, text):

        self.status_label.configure(text=text,)