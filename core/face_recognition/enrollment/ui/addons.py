import sys
from CTkMessagebox import CTkMessagebox

def exit_ui(window):

    msg_ExitSystem \
        = CTkMessagebox(master=window,
                        title="Exit",
                        message="Do you want to exit the FaceID Enrollment System",
                        icon="question",
                        option_1="Cancel",
                        option_2="Exit")
    if msg_ExitSystem.get() == "Exit":
        window.destroy()
        sys.exit()