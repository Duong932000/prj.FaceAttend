#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      enrollment.py
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

import os
import sys
import customtkinter
from CTkMessagebox import CTkMessagebox
from core.face_recognition.enrollment.ui.window import MainWindow


if __name__ == "__main__":
    # f_DashboardInitialized = False

    # try:
        MainWindow().mainloop()
    # except Exception as e:
    #     if not f_DashboardInitialized:
    #         root_error = customtkinter.CTk()
    #         root_error.withdraw()
    #         msg_error = CTkMessagebox(master=root_error,
    #                                   title="FaceID Enrollment",
    #                                   message=f"Error: {e}",
    #                                   icon="warning",
    #                                   option_1="Ok")
    #         if msg_error.get() == "Ok":
    #             root_error.destroy()
    #             sys.exit()
