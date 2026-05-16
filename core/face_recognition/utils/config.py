#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      config.py
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
import yaml
from pathlib import Path

def _get_root_dir():

    return Path(os.getenv("FACE_ATTEND_ROOT", ".")).resolve()

def _load_yml_file(config_path):

    if not config_path.exists():
        raise FileNotFoundError(f"config.yml not found at: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_common_config():

    root_dir = _get_root_dir()

    config = _load_yml_file(Path(root_dir / "core" / "face_recognition" / "configs" /"common.yml"))

    return root_dir, config

def load_enrollment_config():

    root_dir = _get_root_dir()

    config = _load_yml_file(Path(root_dir / "core" / "face_recognition" / "configs" /"enrollment.yml"))

    return root_dir, config