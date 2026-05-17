
import os
from core.face_recognition.utils.config import _get_root_dir

def asset_resources(relative_path):

    root_repo = _get_root_dir()
    assets_path = root_repo / "assets" / "icons"
    base_path = os.path.abspath(assets_path)

    return os.path.join(base_path, relative_path)