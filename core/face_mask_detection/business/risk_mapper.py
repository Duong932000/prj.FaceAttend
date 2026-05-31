RISK_MAPPING = {
    "mask": "safe",
    "mask_chin": "unsafe",
    "mask_mouth_chin": "unsafe",
    "mask_nose_mouth": "unsafe",
    "no_mask": "unsafe"
}


def map_risk(detail_class_name):

    return RISK_MAPPING[detail_class_name]