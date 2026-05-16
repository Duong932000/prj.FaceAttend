
class PoseValidator:
    def __init__(self, pose_config):
        self.pose_config = pose_config

    def validate(self, face, target_pose):
        
        if target_pose not in self.pose_config:
            return False
    
        pose_cfg = self.pose_config[target_pose]
        yaw, pitch, _ = face.pose

        return (
            pose_cfg["yaw_min"] <= yaw <= pose_cfg["yaw_max"] and pose_cfg["pitch_min"] <= pitch <= pose_cfg["pitch_max"])