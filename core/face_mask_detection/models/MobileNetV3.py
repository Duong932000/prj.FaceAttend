#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      MobileNetV3.py
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


import torch
import torch.nn as nn
from typing import Optional
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

def get_model(num_classes: int,
              pretrained: bool = True,
              dropout_rate: float = 0.3,
              freeze_backbone: bool = False,
              width_mult: float = 1.0) -> nn.Module:

    # Load MobileNetV3 LARGE
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights, width_mult=width_mult)

    # Unfreeze all for full fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    # custom classifier (anti-overfit)
    in_features = model.classifier[0].in_features

    model.classifier = nn.Sequential(
        # Global Average Pooling (already in features)
        nn.Linear(in_features, 1024),      # Expand
        nn.Hardswish(),                    # MobileNetV3 activation
        nn.Dropout(dropout_rate),          # Dropout 30%
        
        nn.Linear(1024, 512),
        nn.Hardswish(),
        nn.Dropout(dropout_rate * 0.5),
        
        nn.Linear(512, num_classes)        # Final output
    )

    # Optional: Freeze backbone (early training)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        print("Backbone frozen - only classifier training")

    print(f"MobileNetV3-Large created:")
    print(f"   - Classes: {num_classes}")
    print(f"   - Width: {width_mult}")
    print(f"   - Dropout: {dropout_rate}")
    print(f"   - Backbone frozen: {freeze_backbone}")
    print(f"   - Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M trainable")

    return model
