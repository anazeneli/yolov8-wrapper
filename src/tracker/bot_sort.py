from ultralytics.trackers.bot_sort import ReID
from ultralytics.trackers.bot_sort import BOTSORT as BaseBOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.plotting import save_one_box

from reid.osnet_ain import osnet_ain_x0_25, osnet_ain_x0_5, osnet_ain_x0_75, osnet_ain_x1_0
from reid.osnet_encoder import OSNetEncoder
import torch
import re

class BOTSort(BaseBOTSORT):
    def __init__(self, args, frame_rate=30):
        # Store original model path before parent init
        original_model = getattr(args, 'model', None)
        original_with_reid = getattr(args, 'with_reid', False)
        
        # Temporarily disable ReID in parent class
        if hasattr(args, 'with_reid'):
            args.with_reid = False
        
        # Call parent constructor (this won't create ReID now)
        super().__init__(args, frame_rate)
        
        # Restore original values
        args.model = original_model
        args.with_reid = original_with_reid
        
        # Now create YOUR custom ReID encoder
        device = getattr(args, "device", "cpu")
        if isinstance(device, int):
            device = f"cuda:{device}"
        device = str(device)

        if args.with_reid and original_model:
            if original_model == "auto":
                self.encoder = lambda feats, s: [f.cpu().numpy() for f in feats]
            elif original_model.endswith('.pth') or original_model.endswith('.pth.tar'):
                # Your OSNet setup
                arch_search = re.search(r'x([01]_0|0_25|0_5|0_75|1_0)', original_model)
                if arch_search:
                    arch_str = arch_search.group(0)
                else:
                    arch_str = 'x0_25'
                    
                if 'x0_25' in arch_str:
                    osnet_model = osnet_ain_x0_25(num_classes=2, pretrained=False)
                elif 'x0_5' in arch_str:
                    osnet_model = osnet_ain_x0_5(num_classes=2, pretrained=False)
                elif 'x0_75' in arch_str:
                    osnet_model = osnet_ain_x0_75(num_classes=2, pretrained=False)
                elif 'x1_0' in arch_str:
                    osnet_model = osnet_ain_x1_0(num_classes=2, pretrained=False)
                else:
                    osnet_model = osnet_ain_x0_25(num_classes=2, pretrained=False)
                    
                print(f"Loading OSNet model from: {original_model}")
                state_dict = torch.load(original_model, map_location=device)
                state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
                osnet_model.load_state_dict(state_dict)
                osnet_model.eval()
                self.encoder = OSNetEncoder(osnet_model, device=device)
                print("Custom OSNet ReID encoder created successfully!")
            else:
                # Fall back to default ReID for other model types
                self.encoder = ReID(original_model)
        else:
            self.encoder = None