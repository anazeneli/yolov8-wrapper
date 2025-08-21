import torch 
import numpy as np

class OSNetEncoder:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def __call__(self, img, bboxes):
        if len(bboxes) == 0:
            return np.zeros((0, getattr(self.model, 'feature_dim', 512)))
            
        # Convert to tensor once
        img_tensor = torch.from_numpy(img).float().to(self.device)
        h_img, w_img = img.shape[:2]
        
        crops = []
        for xywh in bboxes:
            x_c, y_c, w, h = xywh[:4]
            
            x1 = max(0, int(x_c - w / 2))
            y1 = max(0, int(y_c - h / 2))
            x2 = min(w_img, int(x_c + w / 2))
            y2 = min(h_img, int(y_c + h / 2))
            
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
                
            # Extract crop and resize in one go
            crop = img_tensor[y1:y2, x1:x2, :]
            crop = crop.permute(2, 0, 1)  # HWC -> CHW
            crop = torch.nn.functional.interpolate(
                crop.unsqueeze(0), size=(128, 64), mode='bilinear'
            ).squeeze(0)
            
            # Normalize
            crop = crop / 255.0
            crop = (crop - torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(self.device)) / \
                   torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(self.device)
            
            crops.append(crop)
        
        if crops:
            batch = torch.stack(crops)
            with torch.no_grad():
                features = self.model(batch)
            return features.cpu().numpy()
        else:
            return np.zeros((0, getattr(self.model, 'feature_dim', 512)))