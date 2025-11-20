from dinov3.hub import backbones
from dinov3.hub.backbones import dinov3_vitl16

# weights_path = "dinov3_vit-l16.pth"
weights_path = "./weights/dinov3_vit-l16-ABCDEFGH.pth"
model = dinov3_vitl16(weights=weights_path, pretrained=True)

print("Model loaded successfully")