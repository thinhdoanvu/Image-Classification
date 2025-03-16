import torch
import torchvision.models as models

# Load VGG19 model
from model import *
model_vgg = VGG19(num_classes=1)
model_vggse = SEVGG19(num_classes=1)
model_vggcbam = CBAMVGG19(num_classes=1)
model_vggeca = ECAVGG19(num_classes=1)


# Count total trainable parameters
total_params_vgg = sum(p.numel() for p in model_vgg.parameters() if p.requires_grad)
total_params_vggse = sum(p.numel() for p in model_vggse.parameters() if p.requires_grad)
total_params_vggcbam = sum(p.numel() for p in model_vggcbam.parameters() if p.requires_grad)
total_params_vggeca = sum(p.numel() for p in model_vggeca.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_params_vgg}")
print(f"Total Trainable Parameters: {total_params_vggse}")
print(f"Total Trainable Parameters: {total_params_vggcbam}")
print(f"Total Trainable Parameters: {total_params_vggeca}")

# ----------------------------------------------------#
# Compute FLOPs (Floating Point Operations)
from fvcore.nn import FlopCountAnalysis

# Input tensor (e.g., 3x224x224 for ImageNet)
input_tensor = torch.randn(1, 3, 224, 224)

# Compute FLOPs
flops_vgg = FlopCountAnalysis(model_vgg, input_tensor)
flops_vggse = FlopCountAnalysis(model_vggse, input_tensor)
flops_vggcbam = FlopCountAnalysis(model_vggcbam, input_tensor)
flops_vggeca = FlopCountAnalysis(model_vggeca, input_tensor)

print(f"VGG19 FLOPs totally: {flops_vgg.total() / 1e9} GFLOPs")
print(f"VGG19_SE FLOPs totally: {flops_vggse.total() / 1e9} GFLOPs")
print(f"VGG19_CBAM FLOPs totally: {flops_vggcbam.total() / 1e9} GFLOPs")
print(f"VGG19_ECA FLOPs totally: {flops_vggeca.total() / 1e9} GFLOPs")

# ----------------------------------------------------#
# Model Summary
from torchsummary import summary
summary(model_vgg, (3, 224, 224))
summary(model_vggse, (3, 224, 224))
summary(model_vggcbam, (3, 224, 224))
summary(model_vggeca, (3, 224, 224))
