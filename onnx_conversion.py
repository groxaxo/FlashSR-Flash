
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn.functional as F

from FastAudioSR import FASR
model = FASR(f'models/upsampler.pth')

# Export model as-is
dummy_input = torch.randn(1, 1, 16000)
seq_dim = torch.export.Dim("sequence_length", min=128, max=16000 * 120)

# Export via torch.export first
exported = torch.export.export(
    model.model,
    (dummy_input,),
    dynamic_shapes={"x": {2: seq_dim}}  # or ({2: seq_dim},) for positional
)

# Then convert to ONNX
onnx_program = torch.onnx.export(exported, dynamo=True)
onnx_program.save("models/model.onnx")
print("Successfully exported full model to models/model.onnx")

# Export model with optimization for 2x faster processing
class FastUpSample1d(torch.nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
    
    def forward(self, x):
        return F.interpolate(x, scale_factor=float(self.scale_factor), mode='linear', align_corners=False)

def replace_upsample(module):
    for name, child in module.named_children():
        if child.__class__.__name__ == 'UpSample1d':
            setattr(module, name, FastUpSample1d())
        else:
            replace_upsample(child)

replace_upsample(model.model)

# Export via torch.export first
exported = torch.export.export(
    model.model,
    (dummy_input,),
    dynamic_shapes={"x": {2: seq_dim}}  # or ({2: seq_dim},) for positional
)

# Then convert to ONNX
onnx_program = torch.onnx.export(exported, dynamo=True)
onnx_program.save("models/model_lite.onnx")
print("Successfully exported optimized model to models/model_lite.onnx")