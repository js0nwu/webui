import sys
sys.path.append("../../models/screenrecognition")

from tqdm import tqdm
import glob
import torch
from ui_models import *

checkpoints = glob.glob("../../downloads/checkpoints/screenrecognition*ckpt")
for checkpoint in tqdm(checkpoints):
    m = UIElementDetector.load_from_checkpoint(checkpoint).eval()
    s = torch.jit.script(m.model, torch.rand(1, 3, 256, 256))

    test_input = [torch.rand(3, 384, 512)]
    o1 = m.model(test_input)
    o2 = s(test_input)

    print(torch.allclose(o1[0]['boxes'], o2[1][0]['boxes']))
    torch.jit.save(s, checkpoint.replace(".ckpt", ".torchscript"))