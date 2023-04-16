import sys
sys.path.append("../../models/screenclassification")

from tqdm import tqdm
import glob
import torch
from ui_models import *

checkpoints = glob.glob("../../downloads/checkpoints/screenclassification*ckpt")
for checkpoint in tqdm(checkpoints):
    m = UIScreenClassifier.load_from_checkpoint(checkpoint).eval()
    s = m.to_torchscript(method="trace", example_inputs=[torch.rand(1, 3, 256, 256)])

    test_input = torch.rand(1, 3, 384, 512)
    o1 = m(test_input)
    o2 = s(test_input)

    print(torch.allclose(o1, o2))
    torch.jit.save(s, checkpoint.replace(".ckpt", ".torchscript"))