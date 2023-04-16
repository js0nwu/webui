import sys
sys.path.append("../../models/screensim")

from tqdm import tqdm
import glob
import torch
from ui_models import *

checkpoints = glob.glob("../../downloads/checkpoints/screensim*ckpt")
for checkpoint in tqdm(checkpoints):
    m = UIScreenEmbedder.load_from_checkpoint(checkpoint).eval()
    s = torch.jit.script(m.model)

    test_input = torch.rand(1, 3, 512, 1024)
    o1 = m(test_input)
    o2 = s(test_input)

    print(torch.allclose(o1, o2))

    torch.jit.save(s, checkpoint.replace(".ckpt", ".torchscript"))