import torch
from iopath.common.file_io import g_pathmgr

checkpoint_path = 'output/dvgt2/train/default_debug/ckpts/checkpoint_9.pt'

with g_pathmgr.open(checkpoint_path, "rb") as f:
    checkpoint = torch.load(f, map_location="cpu")

new_ckpt = checkpoint['model']

save_path = 'ckpt/dvgt2.pt'
torch.save(new_ckpt, save_path)