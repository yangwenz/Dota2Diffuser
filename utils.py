import os
import torch
import pickle
import numpy as np


def dump_lora_weights(unet, hero_index, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "pytorch_lora_weights.bin")
    try:
        state_dict = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"The file {path} doesn't exist, save the UNet weights directly.")
        unet.save_attn_procs(output_dir)
        state_dict = torch.load(path, map_location="cpu")

    weights = {}
    for name, param in unet.named_parameters():
        if name.find("lora") != -1:
            data = param.cpu().detach().numpy()
            state_dict[name][hero_index] = torch.tensor(data[hero_index], dtype=param.dtype)
            weights[name] = data[hero_index]

    # Save the trained LoRA weights
    backup_dir = os.path.join(output_dir, "backups")
    os.makedirs(backup_dir, exist_ok=True)
    with open(os.path.join(backup_dir, f"{hero_index}"), "wb") as f:
        pickle.dump(weights, f)
    torch.save(state_dict, path)

    # Verify
    state_dict = torch.load(path, map_location="cpu")
    for name, param in unet.named_parameters():
        if name.find("lora") != -1:
            a = param.cpu().detach().numpy()[hero_index]
            b = state_dict[name][hero_index].detach().numpy()
            c = weights[name]
            assert np.sum(np.abs(a - b)) < 1e-6
            assert np.sum(np.abs(a - c)) < 1e-6
