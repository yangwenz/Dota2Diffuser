import os
import torch
import pickle


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
            state_dict[name][hero_index] = torch.FloatTensor(data[hero_index])
            weights[name] = data[hero_index]

    backup_dir = os.path.join(output_dir, "backups")
    os.makedirs(backup_dir, exist_ok=True)
    with open(os.path.join(backup_dir, f"{hero_index}"), "wb") as f:
        pickle.dump(weights, f)
    torch.save(state_dict, path)
