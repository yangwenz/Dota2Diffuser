import numpy as np
import torch

from diffusers import StableDiffusionPipeline
from configs.parser import ConfigParser


def reset_weights():
    model_dir = "/home/ywz/data/dota2/model"
    hero = "Morphling"
    rank = 4

    config = ConfigParser()
    hero_index = config.hero2index[hero]

    pipeline = StableDiffusionPipeline.from_pretrained(
        "/home/ywz/data/models/stable-diffusion-v1-5",
        safety_checker=None,
        requires_safety_checker=False
    )
    pipeline.unet.load_attn_procs(model_dir)

    # Reset the weights of a particular hero
    state_dict = pipeline.unet.state_dict()
    for name, param in pipeline.unet.named_parameters():
        if name.find("lora") != -1:
            data = param.cpu().detach().numpy()
            if name.find("down.weight_extended") != -1:
                data[hero_index] = np.random.randn(*param.shape[1:]) * (1 / rank)
            elif name.find("up.weight_extended") != -1:
                data[hero_index] = 0
            state_dict[name] = torch.tensor(data, dtype=param.dtype)
    pipeline.unet.load_state_dict(state_dict)
    # Save the modified weights
    pipeline.unet.save_attn_procs(model_dir)


if __name__ == "__main__":
    reset_weights()
