import os
import unittest
from diffusers import StableDiffusionPipeline


class TestDiffuser(unittest.TestCase):

    def test_diffuser(self):
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pipeline = StableDiffusionPipeline.from_pretrained("/home/ywz/data/models/stable-diffusion-v1-5")
        pipeline.unet.load_attn_procs("/home/ywz/data/dota2/models")
        pipeline.to("cuda")
        image = pipeline("a dota 2 hero Juggernaut with a sword fights in forest").images[0]
        image.save(os.path.join(output_dir, "test_1.png"))


if __name__ == "__main__":
    unittest.main()
