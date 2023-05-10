import os
import unittest
from diffusers import StableDiffusionPipeline


class TestDiffuser(unittest.TestCase):

    def setUp(self) -> None:
        self.prompt_prefix = "dota 2 hero"
        self.prompt_suffix = \
            "full body, digital painting, high quality, artstation, highly detailed, " \
            "sharp focus, cinematic lighting, hyperrealistic"

        self.negative_prompt = \
            "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, " \
            "out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, " \
            "watermark, signature, cut off, low contrast, underexposed, overexposed, " \
            "bad art, beginner, amateur, distorted face, blurry, draft, grainy, bad hands, " \
            "missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, bad eyes"

    def test_diffuser(self):
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        hero = "Juggernaut"
        prompt = f"{self.prompt_prefix}, {hero}, beautiful, atmosphere, " \
                 f"mist, smoke, chimney, puddles, melting, dripping, " \
                 f"snow, creek, lush, ice, bridge, {self.prompt_suffix}"

        pipeline = StableDiffusionPipeline.from_pretrained(
            "/home/ywz/data/models/stable-diffusion-v1-5",
            safety_checker=None,
            requires_safety_checker=False
        )
        pipeline.unet.load_attn_procs("/home/ywz/data/dota2/models")
        pipeline.to("cuda")
        image = pipeline(
            prompt=prompt,
            width=640,
            height=480,
            negative_prompt=self.negative_prompt
        ).images[0]
        image.save(os.path.join(output_dir, "test_9.png"))


if __name__ == "__main__":
    unittest.main()
