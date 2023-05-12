import os
import unittest
from diffusers import StableDiffusionPipeline


class TestDiffuser(unittest.TestCase):

    def setUp(self) -> None:
        styles = ["fan art", "anime", "cosplay, realistic"]

        self.prompt_prefix = ""
        self.prompt_suffix = \
            f"full body, digital painting, high quality, artstation, highly detailed, " \
            f"sharp focus, cinematic lighting, {styles[0]}"

        self.negative_prompt = \
            "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, " \
            "out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, " \
            "watermark, signature, cut off, low contrast, underexposed, overexposed, " \
            "bad art, beginner, amateur, distorted face, blurry, draft, grainy, bad hands, " \
            "missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, " \
            "bad eyes, bad legs, bad arms"

    def test_diffuser(self):
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        hero = "Juggernaut"
        content = f"{hero} playing with a white dog in a city"
        background = "atmosphere, mist, melting, dripping, snow, creek, lush, bridge"
        prompt = f"{self.prompt_prefix}, {content}, {self.prompt_suffix}".strip().lower()

        pipeline = StableDiffusionPipeline.from_pretrained(
            "/home/ywz/data/models/stable-diffusion-v1-4",
            safety_checker=None,
            requires_safety_checker=False
        )
        pipeline.unet.load_attn_procs("/home/ywz/data/dota2/new_models")
        pipeline.to("cuda")
        image = pipeline(
            prompt=prompt,
            width=640,
            height=480,
            negative_prompt=self.negative_prompt
        ).images[0]
        image.save(os.path.join(output_dir, "test_16.png"))


if __name__ == "__main__":
    unittest.main()
