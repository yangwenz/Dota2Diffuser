import os
import unittest
from diffusers import StableDiffusionPipeline


class TestDiffuser(unittest.TestCase):

    def setUp(self) -> None:
        styles = [
            "artstation, hyperrealistic, elegant",
            "cosplay, ultra realistic, elegant",
            "gouache, painting",
            "Ukiyo-e, painting",
            "Artemisia Gentileschi",
            "Margaret Macdonald Mackintosh",
            "Alma Thomas",
            "Frederic Edwin Church",
            "Kawanabe Kyosai",
            "Amrita Sher-Gil",
            "Ravi Varma",
            "Max Ernst",
            "Vincent van Gogh",
            "Jacob Lawrence",
            "Pierre-Auguste Renoir"
        ]
        self.prompt_suffix = \
            f"full body, high quality, best quality, highly detailed, ultra detailed, " \
            f"masterpiece, " \
            f"{styles[1]}"

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
        content = f"{hero} standing in a city"
        prompt = f"{content}, {self.prompt_suffix}".strip().lower()

        pipeline = StableDiffusionPipeline.from_pretrained(
            "/home/ywz/data/models/stable-diffusion-v1-4",
            safety_checker=None,
            requires_safety_checker=False
        )
        pipeline.unet.load_attn_procs("/home/ywz/data/dota2/test_1")
        pipeline.to("cuda")
        image = pipeline(
            prompt=prompt,
            width=512,
            height=512,
            negative_prompt=self.negative_prompt
        ).images[0]
        image.save(os.path.join(output_dir, "test_22.png"))


if __name__ == "__main__":
    unittest.main()
