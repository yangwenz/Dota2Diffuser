import os
import unittest
from diffusers import StableDiffusionPipeline


class TestDiffuser(unittest.TestCase):

    def setUp(self) -> None:
        styles = [
            "artstation, digital painting, hyperrealistic, elegant",
            "cosplay, ultra realistic, elegant",
            "anime, elegant, beautiful",
            # "pop up paper card",
            "porcelain statue",

            "Takashi Murakami, painting",
            "Ukiyo-e, painting",
            "Alphonse Mucha, painting",
            "John Collier, painting",
            "Margaret Macdonald Mackintosh, painting",
            "Alma Thomas, painting",
            "Kawanabe Kyosai, painting",
            "Amrita Sher-Gil, painting",
            "Ravi Varma, painting",
            "Vincent van Gogh, painting",
            "Jacob Lawrence, painting",
            "Salvador Dali, painting",
            "John Singer Sargent, painting",
            "Brad Rigney, painting",
            "Andrew Warhol, painting",
            "Android Jones, painting"
        ]
        self.prompt_suffix = \
            f"high quality, best quality, highly detailed, ultra detailed, " \
            f"masterpiece, " \
            f"{styles[0]}"

        self.negative_prompt = \
            "ugly, lowres, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, " \
            "out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, " \
            "watermark, signature, cut off, low contrast, underexposed, overexposed, " \
            "bad art, beginner, amateur, distorted face, blurry, draft, grainy, bad hands, " \
            "missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, " \
            "text, error, normal quality, jpeg artifacts, username, artist name, fused clothes, " \
            "poorly drawn clothes"

    def test_diffuser(self):
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        hero = "windranger_dota"
        content = f"{hero}"
        prompt = f"{content}, {self.prompt_suffix}".strip().lower()
        print(prompt)

        pipeline = StableDiffusionPipeline.from_pretrained(
            "/home/ywz/data/models/stable-diffusion-v1-5",
            safety_checker=None,
            requires_safety_checker=False
        )
        pipeline.unet.load_attn_procs("/home/ywz/data/dota2/test_17")
        pipeline.to("cuda")
        image = pipeline(
            prompt=prompt,
            width=480,
            height=720,
            negative_prompt=self.negative_prompt
        ).images[0]
        image.save(os.path.join(output_dir, "test_24.png"))


if __name__ == "__main__":
    unittest.main()
