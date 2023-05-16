import os
import unittest
from compel import Compel
from diffusers import StableDiffusionPipeline
from configs.parser import ConfigParser


class TestDiffuser(unittest.TestCase):

    def setUp(self) -> None:
        self.config = ConfigParser()
        styles = [
            "illustration, beautiful detailed eyes, depth of field",     # 0
            "artstation, hyperrealistic, elegant",       # 1
            "cosplay, ultra realistic, elegant",         # 2
            "pop up paper card",                         # 3
            "porcelain statue",                          # 4

            "Takashi Murakami, painting",                # 5
            "Ukiyo-e, painting",                         # 6
            "Alphonse Mucha, painting",                  # 7
            "John Collier, painting",                    # 8
            "Margaret Macdonald Mackintosh, painting",   # 9
            "Alma Thomas, painting",                     # 10
            "Kawanabe Kyosai, painting",                 # 11
            "Amrita Sher-Gil, painting",                 # 12
            "Ravi Varma, painting",                      # 13
            "Vincent van Gogh, painting",                # 14
            "Jacob Lawrence, painting",                  # 15
            "Salvador Dali, painting",                   # 16
            "John Singer Sargent, painting",             # 17
            "Brad Rigney, painting",                     # 18
            "Andrew Warhol, painting",                   # 19
            "Android Jones, painting"                    # 20
        ]
        self.prompt_suffix = \
            f"high quality, best quality, highly detailed, ultra detailed, " \
            f"masterpiece, " \
            f"{styles[1]}"

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

        model_dir = "/home/ywz/data/dota2/test_lina_1"
        hero = "Lina"
        hero_index = self.config.hero2index[hero]

        hero_token = f"{hero.lower().replace(' ', '_')}_dota"
        content = f"{hero_token}, bikini++, standing"
        prompt = f"{content}, {self.prompt_suffix}".strip().lower()
        print(prompt)

        pipeline = StableDiffusionPipeline.from_pretrained(
            "/home/ywz/data/models/stable-diffusion-v1-5",
            safety_checker=None,
            requires_safety_checker=False
        )
        compel_proc = Compel(
            tokenizer=pipeline.tokenizer,
            text_encoder=pipeline.text_encoder
        )
        pipeline.unet.load_attn_procs(model_dir)
        pipeline.to("cuda")

        prompt_embeds = compel_proc(prompt)
        image = pipeline(
            # prompt=prompt,
            prompt_embeds=prompt_embeds,
            width=512,
            height=512,
            negative_prompt=self.negative_prompt,
            cross_attention_kwargs={"label": hero_index}
        ).images[0]
        image.save(os.path.join(output_dir, "test_25.png"))


if __name__ == "__main__":
    unittest.main()
