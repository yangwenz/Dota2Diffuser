import os
import unittest
from compel import Compel
from diffusers import StableDiffusionPipeline
from configs.parser import ConfigParser


class TestDiffuser(unittest.TestCase):

    def setUp(self) -> None:
        self.config = ConfigParser()
        styles = [
            "beautiful detailed eyes, cinematic lighting, "
            "trending on artstation, award-winning, 8k wallpaper, highres, superb",

            "cosplay, ultra realistic, highly detailed eyes, cinematic lighting, "
            "8k wallpaper, highres, superb",

            "1girl, (tachi-e)+, original, illustration+, (ink splashing)+, "
            "(color splashing)+, watercolor+, make happy expressions, soft smile, pure, "
            "beautiful detailed face and eyes, beautiful intricacy clothing, outdoors, "
            "(flower, woods)+, rocks, flower background, lake, (full body)++",

            "(porcelain statue)++, perfect face",            # 3
            "(Takashi Murakami)++",                          # 4
            "(Ukiyo-e)++",                                   # 5
            "(Alphonse Mucha)++",                            # 6
            "(John Collier)++",                              # 7
            "(Margaret Macdonald Mackintosh)++",             # 8
            "(Alma Thomas)++",                               # 9
            "(Kawanabe Kyosai)++",                           # 10
            "(Amrita Sher-Gil)++",                           # 11
            "(Ravi Varma)++",                                # 12
            "(Vincent van Gogh)++",                          # 13
            "(Jacob Lawrence)++",                            # 14
            "(Salvador Dali)++",                             # 15
            "(John Singer Sargent)++",                       # 16
            "(Brad Rigney)++",                               # 17
            "(Andrew Warhol)++",                             # 18
            "(Android Jones)++"                              # 19
        ]
        self.prompt_suffix = \
            f"best quality, highest quality, ultra detailed, masterpiece, " \
            f"intricate, {styles[0]}"

        self.negative_prompt = \
            "ugly, lowres, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, " \
            "out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, " \
            "watermark, signature, cut off, low contrast, underexposed, overexposed, " \
            "bad art, beginner, amateur, distorted face, blurry, draft, grainy, bad hands, " \
            "missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, " \
            "text, error, normal quality, jpeg artifacts, artist logo, artist name, fused clothes, " \
            "poorly drawn clothes, missing arms, missing legs, extra arms, extra legs, extra fingers"

    def test_diffuser(self):
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_dir = "/home/ywz/data/dota2/model"
        hero = "Crystal Maiden"
        hero_index = self.config.hero2index[hero]
        hero_token = f"{hero.lower().replace(' ', '_')}_dota"

        content = "a person standing on the street"
        prompt = f"({hero_token})+, {content}, {self.prompt_suffix}".strip().lower()
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
            width=480,
            height=720,
            negative_prompt=self.negative_prompt,
            cross_attention_kwargs={"label": hero_index}
        ).images[0]
        image.save(os.path.join(output_dir, "test_39.png"))


if __name__ == "__main__":
    unittest.main()
