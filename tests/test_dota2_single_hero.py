import os
import torch
import unittest
from compel import Compel
from diffusers import StableDiffusionPipeline
from configs.parser import ConfigParser


class TestDiffuser(unittest.TestCase):

    def setUp(self) -> None:
        self.config = ConfigParser()
        self.styles = [
            "beautiful detailed eyes, cinematic lighting, "
            "trending on artstation, award-winning, 8k wallpaper, highres, superb",

            "ultra realistic, highly detailed eyes, cinematic lighting, "
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
            "(Android Jones)++",                             # 19
            "(Pablo Picasso)++"                              # 20
        ]
        self.prompt_suffix = \
            "best quality, highest quality, ultra detailed, masterpiece, " \
            "intricate, {}"

        self.negative_prompt = \
            "ugly, lowres, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, " \
            "out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, " \
            "watermark, signature, cut off, low contrast, underexposed, overexposed, " \
            "bad art, beginner, amateur, distorted face, blurry, draft, grainy, bad hands, " \
            "missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, " \
            "text, error, normal quality, jpeg artifacts, artist logo, artist name, fused clothes, " \
            "poorly drawn clothes, missing arms, missing legs, extra arms, extra legs, extra fingers, " \
            "duplicate, cloned face, fused fingers, long neck, malformed limbs, morbid, " \
            "mutated hands, mutation, mutilated"

        model_dir = "/home/ywz/data/dota2/model_2/final"
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "/home/ywz/data/models/stable-diffusion-v1-5",
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=torch.float16
        )
        self.compel_proc = Compel(
            tokenizer=self.pipeline.tokenizer,
            text_encoder=self.pipeline.text_encoder
        )
        self.pipeline.unet.load_attn_procs(model_dir)
        self.pipeline.to("cuda")

    def _run_pipeline(self, prompt, hero_index, filename):
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        prompt_embeds = self.compel_proc(prompt)
        image = self.pipeline(
            # prompt=prompt,
            prompt_embeds=prompt_embeds,
            width=480,
            height=720,
            negative_prompt=self.negative_prompt,
            cross_attention_kwargs={"label": hero_index}
        ).images[0]
        image.save(os.path.join(output_dir, filename))

    def test_diffuser(self):
        hero = "Lina"
        hero_index = self.config.hero2index[hero]
        hero_token = f"{hero.lower().replace(' ', '_')}_dota"

        content = "a girl standing, (full body)++"
        test_styles = [0]
        num_samples = 5

        for style in test_styles:
            for _ in range(num_samples):
                prompt_suffix = self.prompt_suffix.format(self.styles[style])
                prompt = f"{hero_token}, {content}, {prompt_suffix}".strip().lower()
                print(prompt)

                self._run_pipeline(
                    prompt=prompt,
                    hero_index=hero_index,
                    filename="test_40.png"
                )
                if len(test_styles) > 1 or num_samples > 1:
                    input("next?")


if __name__ == "__main__":
    unittest.main()
