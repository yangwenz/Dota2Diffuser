# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import torch
from typing import List

from cog import BasePredictor, Input, Path
from compel import Compel
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from configs.parser import ConfigParser

MODEL_ID = "runwayml/stable-diffusion-v1-5"
MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"


class Predictor(BasePredictor):

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.config = ConfigParser()
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_MODEL_ID,
            cache_dir=MODEL_CACHE
        )
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=MODEL_CACHE
        )
        self.compel_proc = Compel(
            tokenizer=self.pipeline.tokenizer,
            text_encoder=self.pipeline.text_encoder
        )
        self.pipeline.unet.load_attn_procs("./models")
        self.pipeline.to("cuda")

    @torch.inference_mode()
    def predict(
            self,
            hero: str = Input(
                description="Input dota 2 hero",
                default="Lina",
            ),
            prompt: str = Input(
                description="Input prompt",
                default="lina_dota, standing on a bridge, "
                        "best quality, highest quality, ultra detailed, "
                        "masterpiece, intricate, beautiful detailed face, beautiful detailed eyes, "
                        "cinematic lighting, painting, award-winning",
            ),
            negative_prompt: str = Input(
                description="Specify things to not see in the output",
                default="ugly, lowres, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, "
                        "out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, "
                        "watermark, signature, cut off, low contrast, underexposed, overexposed, "
                        "bad art, beginner, amateur, distorted face, blurry, draft, grainy, bad hands, "
                        "missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, "
                        "text, error, normal quality, jpeg artifacts, artist logo, artist name, fused clothes, "
                        "poorly drawn clothes, missing arms, missing legs, extra arms, extra legs, extra fingers"
            ),
            width: int = Input(
                description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
                choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
                default=512,
            ),
            height: int = Input(
                description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
                choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
                default=512,
            ),
            num_inference_steps: int = Input(
                description="Number of denoising steps", ge=1, le=500, default=50
            ),
            guidance_scale: float = Input(
                description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
            ),
            seed: int = Input(
                description="Random seed. Leave blank to randomize the seed", default=None
            ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. "
                "Please select a lower width or height."
            )
        generator = torch.Generator("cuda").manual_seed(seed) \
            if seed is not None else None

        hero_index = self.config.hero2index[hero]
        prompt_embeds = self.compel_proc(prompt)
        output = self.pipeline(
            prompt_embeds=prompt_embeds,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            generator=generator,
            cross_attention_kwargs={"label": hero_index}
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            if output.nsfw_content_detected and output.nsfw_content_detected[i]:
                continue
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )
        return output_paths
