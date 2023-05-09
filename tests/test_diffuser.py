import os
import unittest
from diffusers import DiffusionPipeline


class TestDiffuser(unittest.TestCase):

    def test_diffuser(self):
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipeline.to("cuda")
        image = pipeline("An image of a squirrel in Picasso style").images[0]
        image.save(os.path.join(output_dir, "test.png"))


if __name__ == "__main__":
    unittest.main()
