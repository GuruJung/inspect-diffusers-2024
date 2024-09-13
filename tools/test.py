import os
import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:4" if torch.cuda.is_available() else "cpu")

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt" #, torch_dtype=torch.float16, variant="fp16"
).to(device)

#pipe.enable_model_cpu_offload()

print(os.getcwd())

# Load the conditioning image
image = load_image("./images/fire-car-11-700.png")
image = image.resize((1024, 576))


generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

print(len(frames), frames)

#export_to_video(frames, "./results/fire-car-11-700.mp4", fps=7)





