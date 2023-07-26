import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import numpy as np

torch.cuda.empty_cache()
model =  StableDiffusionPipeline.from_pretrained( "Lykon/DreamShaper", custom_pipeline="lpw_stable_diffusion", torch_dtype=torch.float32,resume_download = True ,safety_checker=None )
text = "A sunny day at the lake with mountains in the background"
image = model(prompt=text, width=512, height=512,  num_inference_steps=20).images[0]

prompt_embeds, negative_prompt_embeds = model._encode_prompt(text, None, 1, True, "")
print(prompt_embeds)
print("prompt_shape", prompt_embeds.shape)
print(negative_prompt_embeds)
print("negative_prompt_shape", negative_prompt_embeds.shape)

image.save("1.png")
plt.imshow(image)
plt.show()

# prompt_embeds = prompt_embeds.unsqueeze(-1)
# negative_prompt_embeds = negative_prompt_embeds.unsqueeze(-1)

#image = model(prompt = None, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds)

# plt.imshow(image[0])
# plt.show()