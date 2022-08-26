# make sure you're logged in with `huggingface-cli login`
from genericpath import isfile
from torch import autocast
import torch
import os
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionPipeline
from image_to_image import StableDiffusionImg2ImgPipeline, preprocess


prompt = "a photo of an astronaut riding a horse on mars"
inputPath = "input.png"
accessToken = ""

if os.path.isfile(inputPath):
    print("start image2image")
    pipeimg = StableDiffusionImg2ImgPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="fp16", 
        torch_dtype=torch.float16,
        use_auth_token=accessToken
    ).to("cuda")

    img = Image.open(inputPath)
    img = img.convert("RGB")
    img = img.resize((512, 512))
    img = preprocess(img)
    for i in range(2):
        with autocast("cuda"):
            image = pipeimg(prompt = prompt, init_image=img)["sample"][0]
            image.save("inputed_" + str(i) + ".png")
else:
    print("start txt2image")
    pipe = StableDiffusionPipeline.from_pretrained(
    	"CompVis/stable-diffusion-v1-4", 
        revision="fp16",
        torch_dtype=torch.float16,
	    use_auth_token=accessToken
    ).to("cuda")

    for i in range(2):
        with autocast("cuda"):
            image = pipe(prompt)["sample"][0]  
            image.save(str(i) + ".png")