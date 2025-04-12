from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
import torch
from PIL import Image
import pandas as pd
import argparse
import random
import numpy as np
from tqdm import tqdm
import requests
import os, glob
import pandas as pd
import re
from transformers import CLIPProcessor, CLIPModel

def generate_images(ldm_stable, concept, save_path='/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/bias/bias/bias2', device='cuda:0', guidance_scale=7.5, image_size=512, ddim_steps=100, num_samples=10, random_seed=42):

    if ldm_stable is None:
        raise ValueError("The 'ldm_stable' model must be provided.")

    vae = ldm_stable.vae
    tokenizer = ldm_stable.tokenizer
    text_encoder = ldm_stable.text_encoder
    unet = ldm_stable.unet
    scheduler = ldm_stable.scheduler
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    print("Init done")
    
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Prepare the folder for saving images
    folder_path = f'{save_path}/class'
    os.makedirs(folder_path, exist_ok=True)
    # Generate images
    prompt = [str(concept)]*num_samples
    print("prompt:", prompt)
    height = image_size                        # default height of Stable Diffusion
    width = image_size                         # default width of Stable Diffusion

    num_inference_steps = ddim_steps           # Number of denoising steps

    guidance_scale = guidance_scale            # Scale for classifier-free guidance

    generator = torch.manual_seed(random_seed)    # Seed generator to create the inital latent noise

    batch_size = len(prompt)

    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(device)

    scheduler.set_timesteps(num_inference_steps)

    latents = latents * scheduler.init_noise_sigma

    from tqdm.auto import tqdm

    scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    for num, im in enumerate(pil_images):
        im.save(f"{folder_path}/0_{num}.png")

    print(f"Images generated and saved to {folder_path}")
def sorted_nicely( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def CLIP_classification(im_path, attributes, prompts_path, save_path, from_case, till_case):
    attributes = [attrib.strip() for attrib in attributes.split(',')]
    model = CLIPModel.from_pretrained("/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/clip-vit")
    processor = CLIPProcessor.from_pretrained("/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/clip-vit")
    model_name = os.path.basename(os.path.abspath(im_path))
    print(model_name, attributes, save_path)
    df = pd.read_csv(prompts_path)
    images = os.listdir(im_path)
    images = [im for im in images if '.png' in im]
    images = sorted_nicely(images)
    ratios = {}
    columns = [f"{att.replace(' ','_')}_bias" for att in attributes]
    for col in columns:
        df[col] = np.nan
    for image in images:
        try:
            case_number = int(image.split('_')[0].replace('.png',''))
            if case_number < from_case or case_number > till_case:
                continue

            im = Image.open(os.path.join(im_path, image))

            inputs = processor(text=attributes, images=im, return_tensors="pt", padding=True)

            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
            tmax = probs.max(1, keepdim=True)[0]
            mask = list(probs.ge(tmax)[0].float().numpy())
            ratios[case_number] = ratios.get(case_number, []) + [mask]
        except Exception:
            ratios[case_number] = ratios.get(case_number, []) + [[0]*len(attributes)]
    count_of_ones = 0
    for key in ratios.keys():
        print(np.array(ratios[key]))
        count_of_ones += np.sum(np.array(ratios[key])[:, 0] == 1)  
        for idx, col in enumerate(columns):
            df.loc[key,col] = np.mean(np.array(ratios[key])[:,idx])
    return count_of_ones
# # Iidian, Asian, African, European, Latino
def CLIP_classification_race(im_path, attributes, prompts_path, save_path, from_case, till_case):
    attributes = [attrib.strip() for attrib in attributes.split(',')]
    model = CLIPModel.from_pretrained("/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/clip-vit")
    processor = CLIPProcessor.from_pretrained("/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/clip-vit")
    model_name = os.path.basename(os.path.abspath(im_path))
    print(model_name, attributes, save_path)
    df = pd.read_csv(prompts_path)
    images = os.listdir(im_path)
    images = [im for im in images if '.png' in im]
    images = sorted_nicely(images)
    ratios = {}
    columns = [f"{att.replace(' ','_')}_bias" for att in attributes]
    for col in columns:
        df[col] = np.nan
    for image in images:
        try:
            case_number = int(image.split('_')[0].replace('.png',''))
            if case_number < from_case or case_number > till_case:
                continue

            im = Image.open(os.path.join(im_path, image))

            inputs = processor(text=attributes, images=im, return_tensors="pt", padding=True)

            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
            tmax = probs.max(1, keepdim=True)[0]
            mask = list(probs.ge(tmax)[0].float().numpy())
            ratios[case_number] = ratios.get(case_number, []) + [mask]
        except Exception:
            ratios[case_number] = ratios.get(case_number, []) + [[0]*len(attributes)]
    count_of_ones_indian = 0
    count_of_ones_asian = 0
    count_of_ones_african = 0
    count_of_ones_european = 0
    count_of_ones_latino = 0
    for key in ratios.keys():
        print(np.array(ratios[key]))
        count_of_ones_indian += np.sum(np.array(ratios[key])[:,0] == 1)  
        count_of_ones_asian += np.sum(np.array(ratios[key])[:,1] == 1)
        count_of_ones_african += np.sum(np.array(ratios[key])[:,2] == 1)
        count_of_ones_european += np.sum(np.array(ratios[key])[:,3] == 1)
        count_of_ones_latino += np.sum(np.array(ratios[key])[:,4] == 1)
        for idx, col in enumerate(columns):
            df.loc[key,col] = np.mean(np.array(ratios[key])[:,idx])
    return  count_of_ones_indian, count_of_ones_asian, count_of_ones_african, count_of_ones_european, count_of_ones_latino