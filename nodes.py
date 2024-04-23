import json
import os

import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)

import sys
LaVIT_path=f'{comfy_path}/custom_nodes/ComfyUI-LaVIT'
#sys.path.insert(0,LaVIT_path)

import os
import torch
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from .VideoLaVIT.models import build_model
from PIL import Image
from diffusers.utils import load_image, export_to_video, export_to_gif

class VideoLaVITLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (os.listdir(os.path.join(folder_paths.models_dir,"diffusers")), {"default": "Video-LaVIT-v1"}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("VideoLaVIT",)
    FUNCTION = "run"
    CATEGORY = "LaVIT"

    def run(self,model_path):
        model_path=os.path.join(os.path.join(folder_paths.models_dir,"diffusers"),model_path)
        model_dtype='fp16'
        # Set the load GPU id
        device_id = 0
        torch.cuda.set_device(device_id)
        cpudevice = torch.device('cpu')
        cudadevice = torch.device('cuda')

        # If you have already install xformers, set `use_xformers=True` to save the GPU memory (Xformers is not supported on V100 GPU)
        # If you have already download the checkpoint, set `local_files_only=True`` to avoid auto-downloading from remote
        model = build_model(model_path=model_path, model_dtype=model_dtype, local_files_only=True, 
                        device_id=device_id, use_xformers=True, understanding=False,)
        model = model.to(cpudevice)
        #model.unet.to(cudadevice)
        model.llama_model.to(cudadevice)
        model.visual_tokenizer.to(cudadevice)
        model.tokenizer_decoder.to(cudadevice)

        return (model,)

class VideoLaVITT2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":("VideoLaVIT",),
                "prompt":("STRING",{"multiline": True, "default":"FPV drone footage of an ancient city in autumn"}),
                "keyframe_width":("INT",{"default":1024}),
                "keyframe_height":("INT",{"default":576}),
                "video_width":("INT",{"default":576}),
                "video_height":("INT",{"default":320}),
                "guidance_scale_for_llm":("FLOAT",{"default":4.0}),
                "guidance_scale_for_decoder":("FLOAT",{"default":7.0}),
                "num_inference_steps":("INT",{"default":50}),
                "top_k":("INT",{"default":50}),
                "seed":("INT",{"default":16}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "LaVIT"

    def run(self,model,prompt,keyframe_width,keyframe_height,video_width,video_height,guidance_scale_for_llm,guidance_scale_for_decoder,num_inference_steps,top_k,seed):
        random.seed(seed)
        torch.manual_seed(seed)
        model_dtype='fp16'
        torch_dtype = torch.bfloat16 if model_dtype=="bf16" else torch.float16
        with torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
            videos, keyframes = model.generate_video(prompt, width=keyframe_width, height=keyframe_height, num_return_images=1, video_width=video_width, video_height=video_height, guidance_scale_for_llm=guidance_scale_for_llm,  guidance_scale_for_decoder=guidance_scale_for_decoder, num_inference_steps=num_inference_steps, top_k=top_k,)
        print(f'videos{videos}')
        data = [torch.unsqueeze(torch.tensor(np.array(image).astype(np.float32) / 255.0), 0) for image in videos[0]]
        return torch.cat(tuple(data), dim=0).unsqueeze(0)

class VideoLaVITI2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":("VideoLaVIT",),
                "prompt":("STRING",{"multiline": True, "default":"FPV drone footage of an ancient city in autumn"}),
                "image":("IMAGE",),
                "video_width":("INT",{"default":576}),
                "video_height":("INT",{"default":320}),
                "guidance_scale_for_llm":("FLOAT",{"default":4.0}),
                "num_inference_steps":("INT",{"default":50}),
                "top_k":("INT",{"default":50}),
                "seed":("INT",{"default":16}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "LaVIT"

    def run(self,model,prompt,image,video_width,video_height,guidance_scale_for_llm,num_inference_steps,top_k,seed):
    
        image = 255.0 * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).convert('RGB')
        random.seed(seed)
        torch.manual_seed(seed)
        model_dtype='fp16'
        torch_dtype = torch.bfloat16 if model_dtype=="bf16" else torch.float16
        input_prompts = [(prompt, 'text'), (image, 'image')]
        with torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
            videos, _ = model.multimodal_video_generate(input_prompts, video_width=video_width, video_height=video_height, 
                    guidance_scale_for_llm=guidance_scale_for_llm, top_k=top_k,)
        data = [torch.unsqueeze(torch.tensor(np.array(image).astype(np.float32) / 255.0), 0) for image in videos[0]]
        return torch.cat(tuple(data), dim=0).unsqueeze(0)

class VideoLaVITI2VLong:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":("VideoLaVIT",),
                "prompt":("STRING",{"multiline": True, "default":"FPV drone footage of an ancient city in autumn"}),
                "image":("IMAGE",),
                "clip_num":("INT",{"default":2}),
                "video_width":("INT",{"default":576}),
                "video_height":("INT",{"default":320}),
                "guidance_scale_for_llm":("FLOAT",{"default":4.0}),
                "num_inference_steps":("INT",{"default":50}),
                "top_k":("INT",{"default":50}),
                "seed":("INT",{"default":16}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "LaVIT"

    def run(self,model,prompt,image,clip_num,video_width,video_height,guidance_scale_for_llm,num_inference_steps,top_k,seed):
    
        image = 255.0 * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).convert('RGB')
        random.seed(seed)
        torch.manual_seed(seed)
        model_dtype='fp16'
        torch_dtype = torch.bfloat16 if model_dtype=="bf16" else torch.float16
        input_prompts = [(prompt, 'text'), (image, 'image')]
        with torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
            videos, _ = model.multimodal_video_generate(input_prompts, video_width=video_width, video_height=video_height, 
                    guidance_scale_for_llm=guidance_scale_for_llm, top_k=top_k, clip_num=clip_num)

        clip_videos = videos[0][:24]
        for i_clip in range(1, clip_num):
            clip_videos += videos[0][i_clip * 24 + 1:i_clip * 24 + 24]
        data = [torch.unsqueeze(torch.tensor(np.array(image).astype(np.float32) / 255.0), 0) for image in clip_videos]
        return torch.cat(tuple(data), dim=0).unsqueeze(0)

class VideoLaVITT2VLong:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":("VideoLaVIT",),
                "prompt":("STRING",{"multiline": True, "default":"FPV drone footage of an ancient city in autumn"}),
                "clip_num":("INT",{"default":2}),
                "keyframe_width":("INT",{"default":1024}),
                "keyframe_height":("INT",{"default":576}),
                "video_width":("INT",{"default":576}),
                "video_height":("INT",{"default":320}),
                "guidance_scale_for_llm":("FLOAT",{"default":4.0}),
                "guidance_scale_for_decoder":("FLOAT",{"default":7.0}),
                "num_inference_steps":("INT",{"default":50}),
                "top_k":("INT",{"default":50}),
                "inverse_rate":("FLOAT",{"default":0.9}),
                "seed":("INT",{"default":16}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "LaVIT"

    def run(self,model,prompt,clip_num,keyframe_width,keyframe_height,video_width,video_height,guidance_scale_for_llm,guidance_scale_for_decoder,num_inference_steps,top_k,inverse_rate,seed):
        random.seed(seed)
        torch.manual_seed(seed)
        model_dtype='fp16'
        torch_dtype = torch.bfloat16 if model_dtype=="bf16" else torch.float16

        with torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
            videos, keyframes = model.generate_video(prompt, width=keyframe_width, height=keyframe_height, num_return_images=1, video_width=video_width, video_height=video_height, guidance_scale_for_llm=guidance_scale_for_llm,  guidance_scale_for_decoder=guidance_scale_for_decoder, num_inference_steps=num_inference_steps, top_k=top_k, clip_num=clip_num, inverse_rate=inverse_rate,)

        clip_videos = videos[0][:24]
        for i_clip in range(1, clip_num):
            clip_videos += videos[0][i_clip * 24 + 1:i_clip * 24 + 24]
        data = [torch.unsqueeze(torch.tensor(np.array(image).astype(np.float32) / 255.0), 0) for image in clip_videos]
        return torch.cat(tuple(data), dim=0).unsqueeze(0)

class VideoLaVITI2I:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":("VideoLaVIT",),
                "prompt":("STRING",{"multiline": True, "default":"FPV drone footage of an ancient city in autumn"}),
                "width":("INT",{"default":1024}),
                "height":("INT",{"default":576}),
                "guidance_scale_for_llm":("FLOAT",{"default":4.0}),
                "num_inference_steps":("INT",{"default":50}),
                "top_k":("INT",{"default":50}),
                "temperature":("FLOAT",{"default":1.0}),
                "seed":("INT",{"default":16}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "LaVIT"

    def run(self,model,prompt,width,height,guidance_scale_for_llm,num_inference_steps,top_k,temperature,seed):
    
        random.seed(seed)
        torch.manual_seed(seed)
        model_dtype='fp16'
        torch_dtype = torch.bfloat16 if model_dtype=="bf16" else torch.float16
        with torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
            images = model.generate_image(prompt, width=width, height=height, num_return_images=1, 
                guidance_scale_for_llm=guidance_scale_for_llm, guidance_scale_for_decoder=guidance_scale_for_decoder, num_inference_steps=num_inference_steps, top_k=top_k, temperature=1.0)
        data = [torch.unsqueeze(torch.tensor(np.array(image).astype(np.float32) / 255.0), 0) for image in images]
        return torch.cat(tuple(data), dim=0).unsqueeze(0)

NODE_CLASS_MAPPINGS = {
    "VideoLaVITLoader":VideoLaVITLoader,
    "VideoLaVITT2V":VideoLaVITT2V,
    "VideoLaVITI2V":VideoLaVITI2V,
    "VideoLaVITI2VLong":VideoLaVITI2VLong,
    "VideoLaVITT2VLong":VideoLaVITT2VLong,
    "VideoLaVITI2I":VideoLaVITI2I,
}
