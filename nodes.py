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

class VideoLaVITUnderstandingLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (os.listdir(os.path.join(folder_paths.models_dir,"diffusers")), {"default": "Video-LaVIT-v1"}),
                "max_video_clips":("INT",{"default":16}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("VideoLaVITUnderstanding",)
    FUNCTION = "run"
    CATEGORY = "LaVIT"

    def run(self,model_path,max_video_clips):
        model_path=os.path.join(os.path.join(os.path.join(folder_paths.models_dir,"diffusers"),model_path),"language_model_sft")
        model_dtype='bf16'
        
        device_id = 0
        torch.cuda.set_device(device_id)
        device = torch.device('cuda')

        # For Multi-Modal Understanding
        runner = build_model(model_path=model_path, model_dtype=model_dtype, understanding=True, 
                device_id=device_id, use_xformers=True, max_video_clips=max_video_clips,)

        return (runner,)

class VideoLaVITUnderstandingImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":("VideoLaVITUnderstanding",),
                "image":("IMAGE",),
                "prompt":("STRING",{"multiline": True, "default":"What is unusual about this image?"}),
                "max_length":("INT",{"default":512}),
                "length_penalty":("INT",{"default":1}),
                "temperature":("FLOAT",{"default":1.0}),
                "seed":("INT",{"default":16}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "LaVIT"

    def run(self,model,image,prompt,max_length,length_penalty,temperature,seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        model_dtype='bf16'
        torch_dtype = torch.bfloat16 if model_dtype=="bf16" else torch.float16
        with torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
            image = 255.0 * image[0].cpu().numpy()
            image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).convert('RGB')
            image = model.image_processer(image)
            

            output = model({"image": image.unsqueeze(0), "text_input": prompt}, length_penalty=length_penalty, temperature=temperature,  use_nucleus_sampling=False, num_beams=1, truct_vqa=False, max_length=max_length,)[0]

            return (output,)

class VideoLaVITUnderstandingVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":("VideoLaVITUnderstanding",),
                "video_path":("STRING",{"multiline": True,}),
                "prompt":("STRING",{"multiline": True, "default":"What is the man doing in this video?"}),
                "max_length":("INT",{"default":512}),
                "length_penalty":("INT",{"default":1}),
                "temperature":("FLOAT",{"default":1.0}),
                "seed":("INT",{"default":16}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "LaVIT"

    def run(self,model,video_path,prompt,max_length,length_penalty,temperature,seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model_dtype='bf16'
        torch_dtype = torch.bfloat16 if model_dtype=="bf16" else torch.float16
        with torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
            #video_inputs_list = []
            #visual_inputs, motion_inputs = model.video_processor(video_path)
            #video_inputs_list.append(((visual_inputs, motion_inputs), 'video'))

            output = model({"video": video_path, "text_input": prompt}, length_penalty=length_penalty, use_nucleus_sampling=False, num_beams=1, max_length=max_length, temperature=temperature)[0]

            return (output,)

class VideoLaVITVideoDetokenizerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (os.listdir(os.path.join(folder_paths.models_dir,"diffusers")), {"default": "Video-LaVIT-v1"}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("VideoLaVITVideoDetokenizer",)
    FUNCTION = "run"
    CATEGORY = "LaVIT"

    def run(self,model_path):
        from .VideoLaVIT.models import build_video_detokenizer
        
        model_path=os.path.join(os.path.join(folder_paths.models_dir,"diffusers"),model_path)
        detokenizer_weight = os.path.join(model_path, 'video_3d_unet.bin')

        device_id = 0
        torch.cuda.set_device(device_id)

        model = build_video_detokenizer(model_path, model_dtype='fp16', pretrained_weight=detokenizer_weight)
        model = model.to("cuda")

        return (model,)

def sample_video_clips(video_path,width,height,max_frames):
    from .VideoLaVIT.models.transform import MotionVectorProcessor, extract_motions
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode

    motion_transform = MotionVectorProcessor(width=width // 8, height=height // 8)

    pil_transform = [
        transforms.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
    ]
    pil_transform = transforms.Compose(pil_transform)
    image_transform = [
        transforms.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ]
    image_transform = transforms.Compose(image_transform)

    frames, motions, frame_types = extract_motions(video_path, raw_file=True, temp_dir=folder_paths.temp_directory, fps=12)
    total_frames = len(frame_types)
    start_indexs = np.where(np.array(frame_types)=='I')[0]
    
    if len(start_indexs) == 0:
        raise ValueError(f"Empty Start indexs: {video_path}")

    # FIlter one I-Frame + 11 P-Frame
    if len(start_indexs) > 1:
        end_indexs = start_indexs + 12
        filter_start_indexs = start_indexs[:-1][end_indexs[:-1] == start_indexs[1:]]    
    else:
        filter_start_indexs = start_indexs

    # FIlter the frames that exceed the max frames
    filter_start_indexs = filter_start_indexs[filter_start_indexs + max_frames <= total_frames]

    if len(filter_start_indexs) > 0:
        # Stack the motions
        start_index = np.random.choice(filter_start_indexs)
        indices = np.arange(start_index, start_index + max_frames)
        motions = [torch.from_numpy(motions[i].transpose((2,0,1))) for i in indices]
        motions = torch.stack(motions).float()
        motions = motion_transform(motions)
        filtered_frames = [Image.fromarray(frames[i]).convert("RGB") for i in indices]
        pil_frames = [pil_transform(frame) for frame in filtered_frames]
        frame_tensors = [image_transform(frame) for frame in filtered_frames]
        frame_tensors = torch.stack(frame_tensors)
        frame_tensors = 2.0 * frame_tensors - 1.0
        return pil_frames, frame_tensors, motions

    else:
        raise ValueError(f"Empty Filtered Start indexs: {video_path}")

class VideoLaVITVideoReconstruction:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":("VideoLaVITVideoDetokenizer",),
                "video_path":("STRING",{"multiline": True,}),
                "width":("INT",{"default":576}),
                "height":("INT",{"default":320}),
                "decode_chunk_size":("INT",{"default":1}),
                "max_frames":("INT",{"default":24}),
                "noise_aug_strength":("FLOAT",{"default":0.02}),
                "use_linear_guidance":("BOOLEAN",{"default":True}),
                "max_guidance_scale":("FLOAT",{"default":3.0}),
                "min_guidance_scale":("FLOAT",{"default":1.0}),
                "seed":("INT",{"default":16}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "LaVIT"

    def run(self,model,video_path,width,height,decode_chunk_size,max_frames,noise_aug_strength,use_linear_guidance,max_guidance_scale,min_guidance_scale,seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        video_frames, video_tensors, motions = sample_video_clips(video_path,width,height,max_frames)

        keyframe = video_tensors[0:1]
        motions = motions.unsqueeze(0)

        model_dtype='fp16'
        torch_dtype = torch.bfloat16 if model_dtype=="bf16" else torch.float16
        with torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
            frames =  model.reconstruct_from_token(keyframe.to("cuda"), motions.to("cuda"), decode_chunk_size=8, 
            width=width, height=height, num_frames=max_frames, noise_aug_strength=0.02, cond_on_ref_frame=True, 
            use_linear_guidance=use_linear_guidance, max_guidance_scale=max_guidance_scale, min_guidance_scale=min_guidance_scale,)[0]

            data = [torch.unsqueeze(torch.tensor(np.array(image).astype(np.float32) / 255.0), 0) for image in frames]
            return torch.cat(tuple(data), dim=0).unsqueeze(0)

class VHS_FILENAMES_STRING_LaVIT:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "filenames": ("VHS_FILENAMES",),
                    }
                }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "StreamingT2V"
    FUNCTION = "run"

    def run(self, filenames):
        return (filenames[1][-1],)

NODE_CLASS_MAPPINGS = {
    "VideoLaVITLoader":VideoLaVITLoader,
    "VideoLaVITT2V":VideoLaVITT2V,
    "VideoLaVITI2V":VideoLaVITI2V,
    "VideoLaVITI2VLong":VideoLaVITI2VLong,
    "VideoLaVITT2VLong":VideoLaVITT2VLong,
    "VideoLaVITI2I":VideoLaVITI2I,
    "VideoLaVITUnderstandingLoader":VideoLaVITUnderstandingLoader,
    "VideoLaVITUnderstandingImage":VideoLaVITUnderstandingImage,
    "VideoLaVITUnderstandingVideo":VideoLaVITUnderstandingVideo,
    "VideoLaVITVideoDetokenizerLoader":VideoLaVITVideoDetokenizerLoader,
    "VideoLaVITVideoReconstruction":VideoLaVITVideoReconstruction,
    "VHS_FILENAMES_STRING_LaVIT":VHS_FILENAMES_STRING_LaVIT,
}
