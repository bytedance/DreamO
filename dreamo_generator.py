# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import cv2
import sys
import numpy as np
import torch
import time  # Added for profiling
from PIL import Image
from huggingface_hub import hf_hub_download
from torchvision.transforms.functional import normalize

from tools import BEN2
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from dreamo.dreamo_pipeline import DreamOPipeline
from dreamo.utils import (
    get_device,
    img2tensor,
    tensor2img,
    resize_numpy_image_area,
    resize_numpy_image_long,
    )


class Generator:
    def __init__(self, **args):
        
        # Parsing parameters
        self.version = args.pop("version", "v1.1")
        self.offload = args.pop("offload", True) 
        self.no_turbo = args.pop("no_turbo", False)
        self.quant = args.pop("quant", "none") # Quantize to use: none, int8, nunchaku
        self.device = get_device(args.pop("device", "auto")) # Device to use: auto, cuda, mps, or cpu
        print(f"Using device: {self.device} offload: {self.offload} quant: {self.quant}")
        
        # preprocessing models
        # background remove model: BEN2
        self.bg_rm_model = BEN2.BEN_Base().to(self.device).eval()
        hf_hub_download(repo_id='PramaLLC/BEN2', filename='BEN2_Base.pth', local_dir='models')
        self.bg_rm_model.loadcheckpoints('models/BEN2_Base.pth')
        # face crop and align tool: facexlib
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=self.device,
        )
        if self.offload:
            self.ben_to_device(torch.device('cpu'))
            self.facexlib_to_device(torch.device('cpu'))
            
        # Loading and initializing the dreamo pipeline.
        # If there is a new optimization solution, you can add it here.
        self.dreamo_pipeline:DreamOPipeline = None
        if self.quant == "nunchaku":
            self.dreamo_pipeline_init_nunchaku() # nunchaku,  offload 6.5GB VRAM.
        else:
            self.dreamo_pipeline_init_default() # default or int8, offload 24GB or 16GB VRAM.

            
    # Default full precision or int8 quantized inference
    def dreamo_pipeline_init_default(self):
        # load dreamo
        model_root = 'black-forest-labs/FLUX.1-dev'
        if os.path.exists(f'./models/{model_root}'):model_root = f'./models/{model_root}'
        self.dreamo_pipeline = DreamOPipeline.from_pretrained(model_root, torch_dtype=torch.bfloat16)
        self.dreamo_pipeline.load_dreamo_model(self.device, use_turbo=not self.no_turbo)
        # Quantize the model using int8, which may take some time
        if self.quant == "int8":
            # pip install optimum
            from optimum.quanto import freeze, qint8, quantize
            print('start quantize')
            quantize(self.dreamo_pipeline.transformer, qint8)
            freeze(self.dreamo_pipeline.transformer)
            quantize(self.dreamo_pipeline.text_encoder_2, qint8)
            freeze(self.dreamo_pipeline.text_encoder_2)
            print('done quantize')
        self.dreamo_pipeline = self.dreamo_pipeline.to(self.device)
        if self.offload:
            self.dreamo_pipeline.enable_model_cpu_offload()
            self.dreamo_pipeline.offload = True
        else:
            self.dreamo_pipeline.offload = False
        
        print(f"\n[Profiler] pipeline init done ({self.quant}).")
        pass
        
    # nunchaku support by juntaosun
    def dreamo_pipeline_init_nunchaku(self):
        """ Description: Use Nunchaku to achieve 2-4 times faster inference and <7GB low VRAM usage.
            Reference speed: 3080 graphics card 1024x1024, 12 steps, 15-20s to generate.  
        """
        try:
            # 依赖检测：The current version has been tested: nunchaku v0.3.x
            from nunchaku import NunchakuFluxTransformer2dModel
            from nunchaku.caching.diffusers_adapters.flux import apply_cache_on_pipe # flux
            from nunchaku.utils import get_precision
        except Exception as e:
            message = "\n--------------------------------------------------------------------\n"
            message += f"👉️ 缺失 nunchaku，请前往链接安装 (Missing nunchaku, please go to the link to install it):\n"
            message += f"https://github.com/mit-han-lab/nunchaku/releases/\n"
            message += "--------------------------------------------------------------------\n"
            raise ValueError(message)
    
        
        print("===================== Nunchaku =====================")
        # download models and load file (~7GB)
        precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
        svdq_filename = f"svdq-{precision}_r32-flux.1-dev.safetensors"
        hf_hub_download(repo_id='mit-han-lab/nunchaku-flux.1-dev', filename=svdq_filename, local_dir='models')
        transformer:NunchakuFluxTransformer2dModel = NunchakuFluxTransformer2dModel.from_pretrained(
            f"models/{svdq_filename}",
            offload=self.offload,
        )
        transformer.set_attention_impl("nunchaku-fp16")  # set attention implementation to fp16
        
        print("\n[Profiler] Loading DreamOPipeline from pretrained (FLUX base)...")
        dreamo_pipeline_load_start_time = time.time()
        model_root = 'black-forest-labs/FLUX.1-dev'
        if os.path.exists(f'./models/{model_root}'):model_root = f'./models/{model_root}' # 兼容目录 ./models
        self.dreamo_pipeline:DreamOPipeline = DreamOPipeline.from_pretrained(model_root, transformer=transformer, torch_dtype=torch.bfloat16)
        print(f"[Profiler] DreamOPipeline (FLUX base) loaded in {time.time() - dreamo_pipeline_load_start_time:.2f} seconds.")
        
        print(f"\n[Profiler] Loading DreamO specific models into pipeline... version: {self.version}")
        dreamo_specific_load_start_time = time.time()
        self.dreamo_pipeline.load_dreamo_model_nunchaku(self.device, use_turbo=not self.no_turbo, version=self.version)
        print(f"[Profiler] DreamO specific models loaded in {time.time() - dreamo_specific_load_start_time:.2f} seconds.")
            
        print(f"\n[Profiler] Moving final DreamOPipeline to device ({self.device})...")
        to_device_start_time = time.time()
        apply_cache_on_pipe(self.dreamo_pipeline, residual_diff_threshold=0.05)
        if self.offload:
            self.dreamo_pipeline.enable_sequential_cpu_offload()
            self.dreamo_pipeline.offload = True
        else:
            self.dreamo_pipeline.offload = False
            self.dreamo_pipeline.to(self.device)
        print(f"[Profiler] DreamOPipeline moved to device (with explicit component moves) in {time.time() - to_device_start_time:.2f} seconds.")
            
        self.dreamo_pipeline.enable_attention_slicing()
        self.dreamo_pipeline.enable_vae_tiling()

        print(f"\n[Profiler] Total Generator initialization ({self.quant}).")

    def ben_to_device(self, device):
        self.bg_rm_model.to(device)

    def facexlib_to_device(self, device):
        self.face_helper.face_det.to(device)
        self.face_helper.face_parse.to(device)

    @torch.no_grad()
    def get_align_face(self, img):
        # the face preprocessing code is same as PuLID
        self.face_helper.clean_all()
        image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.face_helper.read_image(image_bgr)
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0:
            return None
        align_face = self.face_helper.cropped_faces[0]

        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
        input = input.to(self.device)
        parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input)
        # only keep the face features
        face_features_image = torch.where(bg, white_image, input)
        face_features_image = tensor2img(face_features_image, rgb2bgr=False)

        return face_features_image
    
    # Input condition preprocessing
    def pre_condition(self, ref_images, ref_tasks, ref_res, seed):
        ref_conds = []
        debug_images = []
        for idx, (ref_image, ref_task) in enumerate(zip(ref_images, ref_tasks)):
            if ref_image is not None:
                if ref_task == "id":
                    if self.offload:
                        self.facexlib_to_device(self.device)
                    ref_image = resize_numpy_image_long(ref_image, 1024)
                    ref_image = self.get_align_face(ref_image)
                    if self.offload:
                        self.facexlib_to_device(torch.device('cpu'))
                elif ref_task != "style":
                    if self.offload:
                        self.ben_to_device(self.device)
                    ref_image = self.bg_rm_model.inference(Image.fromarray(ref_image))
                    if self.offload:
                        self.ben_to_device(torch.device('cpu'))
                if ref_task != "id":
                    ref_image = resize_numpy_image_area(np.array(ref_image), ref_res * ref_res)
                debug_images.append(ref_image)
                ref_image = img2tensor(ref_image, bgr2rgb=False).unsqueeze(0) / 255.0
                ref_image = 2 * ref_image - 1.0
                ref_conds.append(
                    {
                        'img': ref_image,
                        'task': ref_task,
                        'idx': idx + 1,
                    }
                )
        # cleaning
        self.torch_empty_cache()
        seed = torch.Generator(device="cpu").seed() if seed == "-1" else int(seed)
        return ref_conds, debug_images, seed
    
    # Try cleaning the GPU
    def torch_empty_cache(self):
        try:
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()
            elif "mps" in str(self.device):
                torch.mps.empty_cache()
        except Exception as e:
            pass
