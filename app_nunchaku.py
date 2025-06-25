# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
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

# import os
# os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

import argparse
import sys
import time  # Added for profiling
import cv2
import numpy as np
import torch



# 注意：nunchaku 一定要放置在 import gradio 之前
# https://huggingface.co/mit-han-lab/nunchaku/tree/main
try:
    from nunchaku import NunchakuFluxTransformer2dModel
    from nunchaku.caching.diffusers_adapters.flux import apply_cache_on_pipe # flux
    from nunchaku.utils import get_precision
except Exception as e:
    torch_v = '.'.join(torch.__version__.split(".")[:2])
    python_v = ''.join(sys.version.split(".")[:2])
    # 最新版本下载地址 https://github.com/mit-han-lab/nunchaku/releases/download/v0.3.1/
    install_desc = f"pip install https://github.com/mit-han-lab/nunchaku/releases/download/v0.3.1/nunchaku-0.3.1+torch{torch_v}-cp{python_v}-cp{python_v}-win_amd64.whl" # Windows
    message = "\n--------------------------------------------------------------------\n"
    message += f"你需要安装 Nunchaku (You need to install nunchaku):\n"
    message += f"{install_desc}\n"
    message += "--------------------------------------------------------------------\n"
    raise ValueError(message)



import gradio as gr
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from huggingface_hub import hf_hub_download
from PIL import Image

from dreamo.dreamo_pipeline import DreamOPipeline
from dreamo.utils import (
    img2tensor,
    resize_numpy_image_area,
    resize_numpy_image_long,
    tensor2img,
)
from tools import BEN2



parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--version', type=str, default='v1.1')
parser.add_argument('--no_turbo', action='store_true')
parser.add_argument('--offload', action='store_true')
parser.add_argument('--device', type=str, default='auto', help='Device to use: auto, cuda, mps, or cpu')
args = parser.parse_args()

# -----------------------------------------------------------------------------------------------------------------
# 说明：使用 Nunchaku 实现2~4倍高速推理 <~7GB 低显存占用。by juntaosun
# Description: Use Nunchaku to achieve 2~4 times faster inference and <~7GB low VRAM usage. by juntaosun
# -----------------------------------------------------------------------------------------------------------------


def get_device():
    """Automatically detect the best available device"""
    if args.device != 'auto':
        return torch.device(args.device)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

class Generator:
    def __init__(self):
        overall_start_time = time.time()
        
        device = get_device()
        print(f"Using device: {device}")

        # Profile: Background removal model
        print("\n[Profiler] Initializing background removal model...")
        bg_model_start_time = time.time()
        self.bg_rm_model = BEN2.BEN_Base().to(device).eval()
        hf_hub_download(repo_id='PramaLLC/BEN2', filename='BEN2_Base.pth', local_dir='models')
        self.bg_rm_model.loadcheckpoints('models/BEN2_Base.pth')
        print(f"[Profiler] Background removal model initialized in {time.time() - bg_model_start_time:.2f} seconds.")

        # Profile: FaceRestoreHelper
        print("\n[Profiler] Initializing FaceRestoreHelper...")
        face_helper_start_time = time.time()
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=device,
        )
        print(f"[Profiler] FaceRestoreHelper initialized in {time.time() - face_helper_start_time:.2f} seconds.")

        if args.offload:
            # Offloading these components if enabled (timing included above for their init)
            self.ben_to_device(torch.device('cpu'))
            self.facexlib_to_device(torch.device('cpu'))
            
            
        # --- DreamOPipeline Loading Logic ---
            
        print("===================== Nunchaku =====================")
        
        
        # download models and load file (~7GB)
        precision = get_precision()  # auto-detect your precision is 'int4' or 'fp4' based on your GPU
        svdq_filename = f"svdq-{precision}_r32-flux.1-dev.safetensors"
        hf_hub_download(repo_id='mit-han-lab/nunchaku-flux.1-dev', filename=svdq_filename, local_dir='models')
        transformer:NunchakuFluxTransformer2dModel = NunchakuFluxTransformer2dModel.from_pretrained(
            f"models/{svdq_filename}",
            offload=True,
        )
        
        
        print("\n[Profiler] Full Pipeline Preload is DISABLED. Using original loading logic.")
        print("\n[Profiler] Loading DreamOPipeline from pretrained (FLUX base)...")
        dreamo_pipeline_load_start_time = time.time()
        # model_root = './models/black-forest-labs/FLUX.1-dev'
        model_root = 'black-forest-labs/FLUX.1-dev'
        self.dreamo_pipeline:DreamOPipeline = DreamOPipeline.from_pretrained(model_root, transformer=transformer, torch_dtype=torch.bfloat16)
        print(f"[Profiler] DreamOPipeline (FLUX base) loaded in {time.time() - dreamo_pipeline_load_start_time:.2f} seconds.")
        
        print(f"\n[Profiler] Loading DreamO specific models into pipeline... version: {args.version}")
        dreamo_specific_load_start_time = time.time()
        self.dreamo_pipeline.load_dreamo_model_nunchaku(device, use_turbo=not args.no_turbo, version=args.version)
        print(f"[Profiler] DreamO specific models loaded in {time.time() - dreamo_specific_load_start_time:.2f} seconds.")
            
        print(f"\n[Profiler] Moving final DreamOPipeline to device ({device})...")
        to_device_start_time = time.time()
        if args.offload:
            self.dreamo_pipeline.enable_model_cpu_offload()
            self.dreamo_pipeline.offload = True
        else:
            self.dreamo_pipeline.offload = False
        print(f"[Profiler] DreamOPipeline moved to device (with explicit component moves) in {time.time() - to_device_start_time:.2f} seconds.")
            
        
        #  参数用于控制是否启用注意力切片技术。当 GPU 显存有限时，启用此参数可以减少 VRAM 使用，但可能会牺牲一定的计算速度。
        self.dreamo_pipeline.enable_attention_slicing()
        self.dreamo_pipeline.enable_vae_tiling()
        
        # nunchaku CPU Offloading  CPU 卸载,初始化时设置 offload=True，然后调用：
        self.dreamo_pipeline.enable_sequential_cpu_offload()
        # nunchaku 块缓存,建议的值为 0.12，它为 50 级降噪提供高达 2× 的加速
        apply_cache_on_pipe(self.dreamo_pipeline, residual_diff_threshold=0.05)

        print(f"\n[Profiler] Total Generator initialization time: {time.time() - overall_start_time:.2f} seconds.")

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

        tensor_input = img2tensor(align_face, bgr2rgb=True)
        if isinstance(tensor_input, list):  # Ensure single tensor
            tensor_input = tensor_input[0]
        input = tensor_input.unsqueeze(0) / 255.0
        input = input.to(torch.device("cuda"))
        
        # Resize input to 512x512 for face parsing model
        # input shape is (N, C, H_orig, W_orig), e.g. (1, 3, H_orig, W_orig)
        resized_input_for_parsing = torch.nn.functional.interpolate(input, size=(512, 512), mode='bilinear', align_corners=False)
        # resized_input_for_parsing shape is (N, C, 512, 512)
        
        # parsing_logits_512 is assumed to be (N, num_classes, 512, 512), e.g. (1, 19, 512, 512)
        parsing_logits_512 = self.face_helper.face_parse(resized_input_for_parsing)[0]
        
        bg_label = [0, 16] # background, cloth (adjust labels as needed)
        
        # Get predicted segmentation class for each pixel: (N, 512, 512)
        pred_seg_512 = torch.argmax(parsing_logits_512, dim=1) # Use dim=1 for class dimension
        
        # Create a boolean mask (N, 512, 512) indicating background pixels
        bg_mask_512 = torch.zeros_like(pred_seg_512, dtype=torch.bool, device=input.device)
        for label_val in bg_label:
            bg_mask_512 = bg_mask_512 | (pred_seg_512 == label_val)
        
        # Get original spatial dimensions (H_orig, W_orig) from the input tensor
        H_orig = input.shape[2]
        W_orig = input.shape[3]
        
        # Resize the (N, 512, 512) mask to (N, H_orig, W_orig)
        # Add channel dim for interpolate: (N, 1, 512, 512)
        bg_mask_512_float_for_interp = bg_mask_512.float().unsqueeze(1)
        
        # Interpolate to original dimensions: (N, 1, H_orig, W_orig)
        bg_mask_orig_dims_float_batched = torch.nn.functional.interpolate(
            bg_mask_512_float_for_interp, size=(H_orig, W_orig), mode='nearest'
        )
        
        # Convert resized mask to boolean for torch.where. Shape (N, 1, H_orig, W_orig)
        bg_for_where = bg_mask_orig_dims_float_batched.bool()
        
        white_image = torch.ones_like(input)
        
        # Replace original image's background pixels (where bg_for_where is True) with white
        # input is (N, C_orig, H_orig, W_orig)
        # bg_for_where (N, 1, H_orig, W_orig) will broadcast with C_orig dimension.
        face_features_image_tensor = torch.where(bg_for_where, white_image, input)
        
        face_features_image = tensor2img(face_features_image_tensor, rgb2bgr=False)

        return face_features_image
    
    # 释放GPU
    def torch_empty_cache(self):
        try:
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()
            elif "mps" in str(self.device):
                torch.mps.empty_cache()
        except Exception as e:
            pass
            


generator = Generator()


# 参考图：虽然可支持多达 3 个参考图，但根据官方建议，condition 数量增多也会影响稳定性，推荐 2 个效果最佳。
# Reference images: Although up to 3 reference images can be supported, according to official recommendations, 
# an increase in the number of conditions will also affect stability, so 2 is recommended for best results.

@torch.inference_mode()
def generate_image(
    ref_image1,
    ref_image2,
    ref_image3,
    ref_task1,
    ref_task2,
    ref_task3,
    prompt,
    width,
    height,
    ref_res,
    num_steps,
    guidance,
    seed,
    true_cfg,
    cfg_start_step,
    cfg_end_step,
    neg_prompt,
    neg_guidance,
    first_step_guidance,
):
    print(prompt)
    ref_conds = []
    debug_images = []

    ref_images = [ref_image1, ref_image2, ref_image3]
    ref_tasks = [ref_task1, ref_task2, ref_task3]

    for idx, (ref_image, ref_task) in enumerate(zip(ref_images, ref_tasks)):
        if ref_image is not None:
            if ref_task == "id":
                if args.offload:
                    generator.facexlib_to_device(torch.device('cuda'))
                ref_image = resize_numpy_image_long(ref_image, 1024)
                ref_image = generator.get_align_face(ref_image)
                if args.offload:
                    generator.facexlib_to_device(torch.device('cpu'))
            elif ref_task != "style":
                if args.offload:
                    generator.ben_to_device(torch.device('cuda'))
                ref_image = generator.bg_rm_model.inference(Image.fromarray(ref_image))
                if args.offload:
                    generator.ben_to_device(torch.device('cpu'))
            if ref_task != "id":
                ref_image = resize_numpy_image_area(np.array(ref_image), ref_res * ref_res)
            debug_images.append(ref_image)

            tensor_ref_image = img2tensor(ref_image, bgr2rgb=False)
            if isinstance(tensor_ref_image, list):  # Ensure single tensor
                tensor_ref_image = tensor_ref_image[0]
            ref_image = tensor_ref_image.unsqueeze(0) / 255.0
            ref_image = 2 * ref_image - 1.0
            ref_conds.append(
                {
                    'img': ref_image,
                    'task': ref_task,
                    'idx': idx + 1,
                }
            )
            
    # 节省 GPU 资源
    generator.facexlib_to_device(torch.device('cpu'))
    generator.ben_to_device(torch.device('cpu'))
    generator.torch_empty_cache()

    seed = int(seed)
    if seed == -1:
        seed = torch.Generator(device="cpu").seed()
        
        
    print("start dreamo_pipeline... ")
    image = generator.dreamo_pipeline(
        prompt=prompt, # 提示词
        width=width,
        height=height,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        ref_conds=ref_conds,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        true_cfg_scale=true_cfg,
        true_cfg_start_step=cfg_start_step,
        true_cfg_end_step=cfg_end_step,
        negative_prompt=neg_prompt,
        neg_guidance_scale=neg_guidance,
        first_step_guidance_scale=first_step_guidance if first_step_guidance > 0 else guidance,
    ).images[0]
    
    generator.torch_empty_cache()

    return image, debug_images, seed


_HEADER_ = '''
<div style="text-align: center; max-width: 650px; margin: 0 auto;">
    <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem; display: contents;">DreamO</h1>
    <p style="font-size: 1rem; margin-bottom: 1.5rem;">Paper: <a href='https://arxiv.org/abs/2504.16915' target='_blank'>DreamO: A Unified Framework for Image Customization</a> | Codes: <a href='https://github.com/bytedance/DreamO' target='_blank'>GitHub</a></p>
</div>

🚩 Update Notes:
- 2025.06.25: Use Nunchaku to achieve <~7GB VRAM inference and 2~4 times faster inference. by juntaosun.  
- 2025.06.24: Updated to v1.1 with significant improvements in image quality, reduced likelihood of body composition errors, and enhanced aesthetics. <a href='https://github.com/bytedance/DreamO/blob/main/dreamo_v1.1.md' target='_blank'>Learn more about this model</a>
- 2025.05.11: We have updated the model to mitigate over-saturation and plastic-face issues. The new version shows consistent improvements over the previous release.

❗️❗️❗️**User Guide:**
- The most important thing to do first is to try the examples provided below the demo, which will help you better understand the capabilities of the DreamO model and the types of tasks it currently supports
- For each input, please select the appropriate task type. For general objects, characters, or clothing, choose IP — we will remove the background from the input image. If you select ID, we will extract the face region from the input image (similar to PuLID). If you select Style, the background will be preserved, and you must prepend the prompt with the instruction: 'generate a same style image.' to activate the style task.
- To accelerate inference, we adopt FLUX-turbo LoRA, which reduces the sampling steps from 25 to 12 compared to FLUX-dev. Additionally, we distill a CFG LoRA, achieving nearly a twofold reduction in steps by eliminating the need for true CFG

'''  # noqa E501

_CITE_ = r"""
If DreamO is helpful, please help to ⭐ the <a href='https://github.com/bytedance/DreamO' target='_blank'> Github Repo</a>. Thanks!
---

📧 **Contact**
If you have any questions or feedbacks, feel free to open a discussion or contact <b>wuyanze123@gmail.com</b> and <b>eechongm@gmail.com</b>
"""  # noqa E501


def create_demo():

    with gr.Blocks() as demo:
        gr.Markdown(_HEADER_)

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    ref_image1 = gr.Image(label="ref image 1", type="numpy", height=256) # 参考图 1
                    ref_image2 = gr.Image(label="ref image 2", type="numpy", height=256) # 参考图 2
                    ref_image3 = gr.Image(label="ref image 3", type="numpy", height=256) # 参考图 3
                with gr.Row():
                    with gr.Group():
                        ref_task1 = gr.Dropdown(choices=["ip", "id", "style"], value="ip", label="task for ref image 1")
                    with gr.Group():
                        ref_task2 = gr.Dropdown(choices=["ip", "id", "style"], value="ip", label="task for ref image 2")
                    with gr.Group():
                        ref_task3 = gr.Dropdown(choices=["ip", "id", "style"], value="ip", label="task for ref image 3")
                prompt = gr.Textbox(label="Prompt", value="a person playing guitar in the street")
                generate_btn = gr.Button("Generate")
                
                width = gr.Slider(768, 1280, 1024, step=16, label="Width")
                height = gr.Slider(768, 1280, 1024, step=16, label="Height")
                num_steps = gr.Slider(8, 30, 12, step=1, label="Number of steps") # 默认步数 12
                guidance = gr.Slider(1.0, 10.0, 3.5 if args.version == 'v1.1' else 3.5, step=0.1, label="Guidance") # 建议 3.5
                seed = gr.Textbox(label="Seed (-1 for random)", value="-1")
                ref_res = gr.Slider(512, 1024, 512, step=16, label="resolution for ref image, increase it if necessary")
                with gr.Accordion("Advanced Options", open=False, visible=True):
                    neg_prompt = gr.Textbox(label="Neg Prompt", value="")
                    neg_guidance = gr.Slider(1.0, 10.0, 3.5, step=0.1, label="Neg Guidance")
                    true_cfg = gr.Slider(1, 5, 1, step=0.1, label="true cfg")
                    cfg_start_step = gr.Slider(0, 30, 0, step=1, label="cfg start step")
                    cfg_end_step = gr.Slider(0, 30, 0, step=1, label="cfg end step")
                    first_step_guidance = gr.Slider(0, 10, 0, step=0.1, label="first step guidance")
                gr.Markdown(_CITE_)

            with gr.Column():
                output_image = gr.Image(label="Generated Image", format='png')
                debug_image = gr.Gallery(
                    label="Preprocessing output (including possible face crop and background remove)",
                    elem_id="gallery",
                )
                seed_output = gr.Textbox(label="Used Seed")

        with gr.Row(), gr.Column():
            gr.Markdown("## Examples")
            example_inps = [
                [
                    'example_inputs/woman1.png',
                    'ip',
                    'profile shot dark photo of a 25-year-old female with smoke escaping from her mouth, the backlit smoke gives the image an ephemeral quality, natural face, natural eyebrows, natural skin texture, award winning photo, highly detailed face, atmospheric lighting, film grain, monochrome',  # noqa E501
                    9180879731249039735,
                ],
                [
                    'example_inputs/man1.png',
                    'ip',
                    'a man sitting on the cloud, playing guitar',
                    1206523688721442817,
                ],
                [
                    'example_inputs/toy1.png',
                    'ip',
                    'a purple toy holding a sign saying "DreamO", on the mountain',
                    10441727852953907380,
                ],
                [
                    'example_inputs/perfume.png',
                    'ip',
                    'a perfume under spotlight',
                    116150031980664704,
                ],
            ]
            gr.Examples(examples=example_inps, inputs=[ref_image1, ref_task1, prompt, seed], label='IP task')

            example_inps = [
                [
                    'example_inputs/hinton.jpeg',
                    'id',
                    'portrait, Chibi',
                    5443415087540486371,
                ],
            ]
            gr.Examples(
                examples=example_inps,
                inputs=[ref_image1, ref_task1, prompt, seed],
                label='ID task (similar to PuLID, will only refer to the face)',
            )

            example_inps = [
                [
                    'example_inputs/mickey.png',
                    'style',
                    'generate a same style image. A rooster wearing overalls.',
                    6245580464677124951,
                ],
                [
                    'example_inputs/mountain.png',
                    'style',
                    'generate a same style image. A pavilion by the river, and the distant mountains are endless',
                    5248066378927500767,
                ],
            ]
            gr.Examples(examples=example_inps, inputs=[ref_image1, ref_task1, prompt, seed], label='Style task')

            example_inps = [
                [
                    'example_inputs/shirt.png',
                    'example_inputs/skirt.jpeg',
                    'ip',
                    'ip',
                    'A girl is wearing a short-sleeved shirt and a short skirt on the beach.',
                    9514069256241143615,
                ],
                [
                    'example_inputs/woman2.png',
                    'example_inputs/dress.png',
                    'id',
                    'ip',
                    'the woman wearing a dress, In the banquet hall',
                    42,
                ],
            ]
            gr.Examples(
                examples=example_inps,
                inputs=[ref_image1, ref_image2, ref_task1, ref_task2, prompt, seed],
                label='Try-On task',
            )

            example_inps = [
                [
                    'example_inputs/dog1.png',
                    'example_inputs/dog2.png',
                    'ip',
                    'ip',
                    'two dogs in the jungle',
                    6187006025405083344,
                ],
                [
                    'example_inputs/woman3.png',
                    'example_inputs/cat.png',
                    'ip',
                    'ip',
                    'A girl rides a giant cat, walking in the noisy modern city. High definition, realistic, non-cartoonish. Excellent photography work, 8k high definition.',  # noqa E501
                    11980469406460273604,
                ],
                [
                    'example_inputs/man2.jpeg',
                    'example_inputs/woman4.jpeg',
                    'ip',
                    'ip',
                    'a man is dancing with a woman in the room',
                    42,
                ],
            ]
            gr.Examples(
                examples=example_inps,
                inputs=[ref_image1, ref_image2, ref_task1, ref_task2, prompt, seed],
                label='Multi IP',
            )

        generate_btn.click(
            fn=generate_image,
            inputs=[
                ref_image1,
                ref_image2,
                ref_image3,
                ref_task1,
                ref_task2,
                ref_task3,
                prompt,
                width,
                height,
                ref_res,
                num_steps,
                guidance,
                seed,
                true_cfg,
                cfg_start_step,
                cfg_end_step,
                neg_prompt,
                neg_guidance,
                first_step_guidance,
            ],
            outputs=[output_image, debug_image, seed_output],
        )

    return demo


if __name__ == '__main__':
    demo = create_demo()
    demo.queue().launch(server_name='0.0.0.0', server_port=args.port)
