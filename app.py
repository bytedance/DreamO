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

import torch
import gradio as gr
from dreamo_generator import Generator

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--version', type=str, default='v1.1', choices=['v1.1', 'v1'],
                    help='default will use the latest v1.1 model, you can also switch back to v1')
parser.add_argument('--offload', action='store_true', help="Enable 'quant=nunchaku' and 'offload' to reduce the original 24GB VRAM to 6.5GB.")
parser.add_argument('--no_turbo', action='store_true', help='Use turbo to reduce the original 25 steps to 12 steps.')
parser.add_argument('--quant', type=str, default='none', choices=['none', 'int8', 'nunchaku'],
                    help='Quantize to use: none(bf16), int8, nunchaku')
parser.add_argument('--device', type=str, default='auto', help='Device to use: auto, cuda, mps, or cpu')
args = parser.parse_args()


# ✨️ 温馨提示：消费级 16G 以下显卡, 建议启用 'quant=nunchaku' 。低显存 2-4 倍快速推理，小于 20 秒生成图像（1024x1024）~
# -------------------------------------------------------------------------------------------------------------------
# Tips: For consumer-grade graphics cards below 16G, it is recommended to enable quant=nunchaku.
# 2-4 times faster reasoning with low VRAM, generating images（1024x1024） in less than 20 seconds~
# -------------------------------------------------------------------------------------------------------------------

# 👉️ Parameter Description: quant = 'nunchaku', no_turbo = False, offload = True
# -------------------------------------------------------------------------------------------------------------------
# [ no_turbo = False ]:  Use 'turbo' to reduce the original 25 steps to 12 steps.
# [ offload = True   ]:  Enable 'quant=nunchaku' and 'offload' to reduce the original 24GB VRAM to 6.5GB.
# -------------------------------------------------------------------------------------------------------------------

# 👉️ Inference VRAM usage: For example, NVIDIA RTX 3080-10G 
# -------------------------------------------------------------------------------------------------------------------
# [ quant= 'none' ]:  offload, 24GB  VRAM.  ⚠️ CUDA out of memory.
# [ quant= 'int8'    ]:  offload, 16GB  VRAM.  ⚠️ CUDA out of memory.
# [ quant= 'nunchaku']:  offload, 6.5GB VRAM.  ✅️ Working fine! So it supports consumer-grade GPUs (8GB or higher) now.
# -------------------------------------------------------------------------------------------------------------------

# DreamO Generator
generator = Generator(**vars(args))


@torch.inference_mode()
def generate_image(
    ref_image1,
    ref_image2,
    ref_task1,
    ref_task2,
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
    ref_conds, debug_images, seed = generator.pre_condition(
        ref_images=[ref_image1, ref_image2],
        ref_tasks=[ref_task1, ref_task2],
        ref_res=ref_res,
        seed=seed,
        )
    print(prompt, seed)

    print("start dreamo_pipeline... ")
    image = generator.dreamo_pipeline(
        prompt=prompt,
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

    return image, debug_images, seed


_HEADER_ = '''
<div style="text-align: center; max-width: 650px; margin: 0 auto;">
    <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem; display: contents;">DreamO</h1>
    <p style="font-size: 1rem; margin-bottom: 1.5rem;">Paper: <a href='https://arxiv.org/abs/2504.16915' target='_blank'>DreamO: A Unified Framework for Image Customization</a> | Codes: <a href='https://github.com/bytedance/DreamO' target='_blank'>GitHub</a></p>
</div>

🚩 Update Notes:
- 2025.06.26: Use Nunchaku to achieve <7GB VRAM inference and 2-4 times faster inference. Contributed by juntaosun.  
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
                    ref_image1 = gr.Image(label="ref image 1", type="numpy", height=256)
                    ref_image2 = gr.Image(label="ref image 2", type="numpy", height=256)
                with gr.Row():
                    with gr.Group():
                        ref_task1 = gr.Dropdown(choices=["ip", "id", "style"], value="ip", label="task for ref image 1")
                    with gr.Group():
                        ref_task2 = gr.Dropdown(choices=["ip", "id", "style"], value="ip", label="task for ref image 2")
                prompt = gr.Textbox(label="Prompt", value="a person playing guitar in the street")
                generate_btn = gr.Button("🎉 Generate")
                
                width = gr.Slider(768, 1024, 1024, step=16, label="Width")
                height = gr.Slider(768, 1024, 1024, step=16, label="Height")
                num_steps = gr.Slider(8, 30, 12, step=1, label="Number of steps")
                guidance = gr.Slider(1.0, 10.0, 4.5 if args.version == 'v1.1' else 3.5, step=0.1, label="Guidance")
                seed = gr.Textbox(label="Seed (-1 for random)", value="-1")
                ref_res = gr.Slider(512, 1024, 512, step=16, label="resolution for ref image, increase it if necessary")
                with gr.Accordion("Advanced Options", open=False, visible=False):
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
                ref_task1,
                ref_task2,
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
