# Prediction interface for Cog ⚙️
# https://cog.run/python

import os

# Set cache directories
MODEL_CACHE = "model_cache"
BASE_URL = "https://weights.replicate.delivery/default/dreamo-flux-dev/model_cache/"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

import mimetypes
mimetypes.add_type("image/webp", ".webp")

import cv2
import time
import torch
import subprocess
import numpy as np
from PIL import Image
from cog import BasePredictor, Input, Path
from huggingface_hub import hf_hub_download
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize

from dreamo.dreamo_pipeline import DreamOPipeline
from dreamo.utils import img2tensor, resize_numpy_image_area, tensor2img, resize_numpy_image_long
from tools import BEN2 # Assuming BEN2 is available in tools directory or installed

def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Create model cache directory if it doesn't exist
        os.makedirs(MODEL_CACHE, exist_ok=True)
            
        model_files = [
            "detection_Resnet50_Final.pth",
            "models--black-forest-labs--FLUX.1-dev.tar",
            "parsing_bisenet.pth",
        ]
        
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)
        
        print("Setting up predictor...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.torch_dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
        print(f"Using dtype: {self.torch_dtype}")

        # --- Preprocessing models ---
        # Background remove model: BEN2
        print("Loading BEN2 background removal model...")
        # Use local_dir='models' to keep it organized, cache_dir ensures Cog caching
        ben2_model_filename = 'BEN2_Base.pth'
        ben2_model_dir = 'models' # Corresponds to local_dir
        ben2_model_expected_path = os.path.join(ben2_model_dir, ben2_model_filename)

        ben2_model_path = None
        if os.path.exists(ben2_model_expected_path):
            print(f"BEN2 model found locally at: {ben2_model_expected_path}")
            ben2_model_path = ben2_model_expected_path
        else:
            print("BEN2 model not found locally, downloading...")
            ben2_model_path = hf_hub_download(
                repo_id='PramaLLC/BEN2',
                filename=ben2_model_filename,
                local_dir=ben2_model_dir, # Download into 'models' subdir
                cache_dir=MODEL_CACHE # Use overall cache dir for HF management
            )
            print(f"BEN2 model downloaded to: {ben2_model_path}")

        self.bg_rm_model = BEN2.BEN_Base().to(self.device).eval()
        # Load from the determined path
        self.bg_rm_model.loadcheckpoints(ben2_model_path)
        print("BEN2 model loaded.")

        # Face crop and align tool: facexlib
        print("Loading face helper...")
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png', # Internal setting, doesn't affect output format
            device=self.device,
            # Ensure model files for facexlib are downloaded to the cache
            model_rootpath=MODEL_CACHE
        )
        print("Face helper loaded.")

        # --- Load DreamO Pipeline ---
        print("Loading DreamO pipeline...")
        # Ensure the repo is accessible (public or using HF_TOKEN)
        model_root = 'black-forest-labs/FLUX.1-dev'
        self.dreamo_pipeline = DreamOPipeline.from_pretrained(
            model_root,
            torch_dtype=self.torch_dtype,
            cache_dir=MODEL_CACHE
        )
        # Load the DreamO specific weights (using turbo by default like app.py)
        # Modify use_turbo=True/False based on input if needed later
        self.dreamo_pipeline.load_dreamo_model(self.device, use_turbo=True)
        self.dreamo_pipeline = self.dreamo_pipeline.to(self.device)
        print("DreamO pipeline loaded.")
        print("Setup complete.")


    @torch.no_grad()
    def get_align_face(self, img: np.ndarray) -> np.ndarray | None:
        """
        Aligns the face in the input image (RGB numpy array).
        Returns the aligned face as an RGB numpy array, or None if no face is detected.
        """
        self.face_helper.clean_all()
        # Convert numpy RGB to BGR for OpenCV compatibility within face_helper
        image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.face_helper.read_image(image_bgr)
        # Detect landmarks for the center face
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        # Align and warp the face
        self.face_helper.align_warp_face()

        if len(self.face_helper.cropped_faces) == 0:
            print("Warning: No face detected by face_helper.")
            return None

        # Get the first cropped face (it's in BGR format from face_helper)
        align_face_bgr = self.face_helper.cropped_faces[0]

        # --- Face Parsing and Background Masking ---
        # Convert aligned BGR face to RGB tensor for parsing model
        input_tensor = img2tensor(align_face_bgr, bgr2rgb=True).unsqueeze(0) / 255.0
        input_tensor = input_tensor.to(self.device)
        # Normalize before parsing (using standard ImageNet stats)
        normalized_input = normalize(input_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # Get segmentation map
        parsing_out = self.face_helper.face_parse(normalized_input)[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)

        # Define background labels for segmentation mask (same as app.py)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        # Create a boolean mask where True indicates background pixels
        bg_mask = sum(parsing_out == i for i in bg_label).bool()

        # Create a white image tensor with the same shape as input
        white_image = torch.ones_like(input_tensor)
        # Replace background pixels with white, keeping face features
        face_features_tensor = torch.where(bg_mask, white_image, input_tensor)
        # Convert the resulting tensor back to an RGB numpy image
        face_features_image_rgb = tensor2img(face_features_tensor, rgb2bgr=False) # Output RGB numpy

        return face_features_image_rgb

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Prompt for image generation"),
        ref_image1: Path = Input(description="Reference image 1 (optional)", default=None),
        ref_task1: str = Input(choices=["ip", "id", "style"], default="ip", description="Task for reference image 1 ('ip': object/character, 'id': face identity, 'style': preserve style/background)"),
        ref_image2: Path = Input(description="Reference image 2 (optional)", default=None),
        ref_task2: str = Input(choices=["ip", "id", "style"], default="ip", description="Task for reference image 2 ('ip': object/character, 'id': face identity, 'style': preserve style/background)"),
        width: int = Input(description="Width of the output image (must be multiple of 16)", default=1024, ge=768, le=1024),
        height: int = Input(description="Height of the output image (must be multiple of 16)", default=1024, ge=768, le=1024),
        num_steps: int = Input(description="Number of inference steps", default=12, ge=8, le=30),
        guidance: float = Input(description="Guidance scale. Lower for less intensity/more realism (e.g., faces), higher for stronger prompt adherence.", default=3.5, ge=1.0, le=10.0),
        seed: int = Input(description="Random seed. Leave blank or set to -1 for random.", default=None),
        # Advanced options (matching Gradio defaults, but maybe hide by default in UI if possible)
        ref_res: int = Input(description="Resolution for non-ID reference image preprocessing (target pixel area)", default=512, ge=256, le=1024),
        neg_prompt: str = Input(description="Negative prompt", default=""),
        neg_guidance: float = Input(description="Negative guidance scale", default=3.5, ge=1.0, le=10.0),
        true_cfg: float = Input(description="True CFG scale (advanced, requires distilled CFG LoRA)", default=1.0, ge=1.0, le=5.0),
        cfg_start_step: int = Input(description="CFG start step (advanced)", default=0, ge=0, le=30),
        cfg_end_step: int = Input(description="CFG end step (advanced)", default=0, ge=0, le=30),
        first_step_guidance: float = Input(description="First step guidance scale override (advanced, 0 uses main guidance)", default=0, ge=0, le=10.0),
        output_format: str = Input(description="Format of the output image", choices=["webp", "jpg", "png"], default="webp"),
        output_quality: int = Input(description="Output quality for lossy formats (jpg, webp)", default=90, ge=1, le=100),

    ) -> Path:
        """Generate an image based on prompt and reference images using the DreamO pipeline."""
        print("--- Prediction Start ---")
        print(f"Prompt: {prompt}")
        print(f"Seed: {'Random' if seed is None or seed == -1 else seed}")
        print(f"Dimensions: {width}x{height}")
        print(f"Steps: {num_steps}, Guidance: {guidance}")
        if neg_prompt: print(f"Negative Prompt: {neg_prompt}, Neg Guidance: {neg_guidance}")
        print(f"Output Format: {output_format}, Quality: {output_quality}")


        # --- Input Validation ---
        if width % 16 != 0 or height % 16 != 0:
             # Use print for warnings/info in Cog, raise for fatal errors
             print(f"Warning: Width ({width}) or Height ({height}) not multiple of 16. Adjusting...")
             width = (width // 16) * 16
             height = (height // 16) * 16
             print(f"Adjusted Dimensions: {width}x{height}")
        if cfg_start_step >= num_steps or cfg_end_step >= num_steps:
            print(f"Warning: cfg_start_step ({cfg_start_step}) or cfg_end_step ({cfg_end_step}) is too high for num_steps ({num_steps}). Adjusting end step.")
            cfg_end_step = min(cfg_end_step, num_steps -1)
            cfg_start_step = min(cfg_start_step, cfg_end_step)


        # --- Preprocess Reference Images ---
        ref_conds = []
        ref_inputs = [(ref_image1, ref_task1), (ref_image2, ref_task2)]
        # debug_images = [] # Keep track of preprocessed images if needed for debugging output

        for idx, (ref_image_path, ref_task) in enumerate(ref_inputs):
            if ref_image_path is not None and os.path.exists(str(ref_image_path)):
                print(f"Processing reference image {idx+1} ({ref_image_path}) with task: {ref_task}")
                try:
                    # Load image using PIL, convert to RGB, then to numpy
                    ref_image = Image.open(ref_image_path).convert("RGB")
                    ref_image_np = np.array(ref_image) # Now it's RGB numpy HWC

                    processed_image_np = None # Store the result after task-specific processing

                    if ref_task == "id":
                        print("  Task: ID - Aligning face...")
                        # Resize before face alignment (using long edge logic from app.py)
                        ref_image_resized_np = resize_numpy_image_long(ref_image_np, 1024)
                        aligned_face_np = self.get_align_face(ref_image_resized_np) # Expects RGB numpy, returns RGB numpy
                        if aligned_face_np is None:
                            print(f"  Warning: Could not detect/align face in reference image {idx+1}. Skipping this reference.")
                            continue # Skip to next reference image
                        processed_image_np = aligned_face_np # Use the aligned face (RGB numpy)
                        print("  Face alignment complete.")
                    elif ref_task == "ip":
                         print("  Task: IP - Removing background...")
                         # BEN2 expects PIL Image
                         ref_image_pil = Image.fromarray(ref_image_np) # Convert numpy back to PIL
                         ref_image_pil_no_bg = self.bg_rm_model.inference(ref_image_pil)
                         processed_image_np = np.array(ref_image_pil_no_bg) # Convert back to numpy (RGB)
                         print("  Background removal complete.")
                    elif ref_task == "style":
                         print("  Task: Style - Keeping original image.")
                         processed_image_np = ref_image_np # Use the original image numpy
                    else:
                        # Should not happen with choices constraint, but good practice
                        print(f"  Warning: Unknown task '{ref_task}'. Skipping this reference.")
                        continue

                    # Resize non-ID images based on area *after* potential BG removal/alignment
                    if ref_task != "id":
                        target_area = ref_res * ref_res
                        print(f"  Resizing reference image {idx+1} towards target area {target_area}...")
                        processed_image_np = resize_numpy_image_area(processed_image_np, target_area) # Expects numpy (RGB)
                        print(f"  Resized to shape: {processed_image_np.shape}")

                    # --- Convert to Tensor for Model ---
                    # img2tensor expects numpy HWC, returns CHW tensor
                    # Pass bgr2rgb=False as input is already RGB numpy
                    ref_image_tensor = img2tensor(processed_image_np, bgr2rgb=False).unsqueeze(0) / 255.0 # Scale to [0, 1]
                    ref_image_tensor = 2 * ref_image_tensor - 1.0 # Normalize to [-1, 1]
                    ref_image_tensor = ref_image_tensor.to(self.device, dtype=self.torch_dtype) # Move to device and set dtype

                    ref_conds.append(
                        {
                            'img': ref_image_tensor,
                            'task': ref_task,
                            'idx': idx + 1, # Keep 1-based indexing as per app.py example
                        }
                    )
                    print(f"Reference image {idx+1} processed successfully.")
                    # Optionally save debug image:
                    # debug_images.append(Image.fromarray(processed_image_np))

                except Exception as e:
                    print(f"Error processing reference image {idx+1}: {e}")
                    # Continue without this reference image
            elif ref_image_path is not None:
                 print(f"Warning: Reference image {idx+1} path not found or invalid: {ref_image_path}. Skipping.")
            else:
                 print(f"Reference image {idx+1} not provided.")

        # --- Seed ---
        if seed is None or seed == -1:
            seed = torch.Generator(device="cpu").seed() # Generate a random seed if needed
        print(f"Using seed for generation: {seed}")
        # Use a CPU generator for reproducibility across runs, even if inference is on GPU
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # --- Run Inference ---
        print("Starting DreamO pipeline inference...")
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

        output_image = self.dreamo_pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            ref_conds=ref_conds,
            generator=generator,
            true_cfg_scale=true_cfg,
            true_cfg_start_step=cfg_start_step,
            true_cfg_end_step=cfg_end_step,
            negative_prompt=neg_prompt,
            neg_guidance_scale=neg_guidance,
            # Use main guidance if first_step_guidance is 0 or less
            first_step_guidance_scale=first_step_guidance if first_step_guidance > 0 else guidance,
        ).images[0] # Get the first PIL image from the output list

        end_time.record()
        torch.cuda.synchronize()
        inference_time = start_time.elapsed_time(end_time) / 1000
        print(f"Inference complete in {inference_time:.2f} seconds.")

        # --- Prepare Output ---
        # Ensure image is in RGB mode (DreamO likely outputs PIL RGB, but double-check)
        if output_image.mode != "RGB":
            print("Converting output image to RGB...")
            output_image = output_image.convert("RGB")

        # Prepare saving arguments
        output_format = output_format.lower()
        extension = output_format
        if extension == "jpg":
            extension = "jpeg" # PIL uses 'jpeg'

        output_path_str = f"/tmp/output.{extension}"
        save_params = {}
        if output_format in ["jpeg", "webp", "jpg"]:
            print(f"Using output quality: {output_quality}")
            save_params["quality"] = output_quality
            if output_format == "webp":
                 save_params["lossless"] = False # Ensure lossy webp if quality < 100
            # Optimize is generally good for web formats
            save_params["optimize"] = True

        # Save the image
        output_image.save(output_path_str, **save_params)
        print(f"Output image saved to {output_path_str}")
        print("--- Prediction End ---")

        # Return the path to the saved image
        return Path(output_path_str)
