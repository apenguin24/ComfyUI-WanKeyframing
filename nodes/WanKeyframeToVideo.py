import re, torch
import nodes
import comfy
import comfy.utils
import comfy.model_management

MAX_RES = 8192

class WanKeyframeToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "positive_prompt": ("STRING", {"multiline": True, "default": "[0] A beautiful meadow\n[1] A misty forest\n[2] A glowing futuristic city"}),
                "negative_prompt": ("STRING", {"multiline": False, "default": "low quality, blurry, bad lighting"}),
                "vae": ("VAE",),
                "width": ("INT", {"default": 496, "min": 16, "max": MAX_RES, "step": 16}),
                "height": ("INT", {"default": 496, "min": 16, "max": MAX_RES, "step": 16}),
                "fps": ("INT", {"default": 16, "min": 1, "max": 120}),
                "seconds": ("INT", {"default": 1, "min": 1, "max": 60}),
            },
            "optional": {
                "keyframes": ("IMAGE",),
                "clip_vision_outputs": ("CLIP_VISION_OUTPUT",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "conditioning/video_models"
    OUTPUT_IS_LIST = (False, False, False)

    def ensure_batch(self, img):
        """
        Accepts IMAGE input that may be shape [H,W,C] or [N,H,W,C].
        Returns either None or a tensor shaped [N,H,W,C].
        """
        if img is None:
            return None
        if not hasattr(img, "dim"):
            return None
        if img.dim() == 3:
            return img.unsqueeze(0)
        return img

    def encode(
        self, clip, positive_prompt, negative_prompt, vae, width, height, fps, seconds,
        keyframes=None, clip_vision_outputs=None
    ):
        device = comfy.model_management.intermediate_device()
        
        # Calculate length for specified seconds of video
        length = (fps * seconds) + 1  # +1 to ensure we have full second
        # Round to nearest multiple of 4 for compatibility
        length = ((length - 1) // 4) * 4 + 1

        # --- Parse prompts by line format ---
        lines = [line.strip() for line in positive_prompt.split('\n') if line.strip()]
        if len(lines) == 0:
            lines = [positive_prompt]
        
        indexed_prompts = {} 
        
        bracket_pattern = r"^\[(\d+)\]\s*(.+)$"
        for line in lines:
            match = re.match(bracket_pattern, line)
            if match:
                idx = int(match.group(1))
                prompt = match.group(2).strip()
                indexed_prompts[idx] = prompt
        
        # --- Process keyframes ---
        keyframes = self.ensure_batch(keyframes)
        num_keyframes = keyframes.shape[0] if keyframes is not None and keyframes.shape[0] > 0 else 0
        
        if num_keyframes < 2:
            raise ValueError("At least 2 keyframes are required to create video segments")
        
        num_segments = num_keyframes - 1
        
        spacial_scale = vae.spacial_compression_encode()
        
        # Prepare storage for stacked latents and conditioning
        stacked_latent = torch.zeros(
            [num_segments, vae.latent_channels, ((length - 1) // 4) + 1, height // spacial_scale, width // spacial_scale],
            device=device
        )
        
        stacked_positive_cond = []
        stacked_negative_cond = []
        stacked_concat_latents = []
        stacked_masks = []
        stacked_clip_vision_outputs = []
        
        for i in range(num_segments):
            # Extract frames - keyframes are already in [N, H, W, C] format after ensure_batch
            start_frame = keyframes[i:i+1]  # [1, H, W, C]
            end_frame = keyframes[i+1:i+2]  # [1, H, W, C]
            
            # Upscale frames: [1, H, W, C] -> [1, C, H, W] -> upscale -> [1, C, height, width]
            start_frame = comfy.utils.common_upscale(
                start_frame.movedim(-1, 1), 
                width, height, "bilinear", "center"
            ).movedim(1, -1)  # -> [1, height, width, C]
            
            end_frame = comfy.utils.common_upscale(
                end_frame.movedim(-1, 1), 
                width, height, "bilinear", "center"
            ).movedim(1, -1)  # -> [1, height, width, C]
            
            # Get the prompt for this segment
            if i in indexed_prompts:
                formatted_positive = indexed_prompts[i]
            elif 0 in indexed_prompts:
                formatted_positive = indexed_prompts[0]
            else:
                formatted_positive = lines[0] if lines else positive_prompt
            
            print(f"[Segment {i}] Using prompt from keyframe {i}: '{formatted_positive[:50]}...'")
            
            # Encode positive and negative prompts for this segment
            positive_cond = nodes.CLIPTextEncode().encode(clip=clip, text=formatted_positive)[0]
            negative_cond = nodes.CLIPTextEncode().encode(clip=clip, text=negative_prompt)[0]
            
            # Create image tensor with start and end frames
            image = torch.ones((length, height, width, 3)) * 0.5

            image[0:1] = start_frame
            image[-1:] = end_frame
            
            concat_latent_image = vae.encode(image[:, :, :, :3])
            
            # Create mask with correct latent dimensions
            latent_time_dim = concat_latent_image.shape[2]
            latent_height = concat_latent_image.shape[3]
            latent_width = concat_latent_image.shape[4]
            
            mask = torch.ones((1, 1, latent_time_dim * 4, latent_height, latent_width))
            
            start_mask_frames = min(4, latent_time_dim * 4)
            mask[:, :, :start_mask_frames] = 0.0
            
            end_mask_frames = min(1, latent_time_dim * 4)
            mask[:, :, -end_mask_frames:] = 0.0
            
            # Reshape mask to match expected latent format
            mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
            
            # Store for stacking
            stacked_concat_latents.append(concat_latent_image)
            stacked_masks.append(mask)
            
            # Handle CLIP vision if provided
            clip_vision_output = None
            
            if clip_vision_outputs is not None:
                if hasattr(clip_vision_outputs, '__len__') and len(clip_vision_outputs) > i:
                    clip_vision_output = clip_vision_outputs[i]
                
                if hasattr(clip_vision_outputs, '__len__') and len(clip_vision_outputs) > i + 1:
                    end_cv = clip_vision_outputs[i + 1]
                    
                    if clip_vision_output is not None:
                        # Merge start and end clip vision outputs
                        states = torch.cat([
                            clip_vision_output.penultimate_hidden_states,
                            end_cv.penultimate_hidden_states
                        ], dim=-2)
                        
                        clip_vision_output = comfy.clip_vision.Output()
                        clip_vision_output.penultimate_hidden_states = states
                    else:
                        clip_vision_output = end_cv
            
            stacked_clip_vision_outputs.append(clip_vision_output)
            stacked_positive_cond.append(positive_cond)
            stacked_negative_cond.append(negative_cond)
        
        # Stack all conditioning data
        final_positive_cond = self._stack_conditioning(
            stacked_positive_cond, stacked_concat_latents, stacked_masks, stacked_clip_vision_outputs
        )
        final_negative_cond = self._stack_conditioning(
            stacked_negative_cond, stacked_concat_latents, stacked_masks, stacked_clip_vision_outputs
        )
        
        return (final_positive_cond, final_negative_cond, {"samples": stacked_latent})
    
    def _stack_conditioning(self, cond_list, concat_latents, masks, clip_vision_outputs):
        """
        Stack conditioning data from multiple segments into a single conditioning tensor
        """
        if not cond_list:
            return []
            
        embeddings = []
        pooled_outputs = []
        
        for cond in cond_list:
            if len(cond) > 0:
                embeddings.append(cond[0][0])  # Text embedding
                if len(cond[0]) > 1 and "pooled_output" in cond[0][1]:
                    pooled_output = cond[0][1]["pooled_output"]
                    if pooled_output is not None:
                        pooled_outputs.append(pooled_output)
        
        # Stack embeddings along batch dimension
        if embeddings:
            try:
                stacked_embeddings = torch.cat(embeddings, dim=0)
            except Exception as e:
                print(f"Warning: Could not stack embeddings: {e}")
                # Use the first embedding as fallback
                stacked_embeddings = embeddings[0]
        else:
            return []
            
        conditioning_dict = {}
        
        # Stack pooled outputs if available - ensure they have compatible shapes
        if pooled_outputs:
            try:
                # Validate that all pooled outputs have the same shape except for batch dimension
                first_shape = pooled_outputs[0].shape
                valid_pooled = []
                
                for pooled in pooled_outputs:
                    if pooled.shape[1:] == first_shape[1:]:  # Same shape except batch dim
                        valid_pooled.append(pooled)
                    else:
                        print(f"Warning: Skipping pooled output with incompatible shape {pooled.shape}, expected {first_shape}")
                
                if valid_pooled:
                    stacked_pooled = torch.cat(valid_pooled, dim=0)
                    conditioning_dict["pooled_output"] = stacked_pooled
            except Exception as e:
                print(f"Warning: Could not stack pooled outputs: {e}")
                # Use the first pooled output as fallback
                if len(pooled_outputs) > 0:
                    conditioning_dict["pooled_output"] = pooled_outputs[0]
        
        # Stack concat latent images
        if concat_latents:
            try:
                stacked_concat_latents = torch.cat(concat_latents, dim=0)
                conditioning_dict["concat_latent_image"] = stacked_concat_latents
            except Exception as e:
                print(f"Warning: Could not stack concat latent images: {e}")
                # Use the first latent as fallback
                conditioning_dict["concat_latent_image"] = concat_latents[0]
        
        # Stack masks
        if masks:
            try:
                stacked_masks = torch.cat(masks, dim=0)
                conditioning_dict["concat_mask"] = stacked_masks
            except Exception as e:
                print(f"Warning: Could not stack masks: {e}")
                # Use the first mask as fallback
                conditioning_dict["concat_mask"] = masks[0]
            
        # Handle clip vision outputs - concatenate if available
        if clip_vision_outputs and any(cv is not None for cv in clip_vision_outputs):
            valid_clip_visions = [cv for cv in clip_vision_outputs if cv is not None]
            if valid_clip_visions:
                try:
                    # Stack the penultimate hidden states
                    stacked_states = torch.cat([
                        cv.penultimate_hidden_states for cv in valid_clip_visions
                    ], dim=0)
                    
                    clip_vision_output = comfy.clip_vision.Output()
                    clip_vision_output.penultimate_hidden_states = stacked_states
                    conditioning_dict["clip_vision_output"] = clip_vision_output
                except Exception as e:
                    print(f"Warning: Could not stack clip vision outputs: {e}")
                    # Use the first clip vision output as fallback
                    conditioning_dict["clip_vision_output"] = valid_clip_visions[0]
        
        return [[stacked_embeddings, conditioning_dict]]