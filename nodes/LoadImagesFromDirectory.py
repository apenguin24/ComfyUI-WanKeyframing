import os
import numpy as np
import torch
import folder_paths

from PIL import Image

MAX_RES = 8192 

class LoadImagesFromDirectory:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "keyframes"}),
                "reload_on_execute": ("BOOLEAN", {"default": False}),
                "target_width": ("INT", {"default": 512, "min": 64, "max": MAX_RES, "step": 8}),
                "target_height": ("INT", {"default": 512, "min": 64, "max": MAX_RES, "step": 8}),
                "resize_mode": (["stretch", "fit", "crop"], {"default": "crop"}),
                "sort_mode": (["name_asc", "name_desc", "date_asc", "date_desc", "size_asc", "size_desc", "none"], {"default": "name_asc"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "load_images"
    CATEGORY = "Load"

    @classmethod
    def IS_CHANGED(cls, directory, reload_on_execute, target_width, target_height, resize_mode, sort_mode):
        """
        Force node to re-execute when reload_on_execute is True.
        Returns a unique value each time to bypass caching.
        """
        if reload_on_execute:
            import time
            return str(time.time())
        return f"{directory}_{target_width}_{target_height}_{resize_mode}_{sort_mode}"

    def resize_image(self, pil_img, target_width, target_height, resize_mode):
        """
        Resize PIL image to target dimensions based on resize mode.
        """
        original_width, original_height = pil_img.size
        
        if resize_mode == "stretch":
            # Simple stretch to exact dimensions
            return pil_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        elif resize_mode == "fit":
            # Fit image inside target dimensions maintaining aspect ratio
            pil_img.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Create a new image with target dimensions and paste the resized image centered
            new_img = Image.new("RGB", (target_width, target_height), (0, 0, 0))
            paste_x = (target_width - pil_img.width) // 2
            paste_y = (target_height - pil_img.height) // 2
            new_img.paste(pil_img, (paste_x, paste_y))
            return new_img
        
        elif resize_mode == "crop":
            # Crop to fit target dimensions maintaining aspect ratio
            original_aspect = original_width / original_height
            target_aspect = target_width / target_height
            
            if original_aspect > target_aspect:
                # Image is wider, crop width
                new_height = original_height
                new_width = int(original_height * target_aspect)
                left = (original_width - new_width) // 2
                top = 0
                right = left + new_width
                bottom = original_height
            else:
                # Image is taller, crop height
                new_width = original_width
                new_height = int(original_width / target_aspect)
                left = 0
                top = (original_height - new_height) // 2
                right = original_width
                bottom = top + new_height
            
            cropped = pil_img.crop((left, top, right, bottom))
            return cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        return pil_img

    def sort_files(self, file_paths, sort_mode):
        """
        Sort file paths based on the specified sort mode.
        """
        if sort_mode == "none":
            return file_paths
        
        if sort_mode == "name_asc":
            return sorted(file_paths)
        elif sort_mode == "name_desc":
            return sorted(file_paths, reverse=True)
        elif sort_mode in ["date_asc", "date_desc", "size_asc", "size_desc"]:
            # Get file stats for sorting
            file_stats = []
            for fpath in file_paths:
                try:
                    stat = os.stat(fpath)
                    if sort_mode.startswith("date"):
                        # Sort by modification time
                        file_stats.append((stat.st_mtime, fpath))
                    else:  # size
                        # Sort by file size
                        file_stats.append((stat.st_size, fpath))
                except OSError:
                    # If we can't get stats, put it at the end with a default value
                    default_value = 0 if sort_mode.startswith("size") else float('inf')
                    file_stats.append((default_value, fpath))
            
            # Sort by the stat value
            reverse = sort_mode.endswith("desc")
            file_stats.sort(key=lambda x: x[0], reverse=reverse)
            return [fpath for _, fpath in file_stats]
        
        return file_paths

    def pil_to_comfy_tensor(self, pil_img):
        """
        Convert a PIL.Image (RGB) to a torch tensor shaped (1, H, W, 3),
        dtype=float32 and values in [0,1], compatible with ComfyUI batch expectations.
        """
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        arr = np.asarray(pil_img)  # shape (H, W, 3), dtype=uint8
        # Ensure contiguous and float32 scaled to 0..1
        arr = np.ascontiguousarray(arr).astype(np.float32) / 255.0
        # Convert to torch tensor
        t = torch.from_numpy(arr)  # shape (H, W, 3), dtype=float32
        # Add batch dim -> (1, H, W, 3)
        t = t.unsqueeze(0)
        return t

    def load_images(self, directory, reload_on_execute, target_width, target_height, resize_mode, sort_mode):
        image_tensors = []

        base_dir = folder_paths.get_input_directory()
        full_dir = os.path.join(base_dir, directory)

        if not os.path.exists(full_dir):
            raise FileNotFoundError(f"Directory not found: {full_dir}")
 

        supported = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
        
        # Collect all valid image files first
        valid_files = []
        for fname in os.listdir(full_dir):
            # skip non-string or hidden entries
            if not isinstance(fname, str) or fname.startswith("."):
                continue

            if not fname.lower().endswith(supported):
                # not a supported extension
                continue

            fpath = os.path.join(full_dir, fname)
            valid_files.append(fpath)
        
        # Sort the files according to sort_mode
        sorted_files = self.sort_files(valid_files, sort_mode)
        
        for fpath in sorted_files:
            fname = os.path.basename(fpath)
            try:
                with Image.open(fpath) as im:
                    original_size = im.size
                    im = im.convert("RGB")
                    
                    # Resize image to target dimensions
                    im = self.resize_image(im, target_width, target_height, resize_mode)
                    
                    t = self.pil_to_comfy_tensor(im)  # (1, H, W, 3)
                    image_tensors.append(t)
            except Exception as e:
                raise RuntimeError(f"Failed to load image {fname}: {str(e)}")

        if not image_tensors:
            raise ValueError(f"No valid images found in: {full_dir}")

        # Concatenate along batch dim -> (batch, H, W, 3)
        batch = torch.cat(image_tensors, dim=0)
        return (batch,)