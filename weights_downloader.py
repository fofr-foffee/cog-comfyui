import subprocess
import time
import os
from weights_manifest import WeightsManifest


class WeightsDownloader:
    supported_filetypes = [
        ".ckpt",
        ".safetensors",
        ".sft",
        ".pt",
        ".pth",
        ".bin",
        ".onnx",
        ".torchscript",
        ".engine",
        ".patch",
    ]

    def __init__(self):
        self.weights_manifest = WeightsManifest()
        self.weights_map = self.weights_manifest.weights_map

    def get_canonical_weight_str(self, weight_str):
        return self.weights_manifest.get_canonical_weight_str(weight_str)

    def get_weights_by_type(self, type):
        return self.weights_manifest.get_weights_by_type(type)

    def download_weights(self, weight_str):
        if weight_str in self.weights_map:
            if self.weights_manifest.is_non_commercial_only(weight_str):
                print(
                    f"⚠️  {weight_str} is for non-commercial use only. Unless you have obtained a commercial license.\nDetails: https://github.com/replicate/cog-comfyui/blob/main/weights_licenses.md"
                )

            if isinstance(self.weights_map[weight_str], list):
                for weight in self.weights_map[weight_str]:
                    self.download_if_not_exists(
                        weight_str, weight["url"], weight["dest"]
                    )
            else:
                self.download_if_not_exists(
                    weight_str,
                    self.weights_map[weight_str]["url"],
                    self.weights_map[weight_str]["dest"],
                )
        else:
            raise ValueError(
                f"{weight_str} unavailable. View the list of available weights: https://github.com/replicate/cog-comfyui/blob/main/supported_weights.md"
            )

    def check_if_file_exists(self, weight_str, dest):
        if dest.endswith(weight_str):
            path_string = dest
        else:
            path_string = os.path.join(dest, weight_str)
        return os.path.exists(path_string)

    def download_if_not_exists(self, weight_str, url, dest):
        if self.check_if_file_exists(weight_str, dest):
            print(f"✅ {weight_str} exists in {dest}")
            return
        WeightsDownloader.download(weight_str, url, dest)

    @staticmethod
    def download(weight_str, url, dest):
        import urllib.request
        from urllib.parse import urlparse
        import shutil

        # Handle weight_str with subfolders
        if "/" in weight_str:
            subfolder = weight_str.rsplit("/", 1)[0]
            dest = os.path.join(dest, subfolder)
            os.makedirs(dest, exist_ok=True)

        base_weight_str = os.path.basename(weight_str)

        # Custom directory-based model downloads from original HF repositories
        if base_weight_str == "bert-base-uncased":
            target_dir = dest if dest.endswith("bert-base-uncased") else os.path.join(dest, "bert-base-uncased")
            os.makedirs(target_dir, exist_ok=True)
            files = ["config.json", "model.safetensors", "tokenizer_config.json", "tokenizer.json", "vocab.txt"]
            print(f"⏳ Downloading bert-base-uncased directory from google-bert/bert-base-uncased to {target_dir}")
            start = time.time()
            for f in files:
                from huggingface_hub import hf_hub_download
                downloaded = hf_hub_download(repo_id="google-bert/bert-base-uncased", filename=f)
                out_path = os.path.join(target_dir, f)
                if os.path.exists(out_path):
                    os.remove(out_path)
                try:
                    os.symlink(downloaded, out_path)
                except OSError:
                    shutil.copy(downloaded, out_path)
            print(f"✅ bert-base-uncased ready at {target_dir} in {time.time() - start:.2f}s")
            return

        if base_weight_str == "antelopev2":
            target_dir = dest if dest.endswith("antelopev2") else os.path.join(dest, "antelopev2")
            os.makedirs(target_dir, exist_ok=True)
            files = ["1k3d68.onnx", "2d106det.onnx", "genderage.onnx", "glintr100.onnx", "scrfd_10g_bnkps.onnx"]
            print(f"⏳ Downloading antelopev2 directory from DIAMONIK7777/antelopev2 to {target_dir}")
            start = time.time()
            for f in files:
                from huggingface_hub import hf_hub_download
                downloaded = hf_hub_download(repo_id="DIAMONIK7777/antelopev2", filename=f)
                out_path = os.path.join(target_dir, f)
                if os.path.exists(out_path):
                    os.remove(out_path)
                try:
                    os.symlink(downloaded, out_path)
                except OSError:
                    shutil.copy(downloaded, out_path)
            print(f"✅ antelopev2 ready at {target_dir} in {time.time() - start:.2f}s")
            return

        if base_weight_str == "buffalo_l":
            target_dir = dest if dest.endswith("buffalo_l") else os.path.join(dest, "buffalo_l")
            os.makedirs(target_dir, exist_ok=True)
            files = ["1k3d68.onnx", "2d106det.onnx", "det_10g.onnx", "genderage.onnx", "w600k_r50.onnx"]
            print(f"⏳ Downloading buffalo_l directory from immich-app/buffalo_l to {target_dir}")
            start = time.time()
            for f in files:
                from huggingface_hub import hf_hub_download
                downloaded = hf_hub_download(repo_id="immich-app/buffalo_l", filename=f)
                out_path = os.path.join(target_dir, f)
                if os.path.exists(out_path):
                    os.remove(out_path)
                try:
                    os.symlink(downloaded, out_path)
                except OSError:
                    shutil.copy(downloaded, out_path)
            print(f"✅ buffalo_l ready at {target_dir} in {time.time() - start:.2f}s")
            return

        # PyTorch Hub or other direct URL models override
        DIRECT_URLS = {
            "mobilenet_v2-b0353104.pth": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
            "vgg16-397923af.pth": "https://download.pytorch.org/models/vgg16-397923af.pth",
            "swin_b-68c6b09e.pth": "https://download.pytorch.org/models/swin_b-68c6b09e.pth",
            "fbcnn_color.pth": "https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth",
        }

        if base_weight_str in DIRECT_URLS:
            direct_url = DIRECT_URLS[base_weight_str]
            target_path = os.path.join(dest, base_weight_str)
            print(f"⏳ Downloading {base_weight_str} directly from {direct_url} to {target_path}")
            start = time.time()
            if os.path.exists(target_path):
                if os.path.islink(target_path) or os.path.isfile(target_path):
                    os.remove(target_path)
                elif os.path.isdir(target_path):
                    shutil.rmtree(target_path)
            os.makedirs(dest, exist_ok=True)
            try:
                req = urllib.request.Request(
                    direct_url,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
                print(f"✅ Successfully downloaded {direct_url} to {target_path} via urllib")
            except Exception as e2:
                raise RuntimeError(f"Direct download failed for {direct_url}. Urllib error: {e2}")

            elapsed_time = time.time() - start
            try:
                file_size_bytes = os.path.getsize(target_path)
                file_size_megabytes = file_size_bytes / (1024 * 1024)
                print(
                    f"✅ {weight_str} ready at {dest} in {elapsed_time:.2f}s, size: {file_size_megabytes:.2f}MB"
                )
            except FileNotFoundError:
                print(f"✅ {weight_str} ready at {dest} in {elapsed_time:.2f}s")
            return

        # Optional redirection to official HF repo where possible
        ORIGINAL_REPOS = {
            # SD 1.5 & XL
            "v1-5-pruned-emaonly.safetensors": ("runwayml/stable-diffusion-v1-5", "v1-5-pruned-emaonly.safetensors"),
            "sd_xl_base_1.0.safetensors": ("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors"),
            "sd_xl_refiner_1.0.safetensors": ("stabilityai/stable-diffusion-xl-refiner-1.0", "sd_xl_refiner_1.0.safetensors"),
            # Depth Anything V2
            "depth_anything_v2_vitl.safetensors": ("depth-anything/Depth-Anything-V2-Large", "depth_anything_v2_vitl.safetensors"),
            "depth_anything_v2_vitb.safetensors": ("depth-anything/Depth-Anything-V2-Base", "depth_anything_v2_vitb.safetensors"),
            "depth_anything_v2_vits.safetensors": ("depth-anything/Depth-Anything-V2-Small", "depth_anything_v2_vits.safetensors"),
            # BiRefNet
            "birefnet-general-epoch150.safetensors": ("ZhengPeng7/BiRefNet", "birefnet-general-epoch150.safetensors"),
            "General.safetensors": ("ZhengPeng7/BiRefNet", "model.safetensors"),
            # LTX-Video
            "ltx-video-2b-v0.9.1.safetensors": ("Lightricks/LTX-Video", "ltx-video-2b-v0.9.1.safetensors"),
        }

        # Parse URL to extract Hugging Face repository and filename path
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip("/").split("/")

        if base_weight_str in ORIGINAL_REPOS:
            repo_id, filename = ORIGINAL_REPOS[base_weight_str]
        elif "huggingface.co" in parsed_url.netloc and len(path_parts) >= 5 and path_parts[2] == "resolve":
            repo_id = f"{path_parts[0]}/{path_parts[1]}"
            filename = "/".join(path_parts[4:])
        else:
            repo_id = "fofr/comfyui"
            # Fallback path parsing
            filename = url.replace("https://weights.replicate.delivery/default/comfy-ui/", "")
            filename = filename.replace("https://huggingface.co/fofr/comfyui/resolve/main/", "")

        # Strip .tar suffix as weights on HF are raw files
        if filename.endswith(".tar"):
            filename = filename[:-4]

        # Final destination path
        target_file_name = os.path.basename(filename)
        target_path = os.path.join(dest, target_file_name)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        print(f"⏳ Downloading {filename} from Hugging Face ({repo_id}) to {target_path}")
        start = time.time()

        try:
            from huggingface_hub import hf_hub_download
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
            )

            # Link or copy to target path
            if os.path.exists(target_path):
                if os.path.islink(target_path) or os.path.isfile(target_path):
                    os.remove(target_path)
                elif os.path.isdir(target_path):
                    shutil.rmtree(target_path)

            try:
                os.symlink(downloaded_path, target_path)
                print(f"✅ Created symlink from {downloaded_path} to {target_path}")
            except OSError:
                shutil.copy(downloaded_path, target_path)
                print(f"✅ Copied {downloaded_path} to {target_path}")

        except Exception as e:
            print(f"Failed downloading via huggingface_hub: {e}")
            print("Falling back to downloading file directly via urllib...")
            direct_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            if os.path.exists(target_path):
                if os.path.islink(target_path) or os.path.isfile(target_path):
                    os.remove(target_path)
                elif os.path.isdir(target_path):
                    shutil.rmtree(target_path)

            try:
                # Use a custom user-agent to avoid Hugging Face blockages
                req = urllib.request.Request(
                    direct_url,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
                print(f"✅ Successfully downloaded {direct_url} to {target_path} via urllib")
            except Exception as e2:
                raise RuntimeError(f"All download attempts failed. HF Hub error: {e}. Urllib error: {e2}")

        elapsed_time = time.time() - start
        try:
            file_size_bytes = os.path.getsize(target_path)
            file_size_megabytes = file_size_bytes / (1024 * 1024)
            print(
                f"✅ {weight_str} ready at {dest} in {elapsed_time:.2f}s, size: {file_size_megabytes:.2f}MB"
            )
        except FileNotFoundError:
            print(f"✅ {weight_str} ready at {dest} in {elapsed_time:.2f}s")

    def delete_weights(self, weight_str):
        if weight_str in self.weights_map:
            weight_path = os.path.join(self.weights_map[weight_str]["dest"], weight_str)
            if os.path.exists(weight_path):
                os.remove(weight_path)
                print(f"Deleted {weight_path}")
