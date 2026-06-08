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

        # Parse URL to extract Hugging Face repository and filename path
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip("/").split("/")

        # HuggingFace URL structure: /<org>/<repo>/resolve/<branch>/<file_path>
        # e.g., /fofr/comfyui/resolve/main/checkpoints/512-inpainting-ema.safetensors
        if "huggingface.co" in parsed_url.netloc and len(path_parts) >= 5 and path_parts[2] == "resolve":
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
