import os
import modal
from pathlib import Path
from typing import List, Optional

# Define the local directory of the repository in Modal
app_name = "comfyui-modal"

# Define the Container Image
# Include standard GPU dependencies and requirements
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "curl",
        "wget",
        "libgl1",
        "libglib2.0-0",
        "ffmpeg",
    )
    .pip_install_from_requirements("requirements.txt")
    .pip_install("huggingface_hub[hf-transfer]", "modal")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "DOWNLOAD_LATEST_WEIGHTS_MANIFEST": "true",
        "YOLO_CONFIG_DIR": "/tmp/Ultralytics",
    })
)

app = modal.App(name=app_name, image=image)

# We mount our code into the container
mounts = [
    modal.Mount.from_local_dir(".", remote_path="/root/comfyui")
]


@app.cls(
    gpu="A10G",  # Default to A10G which is a great budget/performance option
    timeout=1200,
    mounts=mounts,
    workdir="/root/comfyui",
)
class Model:
    @modal.enter()
    def setup(self):
        print("🚀 Initializing ComfyUI Predictor...")
        from predict import Predictor
        self.predictor = Predictor()
        self.predictor.setup()

    @modal.method()
    def predict(
        self,
        workflow_json: str,
        input_file_url: Optional[str] = None,
        return_temp_files: bool = False,
        output_format: str = "webp",
        output_quality: int = 95,
        randomise_seeds: bool = True,
        force_reset_cache: bool = False,
    ) -> List[bytes]:
        """
        Runs a prediction using the specified ComfyUI workflow.
        Returns the output files as a list of bytes.
        """
        import urllib.request
        from predict import INPUT_DIR
        
        input_file_path = None
        if input_file_url:
            print(f"📥 Downloading input file from {input_file_url}...")
            os.makedirs(INPUT_DIR, exist_ok=True)
            # Determine extension or generic name
            ext = os.path.splitext(input_file_url.split("?")[0])[1] or ".bin"
            input_file_path = Path(INPUT_DIR) / f"input_file{ext}"
            
            # Download file
            req = urllib.request.Request(
                input_file_url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as response, open(input_file_path, 'wb') as out_file:
                import shutil
                shutil.copyfileobj(response, out_file)
            print(f"✅ Input file saved to {input_file_path}")

        # Run prediction
        output_paths = self.predictor.predict(
            workflow_json=workflow_json,
            input_file=input_file_path,
            return_temp_files=return_temp_files,
            output_format=output_format,
            output_quality=output_quality,
            randomise_seeds=randomise_seeds,
            force_reset_cache=force_reset_cache,
        )

        # Read output files to return them as bytes
        output_files = []
        for path in output_paths:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    output_files.append(f.read())
                print(f"📤 Loaded output file: {path}")

        return output_files


@app.local_entrypoint()
def main():
    """
    Test running the ComfyUI workflow on Modal.
    """
    print("To run this on Modal:")
    print("modal run modal_app.py")
