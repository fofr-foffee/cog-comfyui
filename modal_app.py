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
    .add_local_dir(".", remote_path="/root/comfyui", ignore=["venv", ".git"])
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "DOWNLOAD_LATEST_WEIGHTS_MANIFEST": "true",
        "YOLO_CONFIG_DIR": "/tmp/Ultralytics",
    })
)

app = modal.App(name=app_name, image=image)


@app.cls(
    gpu="A100",  # Using high-performance A100 GPU as requested
    timeout=1200,
)
class Model:
    @modal.enter()
    def setup(self):
        print("🚀 Initializing ComfyUI Predictor...")
        import os
        os.chdir("/root/comfyui")
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
    Load test cases from cog-safe-push-configs/default.yaml and run them on Modal.
    """
    import yaml
    import sys
    
    config_path = Path("cog-safe-push-configs/default.yaml")
    if not config_path.exists():
        print(f"❌ Test config not found at {config_path}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    test_cases = config.get("predict", {}).get("test_cases", [])
    if not test_cases:
        print("⚠️ No test cases found in config.")
        return

    print(f"📦 Loaded {len(test_cases)} test cases from {config_path}")
    print("🚀 Starting Modal test runner...")
    
    model = Model()
    passed = 0
    failed = 0

    for idx, case in enumerate(test_cases):
        inputs = case.get("inputs", {})
        expected_error = case.get("error_contains")
        
        print("\n" + "="*80)
        print(f"🧪 Test Case {idx + 1}/{len(test_cases)}")
        print(f"   Inputs: {inputs}")
        if expected_error:
            print(f"   Expected Error: '{expected_error}'")
        print("="*80)

        try:
            # We call the remote predict method
            outputs = model.predict.remote(
                workflow_json=inputs.get("workflow_json", ""),
                input_file_url=inputs.get("input_file"),
                return_temp_files=inputs.get("return_temp_files", False),
                output_format=inputs.get("output_format", "webp"),
                output_quality=inputs.get("output_quality", 95),
                randomise_seeds=inputs.get("randomise_seeds", True),
                force_reset_cache=inputs.get("force_reset_cache", False),
            )
            
            if expected_error:
                print(f"❌ Test Case {idx + 1} FAILED: Expected error containing '{expected_error}' but run succeeded.")
                failed += 1
            else:
                print(f"✅ Test Case {idx + 1} PASSED! Received {len(outputs)} output file(s).")
                for o_idx, out in enumerate(outputs):
                    # Save a preview locally
                    save_path = f"test_output_{idx+1}_{o_idx+1}.bin"
                    with open(save_path, "wb") as sf:
                        sf.write(out)
                    print(f"   Saved output preview to {save_path}")
                passed += 1

        except Exception as e:
            err_msg = str(e)
            if expected_error:
                if expected_error.lower() in err_msg.lower():
                    print(f"✅ Test Case {idx + 1} PASSED! Failed as expected with: '{err_msg}'")
                    passed += 1
                else:
                    print(f"❌ Test Case {idx + 1} FAILED: Run failed with unexpected error: '{err_msg}' (expected to contain '{expected_error}')")
                    failed += 1
            else:
                print(f"❌ Test Case {idx + 1} FAILED with error: '{err_msg}'")
                failed += 1

    print("\n" + "#"*80)
    print(f"📊 Test Execution Summary:")
    print(f"   Passed: {passed}/{len(test_cases)}")
    print(f"   Failed: {failed}/{len(test_cases)}")
    print("#"*80)
    
    if failed > 0:
        sys.exit(1)
