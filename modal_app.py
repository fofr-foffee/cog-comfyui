try:
    import torch
    if not hasattr(torch, "float8_e8m0fnu"):
        torch.float8_e8m0fnu = None
except ImportError:
    pass

try:
    import transformers
    import transformers.modeling_utils
    for cls in [transformers.PreTrainedModel, transformers.modeling_utils.ModuleUtilsMixin]:
        if not hasattr(cls, "get_head_mask"):
            def get_head_mask(self, head_mask, num_hidden_layers, is_attention_probs=False):
                if head_mask is None:
                    return [None] * num_hidden_layers
                return head_mask
            cls.get_head_mask = get_head_mask

    if hasattr(transformers.modeling_utils.ModuleUtilsMixin, "get_extended_attention_mask"):
        orig_get_extended_attention_mask = transformers.modeling_utils.ModuleUtilsMixin.get_extended_attention_mask
        def tolerant_get_extended_attention_mask(self, attention_mask, input_shape, *args, **kwargs):
            dtype = kwargs.get("dtype", None)
            device = kwargs.get("device", None)
            for arg in args:
                if isinstance(arg, torch.device) or (isinstance(arg, str) and arg in ["cuda", "cpu"]):
                    device = arg
                elif isinstance(arg, torch.dtype):
                    dtype = arg
            extended_mask = orig_get_extended_attention_mask(self, attention_mask, input_shape, dtype=dtype)
            if device is not None:
                extended_mask = extended_mask.to(device=device)
            return extended_mask
        transformers.modeling_utils.ModuleUtilsMixin.get_extended_attention_mask = tolerant_get_extended_attention_mask
except Exception as e:
    pass

try:
    import comfy.ops
    if hasattr(comfy.ops, "pick_operations"):
        orig_pick_operations = comfy.ops.pick_operations
        def tolerant_pick_operations(weight_dtype, compute_dtype, *args, **kwargs):
            kwargs.pop("scaled_fp8", None)
            return orig_pick_operations(weight_dtype, compute_dtype, *args, **kwargs)
        comfy.ops.pick_operations = tolerant_pick_operations
except Exception as e:
    pass


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
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "DOWNLOAD_LATEST_WEIGHTS_MANIFEST": "true",
        "YOLO_CONFIG_DIR": "/tmp/Ultralytics",
    })
    .add_local_dir(".", remote_path="/root/comfyui", ignore=["venv", ".git"])
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
        import sys
        if "/root/comfyui" not in sys.path:
            sys.path.insert(0, "/root/comfyui")
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


@app.function(
    gpu="A100",
    timeout=3600,
)
@modal.web_server(8188, startup_timeout=180)
def webui():
    """
    Launch the ComfyUI WebUI server on Modal.
    Exposes ComfyUI on port 8188 and prints the public URL.
    """
    import subprocess
    import os
    import sys
    
    if "/root/comfyui" not in sys.path:
        sys.path.insert(0, "/root/comfyui")
    os.chdir("/root/comfyui")
    from predict import ALL_DIRECTORIES
    for directory in ALL_DIRECTORIES:
        os.makedirs(directory, exist_ok=True)
        
    print("🚀 Starting ComfyUI WebUI on port 8188...")
    # Start ComfyUI in the foreground or keep the function active
    # Using Popen is fine as long as Modal waits for the web port 8188 to open
    subprocess.Popen(
        "python ./ComfyUI/main.py --output-directory /tmp/outputs --input-directory /tmp/inputs --disable-metadata --listen 0.0.0.0 --port 8188",
        shell=True
    )


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
            workflow_json_val = inputs.get("workflow_json", "")
            if "raw.githubusercontent.com/replicate/cog-comfyui" in workflow_json_val:
                filename = workflow_json_val.split("/")[-1]
                local_path = Path("examples/api_workflows") / filename
                if local_path.exists():
                    print(f"📂 Intercepted remote URL. Loading local workflow file: {local_path}")
                    with open(local_path, "r") as lf:
                        workflow_json_val = lf.read()

            # We call the remote predict method
            outputs = model.predict.remote(
                workflow_json=workflow_json_val,
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
