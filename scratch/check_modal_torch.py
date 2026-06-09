import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "wget", "libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("huggingface_hub[hf-transfer]", "modal")
)

app = modal.App(name="check-torch-env", image=image)

@app.function(gpu="A100")
def check_env():
    import torch
    import torchvision
    print("==================================================")
    print(f"Torch Version: {torch.__version__}")
    print(f"Torch CUDA: {torch.version.cuda}")
    print(f"Torch CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Torchvision Version: {torchvision.__version__}")
    print(f"Has float8_e8m0fnu: {hasattr(torch, 'float8_e8m0fnu')}")
    print(f"All float8 dtypes in torch: {[x for x in dir(torch) if 'float8' in x]}")
    print("==================================================")

@app.local_entrypoint()
def main():
    check_env.remote()
