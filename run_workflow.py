"""
Helper script to run custom ComfyUI workflows on Modal and download outputs.

Usage:
  venv/bin/python run_workflow.py --json my_workflow.json [--input https://example.com/input.jpg] [--format png]
"""
import argparse
import modal
import os

def main():
    parser = argparse.ArgumentParser(description="Run custom ComfyUI workflows on Modal.")
    parser.add_argument("--json", required=True, help="Path to your custom ComfyUI workflow JSON file.")
    parser.add_argument("--input", help="Optional URL to an input image/file.")
    parser.add_argument("--format", default="webp", help="Desired output format (webp, png, jpg).")
    parser.add_argument("--quality", type=int, default=95, help="Output quality (1-100).")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json):
        print(f"❌ Error: Workflow file not found at {args.json}")
        return

    with open(args.json, "r") as f:
        workflow_content = f.read()

    print("🚀 Connecting to your ComfyUI application on Modal...")
    try:
        Model = modal.Cls.lookup("comfyui-modal", "Model")
    except Exception as e:
        print("❌ Error: Could not look up Modal app. Make sure it is deployed first using:")
        print("   venv/bin/python -m modal deploy modal_app.py")
        print(f"   Details: {e}")
        return

    print("⏳ Invoking workflow on the serverless GPU container...")
    try:
        model_instance = Model()
        outputs = model_instance.predict.remote(
            workflow_json=workflow_content,
            input_file_url=args.input,
            output_format=args.format,
            output_quality=args.quality,
        )
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        return

    if not outputs:
        print("⚠️ No output files were received.")
        return

    print(f"✅ Received {len(outputs)} output file(s). Saving...")
    os.makedirs("outputs", exist_ok=True)
    for idx, out in enumerate(outputs):
        # Resolve correct extension dynamically based on image magic bytes
        if out.startswith(b"\x89PNG\r\n\x1a\n"):
            ext = "png"
        elif out.startswith(b"RIFF") and len(out) >= 12 and out[8:12] == b"WEBP":
            ext = "webp"
        elif out.startswith(b"\xff\xd8\xff"):
            ext = "jpg"
        else:
            ext = args.format
            
        save_path = f"outputs/output_{idx + 1}.{ext}"
        with open(save_path, "wb") as f:
            f.write(out)
        print(f"   Saved to {save_path}")

if __name__ == "__main__":
    main()
