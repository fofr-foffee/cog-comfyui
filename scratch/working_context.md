# Active Working Context - ComfyUI Modal Migration

## Current Status & Success
We have successfully diagnosed and resolved all runtime blocks:
1. **Torch FP8 AttributeError in `transformers>=4.49.0`:** 
   We implemented a monkey-patch `torch.float8_e8m0fnu = None`.
2. **CUDA Allocator Mismatch (`INTERNAL ASSERT FAILED`):**
   * *Problem:* Placing the monkey-patch at the very top of `ComfyUI/main.py` prematurely imported `torch` before ComfyUI's `cuda_malloc` module could set the `PYTORCH_CUDA_ALLOC_CONF` environment variable. This caused a mismatch between load-time and runtime memory allocators in PyTorch 2.6.0.
   * *Solution:* We removed the monkey-patch from the top of `ComfyUI/main.py` and placed it inside the main block *immediately after* `import cuda_malloc` executes:
     ```python
     import cuda_malloc
     if "rocm" in cuda_malloc.get_torch_version_noimport():
         os.environ['OCL_SET_SVM_SIZE'] = '262144'

     # Applied after cuda_malloc sets up the environment variables
     import torch
     if not hasattr(torch, "float8_e8m0fnu"):
         torch.float8_e8m0fnu = None
     ```
   * *Result:* ComfyUI server now starts up beautifully on Modal, and all `transformers`-dependent custom nodes import cleanly and successfully!
3. **Sequential Run OutOfMemory Error (GPU VRAM Leak):**
   * *Problem:* When running the 16 test cases sequentially on the same Modal class container instance, models from prior runs (WAN 2.1, FLUX, LivePortrait, etc.) accumulate in VRAM and cause a `torch.OutOfMemoryError` on Test 4.
   * *Solution:* Added a `free_gpu_memory` method to the `ComfyUI` class in `comfyui.py` which issues a `POST` request to `/free` with `{"unload_models": True, "free_memory": True}` and called it in the `cleanup` method. This frees GPU memory and unloads previous models at the start of every prediction.
4. **Buffalo_l Download 404 Error:**
   * *Problem:* A previous change directed InsightFace's `buffalo_l` model downloads to a non-existent directory on the `lithiumice/insightface` repository, causing 404 entry not found errors.
   * *Solution:* Reverted the `buffalo_l` repository back to the working and official `immich-app/buffalo_l` repository with flat filenames.

---

## Active Task
* **Background Task ID:** `7509cad1-8e63-45b4-8f9a-7486bfb7cd1d/task-125`
* **Command running:** `venv/bin/python -m modal run modal_app.py`
* **Progress:** Running the sequential test cases with the GPU memory clearing and InsightFace downloader fixes active.
* **Log File:** `/home/openclaw/.gemini/antigravity-cli/brain/7509cad1-8e63-45b4-8f9a-7486bfb7cd1d/.system_generated/tasks/task-125.log`

---

## Next Steps Upon Restart
1. **Check Task Results:** Check the output of background task `7509cad1-8e63-45b4-8f9a-7486bfb7cd1d/task-125` using the command `tail -n 100 /home/openclaw/.gemini/antigravity-cli/brain/7509cad1-8e63-45b4-8f9a-7486bfb7cd1d/.system_generated/tasks/task-125.log` (or wait for the task-completed notification).
2. **Review passing test cases:** Ensure all 16 test cases execute perfectly.
3. **Commit & Push:** Once verified, stage all changes, commit, and push them to the GitHub repository.
4. **Deploy / Serve ComfyUI WebUI:** Serve the ComfyUI web server on Modal using `venv/bin/python -m modal serve modal_app.py` to verify public web UI accessibility.
