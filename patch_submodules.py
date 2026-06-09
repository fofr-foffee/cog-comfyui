import os
import sys

def patch_file(filepath, target, replacement, description):
    if not os.path.exists(filepath):
        print(f"⚠️ Warning: File to patch not found at {filepath}")
        return False
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    if replacement in content:
        print(f"✅ Patch already applied to {filepath} ({description})")
        return True
        
    if target not in content:
        print(f"❌ Error: Target signature not found in {filepath} ({description})")
        return False
        
    print(f"⏳ Applying patch to {filepath} ({description})...")
    patched_content = content.replace(target, replacement)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(patched_content)
    print(f"✅ Patch successfully applied to {filepath} ({description})")
    return True

def main():
    print("🛠️ Running ComfyUI serverless submodule patcher...")
    
    # ----------------------------------------------------
    # Patch 1: ComfyUI/server.py (Prompt Interceptor)
    # ----------------------------------------------------
    server_py = "ComfyUI/server.py"
    target_server = """        @routes.post("/prompt")
        async def post_prompt(request):
            logging.info("got prompt")
            json_data =  await request.json()
            json_data = self.trigger_on_prompt(json_data)"""
            
    replacement_server = """        @routes.post("/prompt")
        async def post_prompt(request):
            logging.info("got prompt")
            json_data =  await request.json()

            # --- CUSTOM INTERCEPTOR HOOK FOR DYNAMIC DOWNLOADING ---
            try:
                if "prompt" in json_data:
                    prompt_wf = json_data["prompt"]
                    import sys
                    import os
                    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                    if root_dir not in sys.path:
                        sys.path.insert(0, root_dir)
                    
                    from comfyui import ComfyUI as CustomComfyUI
                    custom_comfy = CustomComfyUI("127.0.0.1:8188")
                    custom_comfy.input_directory = "/tmp/inputs"
                    custom_comfy.output_directory = "/tmp/outputs"
                    
                    logging.info("Custom Interceptor: Pre-checking inputs and weights for the workflow...")
                    custom_comfy.handle_inputs(prompt_wf)
                    custom_comfy.handle_weights(prompt_wf)
                    logging.info("Custom Interceptor: All inputs and weights ready!")
            except Exception as e:
                logging.error(f"Custom Interceptor Error: Failed to pre-download inputs/weights: {e}")
            # --------------------------------------------------------

            json_data = self.trigger_on_prompt(json_data)"""
            
    patch_file(server_py, target_server, replacement_server, "Prompt Downloader Interceptor")

    # ----------------------------------------------------
    # Patch 2: ComfyUI/main.py (Torch & Transformers Compatibility)
    # ----------------------------------------------------
    main_py = "ComfyUI/main.py"
    target_main = """    if "rocm" in cuda_malloc.get_torch_version_noimport():
        os.environ['OCL_SET_SVM_SIZE'] = '262144'  # set at the request of AMD"""
        
    replacement_main = """    if "rocm" in cuda_malloc.get_torch_version_noimport():
        os.environ['OCL_SET_SVM_SIZE'] = '262144'  # set at the request of AMD

    # --- CUSTOM PYTORCH & TRANSFORMERS MONKEY PATCHES ---
    import torch
    if not hasattr(torch, "float8_e8m0fnu"):
        torch.float8_e8m0fnu = None

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
        import logging
        logging.warning(f"Failed to monkey-patch transformers: {e}")

    try:
        import comfy.ops
        if hasattr(comfy.ops, "pick_operations"):
            orig_pick_operations = comfy.ops.pick_operations
            def tolerant_pick_operations(weight_dtype, compute_dtype, *args, **kwargs):
                kwargs.pop("scaled_fp8", None)
                return orig_pick_operations(weight_dtype, compute_dtype, *args, **kwargs)
            comfy.ops.pick_operations = tolerant_pick_operations
    except Exception as e:
        import logging
        logging.warning(f"Failed to monkey-patch comfy.ops.pick_operations: {e}")
    # -----------------------------------------------------"""
    
    patch_file(main_py, target_main, replacement_main, "Torch & Transformers compatibility")

    # ----------------------------------------------------
    # Patch 3: ComfyUI-Kolors-MZ configuration_chatglm.py (use_cache parameter)
    # ----------------------------------------------------
    kolors_config_py = "ComfyUI/custom_nodes/ComfyUI-Kolors-MZ/chatglm3/configuration_chatglm.py"
    target_kolors_sig = """        quantization_bit=0,
        pre_seq_len=None,
        prefix_projection=False,
        **kwargs
    ):"""
    replacement_kolors_sig = """        quantization_bit=0,
        pre_seq_len=None,
        prefix_projection=False,
        use_cache=True,
        **kwargs
    ):"""
    patch_file(kolors_config_py, target_kolors_sig, replacement_kolors_sig, "Kolors signature patch")
    
    target_kolors_body = """        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        super().__init__(**kwargs)"""
    replacement_kolors_body = """        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        self.use_cache = use_cache
        super().__init__(**kwargs)"""
    patch_file(kolors_config_py, target_kolors_body, replacement_kolors_body, "Kolors use_cache assignment")

    # ----------------------------------------------------
    # Patch 4: ComfyUI-ReActor nodes.py (ultralytics separation & restore_swapped_only)
    # ----------------------------------------------------
    reactor_py = "ComfyUI/custom_nodes/ComfyUI-ReActor/nodes.py"
    target_reactor_ultralytics = """if "ultralytics" not in folder_paths.folder_names_and_paths:
    add_folder_path_and_extensions("ultralytics_bbox", [os.path.join(models_dir, "ultralytics", "bbox")], folder_paths.supported_pt_extensions)"""
    
    replacement_reactor_ultralytics = """if "ultralytics_bbox" not in folder_paths.folder_names_and_paths:
    add_folder_path_and_extensions("ultralytics_bbox", [os.path.join(models_dir, "ultralytics", "bbox")], folder_paths.supported_pt_extensions)
if "ultralytics_segm" not in folder_paths.folder_names_and_paths:
    add_folder_path_and_extensions("ultralytics_segm", [os.path.join(models_dir, "ultralytics", "segm")], folder_paths.supported_pt_extensions)
if "ultralytics" not in folder_paths.folder_names_and_paths:
    add_folder_path_and_extensions("ultralytics", [os.path.join(models_dir, "ultralytics")], folder_paths.supported_pt_extensions)"""
    
    patch_file(reactor_py, target_reactor_ultralytics, replacement_reactor_ultralytics, "ReActor Ultralytics folders separate registration")

    target_reactor_options = """                "source_faces_index": ("STRING", {"default": "0"}),
                "detect_gender_source": (["no","female","male"], {"default": "no"}),
                "console_log_level": ([0, 1, 2], {"default": 1}),
                "restore_swapped_only": ("BOOLEAN", {"default": True, "label_off": "no", "label_on": "yes"})
            }
        }"""
        
    replacement_reactor_options = """                "source_faces_index": ("STRING", {"default": "0"}),
                "detect_gender_source": (["no","female","male"], {"default": "no"}),
                "console_log_level": ([0, 1, 2], {"default": 1}),
            },
            "optional": {
                "restore_swapped_only": ("BOOLEAN", {"default": True, "label_off": "no", "label_on": "yes"})
            }
        }"""
        
    patch_file(reactor_py, target_reactor_options, replacement_reactor_options, "ReActorOptions restore_swapped_only optionality")

    target_reactor_exec = """    def execute(self,input_faces_order, input_faces_index, detect_gender_input, source_faces_order, source_faces_index, detect_gender_source, console_log_level, restore_swapped_only):"""
    replacement_reactor_exec = """    def execute(self,input_faces_order, input_faces_index, detect_gender_input, source_faces_order, source_faces_index, detect_gender_source, console_log_level, restore_swapped_only=True):"""
    patch_file(reactor_py, target_reactor_exec, replacement_reactor_exec, "ReActorOptions execute method parameter optionality")

if __name__ == "__main__":
    main()
