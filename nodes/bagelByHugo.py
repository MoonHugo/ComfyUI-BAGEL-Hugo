import os
from dfloat11 import DFloat11Model
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights,dispatch_model

from ..BAGEL.inferencer import InterleaveInferencer

from ..BAGEL.modeling.autoencoder import load_ae

from ..BAGEL.modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from ..BAGEL.modeling.qwen2 import Qwen2Tokenizer

from ..BAGEL.data.transforms import ImageTransform
from ..BAGEL.data.data_utils import pil_img2rgb, add_special_tokens

from ..func import pil2tensor,tensor2pil,clear_memory,set_seed

import folder_paths

#获取ComfyUI的临时目录
temp_dir = os.path.join(folder_paths.get_temp_directory(),'bagel_offload')
os.makedirs(temp_dir, exist_ok=True)

                
class BagelByHugo:
    def __init__(self):
        self.loaded_model = None
        self.loaded_vae_model = None
        self.loaded_tokenizer = None
        self.loaded_vae_transform = None
        self.loaded_vit_transform = None
        self.loaded_new_token_ids = None
        self.precision = ""
        pass
    
    def unload_model(self):
        del self.loaded_model
        del self.loaded_vae_model
        del self.loaded_tokenizer
        del self.loaded_vae_transform
        del self.loaded_vit_transform
        del self.loaded_new_token_ids
        self.loaded_model = None
        self.loaded_vae_model = None
        self.loaded_tokenizer = None
        self.loaded_vae_transform = None
        self.loaded_vit_transform = None
        self.loaded_new_token_ids = None
        clear_memory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_function": (
                   ["imageEdit", "reverse"],
                ),
                "image": ("IMAGE", {"tooltip": "Input image"}),
                #"model": ("BAGEL_MODEL", {"tooltip": "BAGEL model"}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "宫崎骏动画风格",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 666,
                        "min": 0,
                        "max":2**32 - 1,
                        "step": 1,
                    },
                ),
                "cfg_text_scale": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 1.0,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": "控制模型遵循文本提示词的强度。 1.0 表示禁用文本引导。典型范围： 4.0–8.0 。（Controls how strongly the model follows the text prompt. 1.0 disables text guidance. Typical range: 4.0–8.0.）",
                    },
                ),
                "cfg_img_scale": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "控制模型保留输入图像细节的程度。 1.0 表示禁用图像引导。典型范围： 1.0–2.0 。（Controls how much the model preserves input image details. 1.0 disables image guidance. Typical range: 1.0–2.0.）",
                    },
                ),
                'cfg_interval_start':(
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "CFG作用的起始时间步。0.0表示从第一步开始应用CFG。1.0表示在最后一步应用CFG，默认值：0.0。（Start timestep for applying Classifier Free Guidance (CFG). 0.0 means CFG is applied from the first step. 1.0 means CFG is applied at the last step. Typical: 0.0）",
                    },
                ),
                'cfg_interval_end':(
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "CFG作用的结束时间步。0.0表示在第一步结束时应用CFG。1.0表示在最后一步结束时应用CFG，默认值：1.0。（End timestep for applying Classifier Free Guidance (CFG). 0.0 means CFG is applied at the end of the first step. 1.0 means CFG is applied at the end of the last step. Typical: 1.0）",
                    },
                ),
                'timestep_shift':(
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.5,
                        "tooltip": "调整去噪步骤的分布比例。数值越高，初始阶段分配更多步骤（影响整体布局）；数值越低，后期阶段分配更多步骤（提升细节质量）。（Shifts the distribution of denoising steps. Higher values allocate more steps at the start (affects layout); lower values allocate more at the end (improves details).）",
                    },  
                ),
                "num_timesteps": (
                    "INT",
                    {
                        "default": 50,
                        "min": 10,
                        "max": 100,
                        "step": 5,
                        "tooltip": "总去噪步数。典型值： 50 。（Total denoising steps. Typical: 50.）",
                    },
                ),
                "cfg_renorm_min":(
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": " CFG-Renorm 的最小值。设为 1.0 时禁用重归一化（renorm）。典型值：0。（Minimum value for CFG-Renorm. 1.0 disables renorm. Typical: 0.）",
                    },
                ),
                "cfg_renorm_type":(
                    ["global", "channel", "text_channel"],
                    {"default": "global", "tooltip": "CFG-Renorm 的类型。global 在所有标记（tokens）和通道（channels）上进行归一化（文本到图像 T2I 的默认方法）；channel 对每个标记（token）跨通道进行归一化。；text_channel 类似通道（channel）方式，但仅应用于文本条件（适合图像编辑，但可能导致模糊）。如果编辑后的图像显得模糊，请尝试使用全局（global）CFG-Renorm、降低 cfg_renorm_min 或降低 cfg_scale。(Type of CFG-Renorm. 'global' normalizes across all tokens and channels (default for text-to-image T2I). 'channel' normalizes across channels for each token. 'text_channel' is similar to 'channel' but only applies to text conditions (suitable for image editing but may cause blurriness). If edited images appear blurry, try 'global' CFG-Renorm, lower cfg_renorm_min, or reduce cfg_scale."},
                ),
                "show_thinking":(
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "是否显示思考过程。启用此选项可以在推理过程中显示模型的思考过程。（Whether to show the thinking process. Enabling this option will display the model's thinking process during inference.）",
                    }
                ),
                "text_temperature":(
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "控制文本生成中的随机性。较低的值会使输出更确定，较高的值会使输出更随机。（Controls randomness in text generation. Lower values make the output more deterministic, higher values make it more random.）",
                    }
                ),
                "max_think_token_n":(
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 4096,
                        "step": 64,
                        "tooltip": "思考过程中生成的最大token数。较高的值可能会导致更长的思考过程，但也会增加计算负担。（Maximum number of tokens generated during the thinking process. Higher values may lead to longer thinking processes but increase computational load.）",
                    }
                ),
                "do_sample":(
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "是否在文本生成中使用采样。启用此选项可以使生成的文本更加多样化。（Whether to use sampling in text generation. Enabling this option can make the generated text more diverse.）",
                    }
                ),
                "precision": (
                    ["BFloat16", "DFloat11"],
                    {
                        "default": "DFloat11",
                        "tooltip": "选择模型的精度。BFloat16 提供更高的精度，但需要更多的 GPU 内存。DFloat11 是一种更节省内存的精度，体积比原始 BFloat16 模型缩小 32%，却能生成完全一致的输出结果，并在 GPU 上高效运行。（Select the precision for the model. BFloat16 offers higher precision but requires more GPU memory. DFloat11 is a memory-efficient precision that reduces the model size by 32% compared to the original BFloat16 model, while generating consistent outputs and running efficiently on GPUs.）",
                    }
                ),
                "offload_buffers":(
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "指将数据从GPU内存卸载到CPU内存或者硬盘。启用此选项可以节省GPU内存，但可能会导致推理速度变慢。（Whether to offload data from GPU memory to CPU memory. Enabling this option can save GPU memory but may slow down inference speed.）",
                    }
                ),
                "unload_model":(
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "是否在推理后卸载模型。启用此选项可以释放GPU内存，但会导致下次推理时重新加载模型。（Whether to unload the model after inference. Enabling this option can free GPU memory but will reload the model for the next inference.）",
                    }
                ),
                
            },
        }

    RETURN_TYPES = ("IMAGE","STRING")
    RETURN_NAMES = ("image",'thinking')
    FUNCTION = "bagel"
    OUTPUT_NODE = True
    CATEGORY = "🌕HugoTools/Bagel"
  
    def bagel(self, model_function,image, prompt, seed, cfg_text_scale, cfg_img_scale, cfg_interval_start, cfg_interval_end, timestep_shift, num_timesteps, cfg_renorm_min, cfg_renorm_type,show_thinking,text_temperature,max_think_token_n,do_sample,offload_buffers,unload_model,precision):
        try:
            set_seed(seed)

            max_mem_per_gpu = "40GiB"
            
            if precision == "BFloat16":
                model_name = "BAGEL-7B-MoT"
            else:
                model_name = "BAGEL-7B-MoT-DF11"
                
            model_path = os.path.join(os.path.dirname(folder_paths.get_output_directory()),'models','bagel',model_name)
            
            #验证cfg_interval_start不能大于等于cfg_interval_end
            if cfg_interval_start >= cfg_interval_end:
                raise ValueError("cfg_interval_start不能大于等于cfg_interval_end（cfg_interval_start must be less than cfg_interval_end）")
            
            if model_function == "imageEdit":
                inference_hyper=dict(
                    max_think_token_n=max_think_token_n if show_thinking else 1024,
                    do_sample=do_sample if show_thinking else False,
                    text_temperature=text_temperature if show_thinking else 0.3,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=[cfg_interval_start, cfg_interval_end],
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )
            else:
                inference_hyper=dict(
                    max_think_token_n=max_think_token_n,
                    do_sample=do_sample,
                    text_temperature=text_temperature,
                )
            
            if self.loaded_model is None or self.precision != precision:
                if self.precision != precision:
                    self.unload_model()
                    
                
                # LLM config preparing
                llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
                llm_config.qk_norm = True
                llm_config.tie_word_embeddings = False
                llm_config.layer_module = "Qwen2MoTDecoderLayer"
                
                # ViT config preparing
                vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
                vit_config.rope = False
                vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
                if precision == "BFloat16":
                    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
                else:
                    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "vae/ae.safetensors"))
                
                config = BagelConfig(
                    visual_gen=True,
                    visual_und=True,
                    llm_config=llm_config, 
                    vit_config=vit_config,
                    vae_config=vae_config,
                    vit_max_num_patch_per_side=70,
                    connector_act='gelu_pytorch_tanh',
                    latent_patch_size=2,
                    max_latent_size=64,
                )
                
                with init_empty_weights():
                    language_model = Qwen2ForCausalLM(llm_config)
                    vit_model      = SiglipVisionModel(vit_config)
                    model          = Bagel(language_model, vit_model, config)
                    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

                
                # Tokenizer Preparing
                tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
                tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

                # Image Transform Preparing
                vae_transform = ImageTransform(1024, 512, 16)
                vit_transform = ImageTransform(980, 224, 14)
                
                if precision == "BFloat16":
                    device_map = infer_auto_device_map(
                        model,
                        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
                        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
                    )
                    print(device_map)

                    same_device_modules = [
                        'language_model.model.embed_tokens',
                        'time_embedder',
                        'latent_pos_embed',
                        'vae2llm',
                        'llm2vae',
                        'connector',
                        'vit_pos_embed'
                    ]

                    if torch.cuda.device_count() == 1:
                        first_device = device_map.get(same_device_modules[0], "cuda:0")
                        for k in same_device_modules:
                            if k in device_map:
                                device_map[k] = first_device
                            else:
                                device_map[k] = "cuda:0"
                    else:
                        first_device = device_map.get(same_device_modules[0])
                        for k in same_device_modules:
                            if k in device_map:
                                device_map[k] = first_device
                    
                    model = load_checkpoint_and_dispatch(
                        model,
                        checkpoint=os.path.join(model_path, "ema.safetensors"),
                        device_map=device_map,
                        offload_folder=temp_dir,
                        offload_buffers=offload_buffers,
                        dtype=torch.bfloat16,
                        force_hooks=True,
                    )

                else:
                    model = model.to(torch.bfloat16)
                    model.load_state_dict({
                        name: torch.empty(param.shape, dtype=param.dtype, device='cpu') if param.device.type == 'meta' else param
                        for name, param in model.state_dict().items()
                    }, assign=True)

                    DFloat11Model.from_pretrained(
                        model_path,
                        bfloat16_model=model,
                        device='cpu',
                    )
                    device_map = infer_auto_device_map(
                        model,
                        max_memory={0: max_mem_per_gpu},
                        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer", "SiglipVisionModel"],
                    )
                    print(device_map)

                    same_device_modules = [
                        'language_model.model.embed_tokens',
                        'time_embedder',
                        'latent_pos_embed',
                        'vae2llm',
                        'llm2vae',
                        'connector',
                        'vit_pos_embed'
                    ]

                    if torch.cuda.device_count() == 1:
                        first_device = device_map.get(same_device_modules[0], "cuda:0")
                        for k in same_device_modules:
                            if k in device_map:
                                device_map[k] = first_device
                            else:
                                device_map[k] = "cuda:0"
                    else:
                        first_device = device_map.get(same_device_modules[0])
                        for k in same_device_modules:
                            if k in device_map:
                                device_map[k] = first_device

                    model = dispatch_model(model, device_map=device_map, force_hooks=True)

                model = model.eval()
                self.loaded_model = model
                self.loaded_vae_model = vae_model
                self.loaded_tokenizer = tokenizer
                self.loaded_vae_transform = vae_transform
                self.loaded_vit_transform = vit_transform
                self.loaded_new_token_ids = new_token_ids
            
            
            
            inferencer = InterleaveInferencer(
                model=self.loaded_model, 
                vae_model=self.loaded_vae_model, 
                tokenizer=self.loaded_tokenizer, 
                vae_transform=self.loaded_vae_transform, 
                vit_transform=self.loaded_vit_transform, 
                new_token_ids=self.loaded_new_token_ids,
            )
            
            pil_image = pil_img2rgb(tensor2pil(image))
            
            if model_function == "imageEdit":
                output_dict = inferencer(image=pil_image, text=prompt,think=show_thinking, **inference_hyper)
                tensor_image = pil2tensor(output_dict['image'])
            elif model_function == "reverse":
                output_dict = inferencer(image=pil_image, text=prompt,think=show_thinking, understanding_output=True, **inference_hyper)
                tensor_image = image
            
            if show_thinking or model_function == "reverse":
                thinking = output_dict['text']
            else:
                thinking = ''
            self.precision = precision
            
            if unload_model and self.loaded_model is not None:
                self.unload_model()
            
            return (tensor_image,thinking)
        except Exception as e:
            raise ValueError(e)
        