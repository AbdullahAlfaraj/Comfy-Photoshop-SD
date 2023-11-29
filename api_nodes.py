# from ..comfyui_controlnet_aux.node_wrappers.openpose import OpenPose_Preprocessor

import folder_paths
import json
import comfy.samplers
import comfy.sample
import nodes
from comfy_extras.nodes_mask import (
    ImageToMask,
    ImageCompositeMasked,
    LatentCompositeMasked,
)

import torch
import numpy as np
from PIL import Image, ImageOps
import hashlib
import os
from PIL.PngImagePlugin import PngImageFile, PngInfo


class LoadImageWithMetaData:
    @classmethod
    def INPUT_TYPES(s):
        # input_dir = folder_paths.get_input_directory()
        # files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        # print("***files: ",files)
        return {
            "required": {
                "image_path": (
                    "STRING",
                    {
                        "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                        "default": "Hello World!",
                    },
                ),
            },
            # {"image": (sorted(files), )},
            #    "hidden": {"image_path": "PROMPT",}
        }

    CATEGORY = "Auto-Photoshop-SD"

    OUTPUT_NODE = True
    # RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_TYPES = ()
    FUNCTION = "load_image"

    def load_image(self, image_path):
        # image_path = folder_paths.get_annotated_filepath(image)
        # image_path = image
        print("***image_path: ", image_path)

        # Open the image file
        image_temp = Image.open(image_path)

        # Check if the image is a PNG file
        if isinstance(image_temp, PngImageFile):
            # Get the metadata from the image
            metadata = image_temp.info

            print("metadata:", metadata)

            # Print the metadata
            for key, value in metadata.items():
                print(f"{key}: {value}")

        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if "A" in i.getbands():
            mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        # return (image, mask)

        print("type of metadata: ", type(metadata))
        print("type of prompt: ", type(metadata["prompt"]))
        print("type of workflow: ", type(metadata["workflow"]))

        return {"ui": {"prompt": metadata["prompt"], "workflow": metadata["workflow"]}}
        # return  { "prompt":metadata['prompt'],"workflow":metadata['workflow'] }


class GetConfig:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        return {
            "required": {
                "embeddings": (folder_paths.get_folder_paths("embeddings"),),
            },
            "optional": {
                "controlnet_config": (controlnet_config.copy()),
            }
            # {"image": (sorted(files), )},
        }

    CATEGORY = "Auto-Photoshop-SD"

    OUTPUT_NODE = True

    RETURN_TYPES = ()
    FUNCTION = "get_config"

    def get_config(self):
        checkpoints = folder_paths.get_filename_list("checkpoints")
        samplers = comfy.samplers.KSampler.SAMPLERS
        schedulers = comfy.samplers.KSampler.SCHEDULERS
        loras = folder_paths.get_filename_list("loras")
        latent_upscale_methods = [
            "nearest-exact",
            "bilinear",
            "area",
            "bicubic",
            "bislerp",
        ]
        latent_upscale_crop_methods = ["disabled", "center"]

        # print("checkpoints: ", checkpoints)
        return {
            "ui": {
                "checkpoints": checkpoints,
                "samplers": samplers,
                "schedulers": schedulers,
                "latent_upscale_methods": latent_upscale_methods,
                "latent_upscale_crop_methods": latent_upscale_crop_methods,
                "loras": loras,
            }
        }


import base64
from io import BytesIO


class LoadImageBase64:
    @classmethod
    def INPUT_TYPES(s):
        # input_dir = folder_paths.get_input_directory()
        # files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image_base64": (
                    "STRING",
                    {
                        "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                        "default": "",
                    },
                ),
            }
        }

    CATEGORY = "Auto-Photoshop-SD"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image_from_base64"

    def load_image_from_base64(self, image_base64):
        # Decode the base64 string
        imgdata = base64.b64decode(image_base64)

        # Open the image from memory
        i = Image.open(BytesIO(imgdata))
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        if "A" in i.getbands():
            mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        return (image, mask)


from nodes import LoraLoader  # Adjust this import statement to your project structure
import re


class LoadLorasFromPrompt:
    def __init__(self):
        self.lora_loaders = []
        self.lora_list = folder_paths.get_filename_list("loras")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    CATEGORY = "Auto-Photoshop-SD"
    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    FUNCTION = "load_loras_from_prompt"

    def extract_lora_info(self, prompt):
        # Extract LoRA info
        lora_info_list = re.findall(r"<lora:(.*?):(.*?)>", prompt)

        # Remove LoRA symbols from the prompt
        prompt_without_lora = re.sub(r"<lora:(.*?):(.*?)>", "", prompt)

        return prompt_without_lora, lora_info_list

    def load_loras_from_prompt(self, model, clip, prompt):
        # Parse the loras_prompt string
        prompt_without_lora, lora_info_list = self.extract_lora_info(prompt)

        # print("prompt:", prompt)
        # print("prompt_without_lora:", prompt_without_lora)
        # print("lora_info_list:", lora_info_list)

        out_model = model
        out_clip = clip

        # Create a LoraLoader for each lora and load it
        for lora_name, strength in lora_info_list:
            lora_name += (
                ".safetensors"  # Add the .safetensors extension to the lora_name
            )
            strength = float(strength)
            # print("lora_name:", lora_name)
            # print("type(strength):", type(strength))
            if lora_name in self.lora_list:
                lora_loader = LoraLoader()
                out_model, out_clip = lora_loader.load_lora(
                    out_model, out_clip, lora_name, strength, strength
                )
                self.lora_loaders.append((out_model, out_clip))
            else:
                print(
                    f"WARNING: The specified LoRa '{lora_name}' does not exist and will be skipped. Please ensure the LoRa name is correct and that the corresponding .safetensors file is available."
                )

        # return self.lora_loaders[-1]
        # return (out_model,out_clip)
        return (out_model, out_clip, prompt_without_lora)


import numpy as np


class GaussianLatentImage:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 512, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "Auto-Photoshop-SD"

    def generate(self, width, height, batch_size=1, seed=0):
        # Set the seed for reproducibility
        torch.manual_seed(seed)

        # Define the mean and standard deviation
        mean = 0
        var = 10
        sigma = var**0.5

        # Generate Gaussian noise
        gaussian = torch.randn((batch_size, 4, height // 8, width // 8)) * sigma + mean

        # Move the tensor to the specified device
        latent = gaussian.float().to(self.device)

        return ({"samples": latent},)


class APS_LatentBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"latent1": ("LATENT",), "latent2": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "batch"

    CATEGORY = "Auto-Photoshop-SD"

    def batch(self, latent1, latent2):
        latent1_samples = latent1["samples"]
        latent2_samples = latent2["samples"]
        if latent1_samples.shape[1:] != latent2_samples.shape[1:]:
            latent2_samples = comfy.utils.common_upscale(
                latent2_samples.movedim(-1, 1),
                latent1_samples.shape[2],
                latent1_samples.shape[1],
                "bilinear",
                "center",
            ).movedim(1, -1)
        s = torch.cat((latent1_samples, latent2_samples), dim=0)
        return ({"samples": s},)


import io
import base64
from PIL import Image, ImageFilter
from torchvision import transforms


class MaskExpansion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("IMAGE",),
                "expansion": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "expandAndBlur"

    CATEGORY = "Auto-Photoshop-SD"

    def expandAndBlur(self, **kwarg):
        mask = kwarg.get("mask")
        expansion = kwarg.get("expansion")
        blur = kwarg.get("blur")
        # print("type: mask: ", type(mask))
        expanded_mask = self.maskExpansionHandler(mask, expansion, blur)

        # print("expanded_mask:",expanded_mask)
        # print("type: expanded_mask: ", type(expanded_mask))
        return (expanded_mask,)

    def b64_2_img(self, base64_image):
        image = Image.open(io.BytesIO(base64.b64decode(base64_image.split(",", 1)[0])))
        return image

    def reserveBorderPixels(self, img, dilation_img):
        pixels = img.load()
        width, height = img.size
        dilation_pixels = dilation_img.load()
        depth = 1
        for x in range(width):
            for d in range(depth):
                dilation_pixels[x, d] = pixels[x, d]
                dilation_pixels[x, height - (d + 1)] = pixels[x, height - (d + 1)]
        for y in range(height):
            for d in range(depth):
                dilation_pixels[d, y] = pixels[d, y]
                dilation_pixels[width - (d + 1), y] = pixels[width - (d + 1), y]
        return dilation_img

    def maskExpansion(self, mask_img, mask_expansion, blur=10):
        iteration = mask_expansion
        dilated_img = self.applyDilation(mask_img, iteration)
        blurred_image = dilated_img.filter(ImageFilter.GaussianBlur(radius=blur))
        mask_with_border = self.reserveBorderPixels(mask_img, blurred_image)
        return mask_with_border

    async def base64ToPng(self, base64_image, image_path):
        base64_img_bytes = base64_image.encode("utf-8")
        with open(image_path, "wb") as file_to_save:
            decoded_image_data = base64.decodebytes(base64_img_bytes)
            file_to_save.write(decoded_image_data)

    def applyDilation(self, img, iteration=20, max_filter=3):
        dilation_img = img.copy()
        for i in range(iteration):
            dilation_img = dilation_img.filter(ImageFilter.MaxFilter(max_filter))
        return dilation_img

    def maskExpansionHandler(self, input_mask, mask_expansion, blur):
        try:
            # Check if input is a string or a tensor
            if isinstance(input_mask, str):
                self.base64ToPng(input_mask, "original_mask.png")
                mask_image = self.b64_2_img(input_mask)
            elif torch.is_tensor(input_mask):
                # Ensure the tensor is 3-dimensional
                # print("Shape of tensor: ", input_mask.size())
                # print("Number of dimensions: ", input_mask.dim())
                tensor = input_mask.squeeze(0).permute(
                    2, 0, 1
                )  # Remove batch dimension and rearrange dimensions
                transform = transforms.ToPILImage()
                mask_image = transform(tensor)
            else:
                raise ValueError(
                    "Input mask must be a base64 string or a PyTorch tensor"
                )

            expanded_mask_img = self.maskExpansion(mask_image, mask_expansion, blur)

            # Convert PIL Image to PyTorch tensor
            transform = transforms.ToTensor()
            expanded_mask_tensor = transform(expanded_mask_img)
            expanded_mask_tensor = expanded_mask_tensor.unsqueeze(0).permute(0, 2, 3, 1)

            return expanded_mask_tensor
        except:
            raise Exception(f"couldn't perform mask expansion")


preprocessor_list = [
    "None",
    "CannyEdgePreprocessor",
    "OpenposePreprocessor",
    "HEDPreprocessor",
    "FakeScribblePreprocessor",
    "InpaintPreprocessor",
    "LeReS-DepthMapPreprocessor",
    "AnimeLineArtPreprocessor",
    "LineArtPreprocessor",
    "Manga2Anime_LineArt_Preprocessor",
    "MediaPipe-FaceMeshPreprocessor",
    "MiDaS-NormalMapPreprocessor",
    "MiDaS-DepthMapPreprocessor",
    "M-LSDPreprocessor",
    "BAE-NormalMapPreprocessor",
    "OneFormer-COCO-SemSegPreprocessor",
    "OneFormer-ADE20K-SemSegPreprocessor",
    "PiDiNetPreprocessor",
    "ScribblePreprocessor",
    "Scribble_XDoG_Preprocessor",
    "SAMPreprocessor",
    "ShufflePreprocessor",
    "TilePreprocessor",
    "UniFormer-SemSegPreprocessor",
    "SemSegPreprocessor",
    "Zoe-DepthMapPreprocessor",
]

controlnet_config = {
    "CannyEdgePreprocessor": {
        "low_threshold": 100,
        "high_threshold": 200,
        "resolution": 512,
        "threshold_mapping": {
            "threshold_a": "low_threshold",
            "threshold_b": "high_threshold",
        },
        "param_config": {
            "low_threshold": {
                "type": "INT",
                "default": 100,
                "min": 0,
                "max": 255,
                "step": 1,
            },
            "high_threshold": {
                "type": "INT",
                "default": 200,
                "min": 0,
                "max": 255,
                "step": 1,
            },
        },
    },
    "OpenposePreprocessor": {
        "detect_hand": "enable",
        "detect_body": "enable",
        "detect_face": "enable",
        "resolution": 512,
    },
    "HEDPreprocessor": {"safe": "enable"},
    "FakeScribblePreprocessor": {"safe": "enable"},
    "InpaintPreprocessor": {"mask": ""},
    "LeReS-DepthMapPreprocessor": {"boost": "enable"},
    "AnimeLineArtPreprocessor": {"resolution": 512},
    "LineArtPreprocessor": {"resolution": 512, "coarse": "enable"},
    "Manga2Anime_LineArt_Preprocessor": {
        "resolution": 512,
    },
    "MediaPipe-FaceMeshPreprocessor": {
        "max_faces": 10,
        "min_confidence": 0.5,
        "resolution": 512,
        "threshold_mapping": {
            "threshold_a": "max_faces",
            "threshold_b": "min_confidence",
        },
        "param_config": {
            "max_faces": {
                "type": "INT",
                "default": 10,
                "min": 1,
                "max": 50,
                "step": 1,
            },
            "min_confidence": {
                "type": "FLOAT",
                "default": 0.5,
                "min": 0.01,
                "max": 1.0,
                "step": 0.01,
            },
        },
    },
    "MiDaS-NormalMapPreprocessor": {
        "a": np.pi * 2.0,
        "bg_threshold": 0.1,
        "resolution": 512,
        "threshold_mapping": {
            "threshold_a": "a",
            "threshold_b": "bg_threshold",
        },
        "param_config": {
            "a": {
                "type": "FLOAT",
                "default": np.pi * 2.0,
                "min": 0.0,
                "max": np.pi * 5.0,
                "step": 0.05,
            },
            "bg_threshold": {
                "type": "FLOAT",
                "default": 0.1,
                "min": 0,
                "max": 1,
                "step": 0.05,
            },
        },
    },
    "MiDaS-DepthMapPreprocessor": {
        "a": np.pi * 2.0,
        "bg_threshold": 0.1,
        "resolution": 512,
        "threshold_mapping": {
            "threshold_a": "a",
            "threshold_b": "bg_threshold",
        },
        "param_config": {
            "a": {
                "type": "FLOAT",
                "default": np.pi * 2.0,
                "min": 0.0,
                "max": np.pi * 5.0,
                "step": 0.05,
            },
            "bg_threshold": {
                "type": "FLOAT",
                "default": 0.1,
                "min": 0,
                "max": 1,
                "step": 0.05,
            },
        },
    },
    "M-LSDPreprocessor": {
        "score_threshold": 0.1,
        "dist_threshold": 0.1,
        "resolution": 512,
        "threshold_mapping": {
            "threshold_a": "score_threshold",
            "threshold_b": "dist_threshold",
        },
        "param_config": {
            "score_threshold": {
                "type": "FLOAT",
                "default": 0.1,
                "min": 0.01,
                "max": 2.0,
                "step": 0.01,
            },
            "dist_threshold": {
                "type": "FLOAT",
                "default": 0.1,
                "min": 0.01,
                "max": 20.0,
                "step": 0.01,
            },
        },
    },
    "BAE-NormalMapPreprocessor": {"resolution": 512},
    "OneFormer-COCO-SemSegPreprocessor": {"resolution": 512},
    "OneFormer-ADE20K-SemSegPreprocessor": {"resolution": 512},
    "PiDiNetPreprocessor": {"safe": "enable", "resolution": 512},
    "ScribblePreprocessor": {"resolution": 512},
    "Scribble_XDoG_Preprocessor": {
        "threshold": 32,
        "resolution": 512,
        "threshold_mapping": {
            "threshold_a": "threshold",
        },
        "param_config": {
            "threshold": {
                "type": "INT",
                "default": 32,
                "min": 1,
                "max": 64,
                "step": 64,
            },
        },
    },
    # "SAMPreprocessor": {"resolution": 512},
    "ShufflePreprocessor": {"resolution": 512},
    "TilePreprocessor": {
        "pyrUp_iters": 3,
        "resolution": 512,
        "threshold_mapping": {
            "threshold_a": "pyrUp_iters",
        },
        "param_config": {
            "pyrUp_iters": {
                "type": "INT",
                "default": 3,
                "min": 1,
                "max": 10,
                "step": 1,
            }
        },
    },
    "UniFormer-SemSegPreprocessor": {"resolution": 512},
    "SemSegPreprocessor": {"resolution": 512},
    "Zoe-DepthMapPreprocessor": {"resolution": 512},
}


def convert_number(num, num_type):
    if num_type == "INT":
        return int(num)
    elif num_type == "FLOAT":
        return float(num)
    else:
        return "Invalid number type"


class ControlnetUnit:
    def __init__(
        self,
    ):
        self.map = nodes.NODE_CLASS_MAPPINGS
        self.map_param = controlnet_config.copy()

    # @classmethod
    # def INPUT_TYPES(s):
    #     return {"required": { "image": ("IMAGE",),
    #                           "preprocessor_name": (s.preprocessor_list,)},
    #                           "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
    #                           "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
    #                           }
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "preprocessor_name": (preprocessor_list,),
                "control_net_name": (folder_paths.get_filename_list("controlnet"),),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "resolution": (
                    "INT",
                    {"default": 512, "min": 64, "max": 2048, "step": 64},
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "threshold_a": (
                    "FLOAT",
                    {
                        "default": 0.0,
                    },
                ),
                "threshold_b": (
                    "FLOAT",
                    {
                        "default": 0.0,
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("preprocessed_image", "positive", "negative")
    FUNCTION = "preprocessAndApply"

    CATEGORY = "Auto-Photoshop-SD"

    def preprocessAndApply(
        self,
        **kwargs,
    ):
        instance = self.map[kwargs["preprocessor_name"]]
        self.preprocessor = instance()
        self.method = getattr(self.preprocessor, self.preprocessor.FUNCTION)
        self.param = self.map_param.get(kwargs["preprocessor_name"], {}).copy()
        if "mask" in self.param:
            # print("mask:", kwargs["mask"])
            self.param["mask"] = kwargs["mask"]
        if "resolution" in self.param:
            # print("resolution:", kwargs["resolution"])
            self.param["resolution"] = kwargs["resolution"]
        threshold_mapping = self.param.pop("threshold_mapping", None)
        param_config = self.param.pop(
            "param_config", None
        )  # don't pass param_config to method(), delete param_config
        if threshold_mapping:
            threshold_a_param_name = threshold_mapping.get("threshold_a")
            threshold_b_param_name = threshold_mapping.get("threshold_b")
            if threshold_a_param_name and "threshold_a" in kwargs:
                value = kwargs["threshold_a"]
                var_type = param_config[threshold_a_param_name]["type"]
                converted_value = convert_number(value, var_type)
                self.param.update({threshold_a_param_name: converted_value})
            if threshold_b_param_name and "threshold_b" in kwargs:
                value = kwargs["threshold_b"]
                var_type = param_config[threshold_b_param_name]["type"]
                converted_value = convert_number(value, var_type)
                self.param.update({threshold_b_param_name: converted_value})

        res = self.method(kwargs["image"], **self.param)
        preprocessed_image = res
        if "result" in res:
            # print("res:", res)
            (preprocessed_image,) = res["result"]
            # print("type(res['result']):", type(res["result"]))
            # print("type(preprocessed_image): ", type(preprocessed_image))
        elif isinstance(res, tuple):
            (preprocessed_image,) = res

        (controlnet,) = nodes.ControlNetLoader().load_controlnet(
            kwargs["control_net_name"]
        )
        (
            new_positive,
            new_negative,
        ) = nodes.ControlNetApplyAdvanced().apply_controlnet(
            kwargs["positive"],
            kwargs["negative"],
            controlnet,
            preprocessed_image,
            kwargs["strength"],
            kwargs["start_percent"],
            kwargs["end_percent"],
        )

        return (preprocessed_image, new_positive, new_negative)


class ControlNetScript:
    @classmethod
    def INPUT_TYPES(s):
        # model_list =  folder_paths.get_filename_list("controlnet")

        model_list = ["None"] + folder_paths.get_filename_list("controlnet")
        # print("type model_list: ",type (model_list))
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "is_enabled_1": (["disable", "enable"], {"default": "disable"}),
                "preprocessor_name_1": (preprocessor_list,),
                "control_net_name_1": (model_list,),
                "strength_1": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "threshold_a_1": (
                    "FLOAT",
                    {
                        "default": 0.0,
                    },
                ),
                "threshold_b_1": (
                    "FLOAT",
                    {
                        "default": 0.0,
                    },
                ),
                "start_percent_1": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "end_percent_1": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "resolution_1": (
                    "INT",
                    {"default": 512, "min": 64, "max": 2048, "step": 64},
                ),
                "is_enabled_2": (["disable", "enable"], {"default": "disable"}),
                "preprocessor_name_2": (preprocessor_list,),
                "control_net_name_2": (model_list,),
                "strength_2": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "threshold_a_2": (
                    "FLOAT",
                    {
                        "default": 0.0,
                    },
                ),
                "threshold_b_2": (
                    "FLOAT",
                    {
                        "default": 0.0,
                    },
                ),
                "start_percent_2": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "end_percent_2": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "resolution_2": (
                    "INT",
                    {"default": 512, "min": 64, "max": 2048, "step": 64},
                ),
                "is_enabled_3": (["disable", "enable"], {"default": "disable"}),
                "preprocessor_name_3": (preprocessor_list,),
                "control_net_name_3": (model_list,),
                "strength_3": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "threshold_a_3": (
                    "FLOAT",
                    {
                        "default": 0.0,
                    },
                ),
                "threshold_b_3": (
                    "FLOAT",
                    {
                        "default": 0.0,
                    },
                ),
                "start_percent_3": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "end_percent_3": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "resolution_3": (
                    "INT",
                    {"default": 512, "min": 64, "max": 2048, "step": 64},
                ),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "mask_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "mask_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "mask_3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = (
        "preprocessed_image_1",
        "preprocessed_image_2",
        "preprocessed_image_3",
        "positive",
        "negative",
    )
    FUNCTION = "preprocessAndApply"

    CATEGORY = "Auto-Photoshop-SD"

    def preprocessAndApply(self, **kwargs):
        preprocessed_images = [kwargs.get(f"image_{i+1}", "") for i in range(3)]
        last_positive = kwargs["positive"]
        last_negative = kwargs["negative"]

        for i in range(3):
            args = {
                "image": kwargs.get(f"image_{i+1}", ""),
                "mask": kwargs.get(f"mask_{i+1}", ""),
                "preprocessor_name": kwargs.get(f"preprocessor_name_{i+1}", ""),
                "control_net_name": kwargs.get(f"control_net_name_{i+1}", ""),
                "strength": kwargs.get(f"strength_{i+1}", ""),
                "start_percent": kwargs.get(f"start_percent_{i+1}", ""),
                "end_percent": kwargs.get(f"end_percent_{i+1}", ""),
                "resolution": kwargs.get(f"resolution_{i+1}", ""),
                "threshold_a": kwargs.get(f"threshold_a_{i+1}", 0),
                "threshold_b": kwargs.get(f"threshold_b_{i+1}", 0),
                "positive": last_positive,
                "negative": last_negative,
            }

            if (
                kwargs[f"is_enabled_{i+1}"] == "enable"
                and args["preprocessor_name"] != "None"
                and args["control_net_name"] != "None"
            ):
                # load image and mask if they are file name
                if isinstance(args["image"], str) and args["image"] != "":
                    (
                        args["image"],
                        _mask,
                    ) = nodes.LoadImage().load_image(args["image"])
                if (
                    isinstance(args["mask"], str) and args["mask"] != ""
                ):  # mask is string file name
                    (
                        args["mask"],
                        _mask,
                    ) = nodes.LoadImage().load_image(args["mask"])
                    (args["mask"],) = ImageToMask().image_to_mask(args["mask"], "red")
                elif args["mask"] != "":
                    (args["mask"],) = ImageToMask().image_to_mask(args["mask"], "red")

                (
                    preprocessed_images[i],
                    last_positive,
                    last_negative,
                ) = ControlnetUnit().preprocessAndApply(**args)

        return (
            preprocessed_images[0],
            preprocessed_images[1],
            preprocessed_images[2],
            last_positive,
            last_negative,
        )


class ContentMaskLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "content_mask": (
                    ["original", "latent_noise", "latent_nothing"],
                    {"default": "original"},
                ),
                "init_image": ("IMAGE",),
                "mask": ("IMAGE",),
                "width": (
                    "INT",
                    {"default": 512, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1},
                ),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "latents",
        "original_preview",
        "latent_noise_preview",
        "latent_nothing_preview",
    )
    FUNCTION = "generateContentMaskLatent"

    CATEGORY = "Auto-Photoshop-SD"

    def generateContentMaskLatent(self, **kwargs):
        content_mask = kwargs.get("content_mask")
        init_image = kwargs.get("init_image", "")
        mask = kwargs.get("mask", "")
        width = kwargs.get("width")
        height = kwargs.get("height")
        vae = kwargs.get("vae", "")
        seed = kwargs.get("seed", 0)
        original_preview = None
        latent_noise_preview = None
        latent_nothing_preview = None
        latents = ""
        upscale_method = "nearest-exact"
        crop = "disabled"

        # self.map = nodes.NODE_CLASS_MAPPINGS['']
        (upscaled_init_image,) = nodes.ImageScale().upscale(
            init_image, upscale_method, width, height, crop
        )
        (upscaled_mask_image,) = nodes.ImageScale().upscale(
            mask, upscale_method, width, height, crop
        )
        (MASK,) = ImageToMask().image_to_mask(upscaled_mask_image, "red")
        if content_mask == "original":
            (samples,) = nodes.VAEEncode().encode(vae, upscaled_init_image)
            (latents,) = nodes.SetLatentNoiseMask().set_mask(samples, MASK)
            (original_preview,) = nodes.VAEDecode().decode(vae, latents)
        elif content_mask == "latent_noise":
            (latent_noise,) = GaussianLatentImage().generate(
                width, height, batch_size=1, seed=seed
            )
            (latent_noise_image,) = nodes.VAEDecode().decode(vae, latent_noise)
            (latent_noise_preview,) = ImageCompositeMasked().composite(
                upscaled_init_image, latent_noise_image, 0, 0, True, MASK
            )
            (latents,) = nodes.VAEEncode().encode(vae, latent_noise_preview)
            (latents,) = nodes.SetLatentNoiseMask().set_mask(latents, MASK)
        elif content_mask == "latent_nothing":
            # (latents,) = nodes.VAEEncodeForInpaint().encode(
            #     vae, upscaled_init_image, MASK, 0
            # )
            # (latent_nothing_preview,) = nodes.VAEDecode().decode(vae, latents)

            (destination,) = nodes.VAEEncode().encode(vae, upscaled_init_image)
            (source,) = nodes.EmptyLatentImage().generate(width, height)
            (latents,) = LatentCompositeMasked().composite(
                destination, source, 0, 0, True, MASK
            )
            (latents,) = nodes.SetLatentNoiseMask().set_mask(latents, MASK)
            (latent_nothing_preview,) = nodes.VAEDecode().decode(vae, latents)

        return (latents, original_preview, latent_noise_preview, latent_nothing_preview)


class APS_Seed:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = "seed"
    FUNCTION = "getSeed"

    CATEGORY = "Auto-Photoshop-SD"

    def getSeed(self, **kwargs):
        seed = kwargs.get("seed", 0)
        return (seed,)


NODE_CLASS_MAPPINGS = {
    "LoadImageWithMetaData": LoadImageWithMetaData,
    "GetConfig": GetConfig,
    "LoadImageBase64": LoadImageBase64,
    "LoadLorasFromPrompt": LoadLorasFromPrompt,
    "GaussianLatentImage": GaussianLatentImage,
    "APS_LatentBatch": APS_LatentBatch,
    "ControlnetUnit": ControlnetUnit,
    "ControlNetScript": ControlNetScript,
    "ContentMaskLatent": ContentMaskLatent,
    "APS_Seed": APS_Seed,
    "MaskExpansion": MaskExpansion,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageWithMetaData": "load Image with metadata",
    "GetConfig": "get config data",
    "LoadImageBase64": "load image from base64 string",
    "LoadLorasFromPrompt": "Load Loras From Prompt",
    "GaussianLatentImage": "Generate Latent Noise",
    "APS_LatentBatch": "Combine Multiple Latents Into Batch",
    "ControlnetUnit": "General Purpose Controlnet Unit",
    "ControlNetScript": "ControlNet Script",
    "ContentMaskLatent": "Content Mask Latent",
    "APS_Seed": "Auto-Photoshop-SD Seed",
    "MaskExpansion": "Expand and Blur the Mask",
}
