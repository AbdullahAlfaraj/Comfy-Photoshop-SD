ValidNodes = {
    "KSampler": {
        "inputs": {
            "seed": 'TextField',
            "steps": 'NumberField',
            "cfg": 'NumberField',
            "sampler_name": 'Menu',
            "scheduler": 'Menu',
            "denoise": 'NumberField',
        },
        "list_id": {
            "sampler_name": 'samplers',
            "scheduler": 'schedulers',
        },
    },
    "EmptyLatentImage": {
        "inputs": {
            "width": 'NumberField',
            "height": 'NumberField',
            "batch_size": 'NumberField',
        },
    },
    "CLIPTextEncode": {
        "inputs": {
            "text": 'TextArea',
        },
    },

    "LatentUpscale": {
        "inputs": {
            "upscale_method": 'Menu',
            "width": 'NumberField',
            "height": 'NumberField',
            "crop": 'Menu',
        },
        "list_id": {
            "upscale_method": 'latent_upscale_methods',
            "crop": 'latent_upscale_crop_methods',
        },
    },
    "CheckpointLoaderSimple": {
        "inputs": {
            "ckpt_name": 'Menu',
        },
        "list_id": {
            "ckpt_name": 'checkpoints',
        },
    },
    "LoadImage": {
        "inputs": {
            "image": 'ImageBase64',
        },
    },
    "LoadImageBase64": {
        "inputs": {
            "image_base64": 'ImageBase64',
        },
    },
    "LoraLoader": {
        "inputs": {
            "lora_name": 'Menu',
            "strength_model": 'NumberField',
            "strength_clip": 'NumberField',
        },
        "list_id": {
            "lora_name": 'loras',
        },
    },
}
