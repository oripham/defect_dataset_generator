from dataclasses import dataclass, field
from typing import List

DEFAULT_PROMPTS = [
    # 整形不良
    'a small sink mark depression on molded plastic surface, subtle indentation, realistic',
    'a localized shrinkage defect on injection molded part, slight dent, realistic',
    'a weld line mark on plastic surface, thin visible seam, realistic',
    'a flow mark on molded plastic, faint streaking pattern, localized, realistic',
    'a short shot defect on plastic part, incomplete fill at edge, realistic',
    'a cold slug mark on injection molded surface, irregular rough patch, realistic',
    'a jetting defect on plastic surface, snake-like flow pattern, localized, realistic',
    'a delamination defect on molded part, thin layer separation on surface, realistic',
    'a ripple or orange peel texture defect on plastic, localized uneven surface, realistic',
    # バリ
    'a thin plastic flash burr along the parting line, sharp protrusion, realistic',
    'a small burr on molded part edge, excess material overhang, realistic',
    'a micro burr on injection molded edge, hairline protrusion, realistic',
    # ヒケ
    'a sink mark on the surface of injection molded plastic, shallow depression, realistic',
    'a thermal shrinkage pit on molded resin surface, small and shallow, realistic',
    'a dimple-like sink defect on plastic, smooth concave depression, realistic',
    # 気泡・ボイド
    'a small void or bubble defect on plastic surface, pinhole, realistic',
    'a blister defect on molded part, small raised bubble, localized, realistic',
    'a micro-void cluster on resin surface, tiny pitted area, realistic',
    # 傷・スクラッチ
    'a fine hairline scratch on plastic surface, linear mark, shallow, realistic',
    'a surface scuff mark on molded component, localized abrasion, realistic',
    'a gouge mark on plastic surface, slightly deeper linear scratch, realistic',
    'a tool mark scratch on injection molded part, short directional line, realistic',
    # 変色・焼け
    'a discoloration spot on plastic surface, localized color change, realistic',
    'a burn mark on plastic from overheating, localized brown discoloration, realistic',
    'a yellowing discoloration on white plastic surface, small patch, realistic',
    'a silver streak discoloration on dark plastic, localized light streak, realistic',
    # 欠け・クラック
    'a hairline crack on molded plastic surface, thin line fracture, realistic',
    'a chip or notch defect on plastic edge, small material loss, realistic',
    'a stress crack on injection molded part, small radiating fracture, realistic',
    # 異物
    'a foreign particle embedded in plastic surface, small dark speck, realistic',
    'a metal chip inclusion on molded part surface, tiny metallic fragment, realistic',
    'a dust particle contamination on product surface, small visible speck, realistic',
    'a hair strand inclusion in molded plastic, thin filament embedded in surface, realistic',
    'a black spot foreign material on plastic surface, localized, realistic',
    'a fiber contamination embedded in resin surface, short thread visible, realistic',
    'a rubber particle inclusion on plastic surface, small dark foreign body, realistic',
    'a carbon black speck contamination on light plastic, small dark dot, realistic',
    'a grease or oil stain on plastic surface, small irregular spot, realistic',
    'a sand grain or mineral particle on molded surface, tiny protrusion, realistic',
]


@dataclass
class ClassConfig:
    name: str = 'defect'
    class_id: int = 0
    mask_dir: str = ''
    ref_dir: str = ''
    num_images: int = 20
    prompts: List[str] = field(default_factory=lambda: [
        'a defect on a product surface, realistic'
    ])
    negative_prompt: str = 'clean, smooth, perfect, no defect, pattern, texture overlay, full surface coverage, blur'
    guidance_scale: float = 7.5
    steps: int = 30
    strength: float = 0.75
    ip_scale: float = 0.35
    controlnet_scale: float = 0.45
    inject_alpha: float = 0.75
    mask_rotation_min: float = 0.0
    mask_rotation_max: float = 0.0
    has_reference_images: bool = True
    mixed_images_dir: str = ''
    defect_mask_dir: str = ''
    good_pool_dir: str = ''


    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__annotations__}

    @classmethod
    def from_dict(cls, data: dict):
        valid_keys = set(cls.__annotations__.keys())
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

class AppState:
    def __init__(self):
        self.language: str = 'ja'
        self.server_url: str = 'http://localhost:8005'
        self.api_key: str = ''
        self.connected: bool = False
        self.job_name: str = 'my_job'
        self.good_images_path: str = ''
        self.output_root: str = ''
        self.classes: List[ClassConfig] = []
        self.model_name: str = 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1'
        self.device: str = 'cuda'
        self.image_width: int = 1024
        self.image_height: int = 1024
        self._config_save_path: str = ''
        self._current_job_id: str = ''

    def build_api_config(self) -> dict:
        return {
            'model_name': self.model_name,
            'device': self.device,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'classes': [
                {
                    'name': c.name,
                    'class_id': c.class_id,
                    'mask_dir': c.name,
                    'ref_dir': c.name,
                    'generation': {
                        'num_images': c.num_images,
                        'prompts': c.prompts,
                        'prompt': c.prompts[0] if c.prompts else 'a defect on a product surface, realistic',
                        'negative_prompt': c.negative_prompt,
                        'guidance_scale': c.guidance_scale,
                        'steps': c.steps,
                        'strength': c.strength,
                        'ip_scale': c.ip_scale,
                        'controlnet_scale': c.controlnet_scale,
                        'inject_alpha': c.inject_alpha,
                        'mask_rotation_min': c.mask_rotation_min,
                        'mask_rotation_max': c.mask_rotation_max,
                    },
                }
                for c in self.classes
            ],
        }

    def build_config_dict(self) -> dict:
        return {
            'paths': {
                'good_images': '<server-job-dir>/good_images',
                'mask_root': '<server-job-dir>/mask_root',
                'defect_refs': '<server-job-dir>/defect_refs',
                'output_root': '<server-job-dir>/output',
            },
            'model': {
                'base_model': self.model_name,
                'device': self.device,
            },
            'product': {
                'image_size': [self.image_width, self.image_height],
            },
            'classes': [
                {
                    'name': c.name,
                    'class_id': c.class_id,
                    'mask_dir': c.name,
                    'ref_dir': c.name,
                    'generation': {
                        'num_images': c.num_images,
                        'prompts': c.prompts,
                        'prompt': c.prompts[0] if c.prompts else 'a defect on a product surface, realistic',
                        'negative_prompt': c.negative_prompt,
                        'guidance_scale': c.guidance_scale,
                        'steps': c.steps,
                        'strength': c.strength,
                        'ip_scale': c.ip_scale,
                        'controlnet_scale': c.controlnet_scale,
                        'inject_alpha': c.inject_alpha,
                        'mask_rotation_min': c.mask_rotation_min,
                        'mask_rotation_max': c.mask_rotation_max,
                    },
                }
                for c in self.classes
            ],
        }

    def to_dict(self) -> dict:
        data = self.__dict__.copy()
        data['classes'] = [c.to_dict() for c in self.classes]
        return data

    @classmethod
    def from_dict(cls, data: dict):
        obj = cls()
        for k, v in data.items():
            if k == 'classes':
                obj.classes = [ClassConfig.from_dict(c) for c in v]
            elif hasattr(obj, k):
                setattr(obj, k, v)
        return obj

    def reset(self):
        """全設定をリセットして新規プロジェクト状態にする（サーバー接続情報は保持）。"""
        url = self.server_url
        api_key = self.api_key
        connected = self.connected
        lang = self.language
        self.__init__()
        self.server_url = url
        self.api_key = api_key
        self.connected = connected
        self.language = lang
