"""
Module Categorizer for Component-Level Profiling

Categorizes model modules into logical groups (vision_encoder, attention,
feedforward, etc.) using regex pattern matching on module paths. Supports
all 9 VLM families in this benchmark.
"""

import re
from typing import Dict, List, Type


class ModuleCategorizer:
    """
    Categorizes PyTorch modules into logical component groups.

    Uses regex pattern matching on module names to identify their role.
    Extended with patterns for all VLM families: SmolVLM, InternVL2.5,
    Qwen2.5-VL, Gemma3, LFM2-VL, moondream, Ovis2, FastVLM, nanoVLM.
    """

    def __init__(self):
        self.patterns = {
            # Vision encoder
            "vision_encoder": [
                r"vision_tower\.",
                r"visual_encoder\.",
                r"vision_model\.",
                r"vision_backbone\.",
                r"image_encoder\.",
                r"encoder\.layers",
                r"encoder\.blocks",
                r"visual\.transformer",
                r"davit",
                r"vision_encoder\.",
                r"vision\.",
                r"visual\.",
                r"img_encoder",
                # Qwen2.5-VL
                r"visual\.blocks",
                r"visual\.patch_embed",
                # moondream
                r"vision\.encoder",
                # FastVLM MobileCLIP
                r"mobileclip",
            ],
            "vision_embeddings": [
                r"image_position_embed",
                r"image_pos_embed",
                r"patch_embed",
                r"visual_temporal",
                r"pos_embed",
                r"cls_token",
                r"vision_model\.embeddings\.",
            ],

            # Projection layers (vision -> text space)
            "projection": [
                r"multi_modal_projector",
                r"image_projection",
                r"mm_projector",
                r"vision_projection",
                r"image_proj",
                r"connector\.",
                r"modality_projection",
                r"mlp1\.",  # InternVL2.5 projector
                r"visual_tokenizer",  # Ovis2
            ],

            # Text embeddings
            "text_embeddings": [
                r"\.embed_tokens",
                r"\.word_embeddings",
                r"\.token_embedding",
                r"\.wte",
                r"language_model\.model\.embed_tokens",
                r"lm\.embed_tokens",
            ],

            # Attention
            "attention": [
                r"\.self_attn\.",
                r"\.cross_attention\.",
                r"\.attn\.",
                r"attention\.attention",
                r"window_attn",
                r"channel_attn",
            ],

            # Feedforward / MLP
            "feedforward": [
                r"\.mlp\.",
                r"\.ffn\.",
                r"\.feed_forward\.",
                r"intermediate\.",
                r"fc1|fc2",
            ],

            # Normalization — must catch layernorm (no underscore) and layer_norm
            "normalization": [
                r"layernorm",          # catches input_layernorm, post_attention_layernorm
                r"layer_norm",         # catches layer_norm variants
                r"\.norm\d*\b",        # catches .norm, .norm1, .norm2 at word boundary
                r"\.ln_",
                r"batch_norm",
                r"group_norm",
                r"rms_norm",
                r"_norm$",             # catches final_norm, model.norm
            ],

            # Output head
            "output": [
                r"lm_head",
                r"\.head$",
                r"output_projection",
                r"classifier",
            ],
        }

        self.ignore_patterns = [
            r"^$",
            r"language_model$",
            r"vision_tower$",
            r"vision_model$",
            r"vision_encoder$",
            r"model$",
        ]

    def categorize(self, module_path: str, module_type: Type) -> str:
        for pattern in self.ignore_patterns:
            if re.search(pattern, module_path):
                return "ignore"

        for category, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, module_path):
                    return category

        return "other"

    def register_pattern(self, pattern: str, category: str):
        if category not in self.patterns:
            self.patterns[category] = []
        self.patterns[category].append(pattern)

    def categorize_model(self, model) -> Dict[str, List[str]]:
        """Categorize all leaf modules in a model."""
        categorized = {}
        for name, module in model.named_modules():
            if len(list(module.children())) > 0:
                continue
            category = self.categorize(name, type(module))
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(name)
        return categorized

    def print_categorization(self, model):
        categorized = self.categorize_model(model)
        print("\nModule Categorization Summary:")
        print("=" * 80)
        sorted_cats = sorted(categorized.items(), key=lambda x: len(x[1]), reverse=True)
        for category, modules in sorted_cats:
            if category == "ignore":
                continue
            print(f"\n{category}: {len(modules)} modules")
            for mod in modules[:3]:
                print(f"  - {mod}")
            if len(modules) > 3:
                print(f"  ... and {len(modules) - 3} more")
        print("=" * 80)

    def __repr__(self) -> str:
        return f"ModuleCategorizer(categories={len(self.patterns)})"
