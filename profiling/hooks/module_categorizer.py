"""
Module Categorizer for Component-Level Profiling

Categorizes model modules into logical groups using regex pattern matching
on module paths. Supports all 9 VLM families in this benchmark.

Categories (11):
  Vision side:  vision_attention, vision_mlp, vision_normalization,
                vision_embeddings, vision_encoder (catch-all)
  Bridge:       projection
  Text side:    text_embeddings, attention, feedforward, normalization, output
"""

import re
from typing import Dict, List, Type


class ModuleCategorizer:
    """
    Categorizes PyTorch modules into logical component groups.

    Uses regex pattern matching on module names to identify their role.
    Order matters: more specific patterns (vision sub-categories) are checked
    before the broad vision_encoder catch-all.
    """

    def __init__(self):
        # NOTE: Order matters! More specific patterns must come BEFORE
        # the broad vision_encoder catch-all.
        self.patterns = self._build_ordered_patterns()

        self.ignore_patterns = [
            r"^$",
            r"language_model$",
            r"vision_tower$",
            r"vision_model$",
            r"vision_encoder$",
            r"model$",
        ]

    def _build_ordered_patterns(self) -> Dict[str, List[str]]:
        """Build patterns in priority order (specific before general)."""
        from collections import OrderedDict
        patterns = OrderedDict()

        # ── Vision sub-components (MUST come before vision_encoder catch-all) ──

        # Vision embeddings: patch embed, position embed, cls token
        patterns["vision_embeddings"] = [
            r"vision_model\.embeddings\.",
            r"visual\.patch_embed",
            r"image_position_embed",
            r"image_pos_embed",
            r"visual_temporal",
            r"vision_model\.pos_embed",
            r"vision\.encoder\.model\.visual\.cls_token",
            # Ovis2 AIMv2
            r"backbone\..*\.pos_embed",
            r"backbone\..*\.cls_token",
            r"backbone\..*\.patchify",
            # FastVLM MobileCLIP embeddings
            r"mobileclip\..*\.pos_embed",
            r"mobileclip\..*\.patch_emb",
            r"mobileclip\..*\.token_embed",
        ]

        # Vision attention: self_attn/attn inside vision encoder paths
        patterns["vision_attention"] = [
            # SigLIP / InternViT / generic vision transformer
            r"vision_model\.encoder\.layers\.\d+\.self_attn\.",
            r"vision_model\.encoder\.layers\.\d+\.attn\.",
            r"visual_encoder\.blocks\.\d+\.attn\.",
            # InternVL2.5 InternViT
            r"vision_model\.encoder\.layers\.\d+\.attention\.",
            # Qwen2.5-VL
            r"visual\.blocks\.\d+\.attn\.",
            # moondream
            r"vision\.encoder\.model\.visual\.blocks\.\d+\.attn\.",
            # Ovis2 AIMv2
            r"backbone\..*\.blocks\.\d+\.attn\.",
            # FastVLM MobileCLIP
            r"mobileclip\..*\.attn\.",
            # LFM2-VL
            r"vision_tower\..*\.self_attn\.",
            r"vision_tower\..*\.attn\.",
            # Generic: attn inside any vision path
            r"vision.*\.layers\.\d+\.self_attn\.",
            r"visual.*\.layers\.\d+\.self_attn\.",
        ]

        # Vision MLP: mlp/ffn inside vision encoder paths
        patterns["vision_mlp"] = [
            # SigLIP / InternViT / generic vision transformer
            r"vision_model\.encoder\.layers\.\d+\.mlp\.",
            r"visual_encoder\.blocks\.\d+\.mlp\.",
            # InternVL2.5 InternViT
            r"vision_model\.encoder\.layers\.\d+\.feed_forward\.",
            # Qwen2.5-VL
            r"visual\.blocks\.\d+\.mlp\.",
            # moondream
            r"vision\.encoder\.model\.visual\.blocks\.\d+\.mlp\.",
            # Ovis2 AIMv2
            r"backbone\..*\.blocks\.\d+\.mlp\.",
            # FastVLM MobileCLIP
            r"mobileclip\..*\.mlp\.",
            r"mobileclip\..*\.ffn\.",
            # LFM2-VL
            r"vision_tower\..*\.mlp\.",
            r"vision_tower\..*\.ffn\.",
            # Generic
            r"vision.*\.layers\.\d+\.mlp\.",
            r"visual.*\.layers\.\d+\.mlp\.",
        ]

        # Vision normalization: layernorm/norm inside vision encoder paths
        patterns["vision_normalization"] = [
            # SigLIP / generic
            r"vision_model\.encoder\.layers\.\d+\.layer_norm",
            r"vision_model\.post_layernorm",
            r"vision_model\.pre_layernorm",
            r"vision_model\.layernorm",
            # InternVL2.5
            r"vision_model\.encoder\.layers\.\d+\.norm",
            # Qwen2.5-VL
            r"visual\.blocks\.\d+\.norm",
            r"visual\.merger\.ln_q",
            # moondream
            r"vision\.encoder\.model\.visual\.blocks\.\d+\.norm",
            r"vision\.encoder\.model\.visual\.norm",
            # Ovis2 AIMv2
            r"backbone\..*\.blocks\.\d+\.norm",
            r"backbone\..*\.norm",
            r"backbone\..*\.post_trunk_norm",
            # FastVLM MobileCLIP
            r"mobileclip\..*norm",
            # LFM2-VL
            r"vision_tower\..*\.norm",
            r"vision_tower\..*layernorm",
            # Generic
            r"vision.*\.layers\.\d+\.norm",
            r"visual.*\.layers\.\d+\.norm",
        ]

        # Vision encoder catch-all: anything else inside vision paths
        patterns["vision_encoder"] = [
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
            r"visual\.blocks",
            r"visual\.patch_embed",
            r"vision\.encoder",
            r"mobileclip",
            r"backbone\.",
        ]

        # ── Projection layers (vision -> text space) ──
        patterns["projection"] = [
            r"multi_modal_projector",
            r"image_projection",
            r"mm_projector",
            r"vision_projection",
            r"image_proj",
            r"connector\.",
            r"modality_projection",
            r"mlp1\.",  # InternVL2.5 projector
            r"visual_tokenizer",  # Ovis2
            r"visual\.merger\.",  # Qwen2.5-VL (non-norm parts)
        ]

        # ── Text embeddings ──
        patterns["text_embeddings"] = [
            r"\.embed_tokens",
            r"\.word_embeddings",
            r"\.token_embedding",
            r"\.wte",
            r"language_model\.model\.embed_tokens",
            r"lm\.embed_tokens",
        ]

        # ── Text attention ──
        patterns["attention"] = [
            r"\.self_attn\.",
            r"\.cross_attention\.",
            r"\.attn\.",
            r"attention\.attention",
            r"window_attn",
            r"channel_attn",
        ]

        # ── Text feedforward / MLP ──
        patterns["feedforward"] = [
            r"\.mlp\.",
            r"\.ffn\.",
            r"\.feed_forward\.",
            r"intermediate\.",
            r"fc1|fc2",
        ]

        # ── Text normalization ──
        patterns["normalization"] = [
            r"layernorm",
            r"layer_norm",
            r"\.norm\d*\b",
            r"\.ln_",
            r"batch_norm",
            r"group_norm",
            r"rms_norm",
            r"_norm$",
        ]

        # ── Output head ──
        patterns["output"] = [
            r"lm_head",
            r"\.head$",
            r"output_projection",
            r"classifier",
        ]

        return patterns

    def categorize(self, module_path: str, module_type: Type) -> str:
        for pattern in self.ignore_patterns:
            if re.search(pattern, module_path):
                return "ignore"

        for category, pats in self.patterns.items():
            for pattern in pats:
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
