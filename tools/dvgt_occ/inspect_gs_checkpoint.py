from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from dvgt_occ import DEFAULT_DVGT_OCC_CONFIG, DVGTOccConfig
from dvgt_occ.data import DVGTOccClipDataset
from dvgt_occ.training.batch import DEFAULT_CACHE_KEYS, DEFAULT_SUPERVISION_KEYS, collate_dvgt_occ_batch, move_batch_to_device
from dvgt_occ.training.loss_builder import DVGTOccLossBuilder
from dvgt_occ.models.architecture import DVGTOccModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_config(cfg: dict) -> DVGTOccConfig:
    model_cfg = dict(cfg.get("model", {}))
    base = DEFAULT_DVGT_OCC_CONFIG
    values = {
        "batch_size": int(model_cfg.get("batch_size", base.batch_size)),
        "num_frames": int(model_cfg.get("num_frames", base.num_frames)),
        "num_views": int(model_cfg.get("num_views", base.num_views)),
        "image_height": int(model_cfg.get("image_height", base.image_height)),
        "image_width": int(model_cfg.get("image_width", base.image_width)),
        "patch_size": int(model_cfg.get("patch_size", base.patch_size)),
        "token_dim": int(model_cfg.get("token_dim", base.token_dim)),
        "agg_token_dim": int(model_cfg.get("agg_token_dim", base.agg_token_dim)),
        "neck_dim": int(model_cfg.get("neck_dim", base.neck_dim)),
        "full_dim": int(model_cfg.get("full_dim", base.full_dim)),
        "dynamic_query_dim": int(model_cfg.get("dynamic_query_dim", base.dynamic_query_dim)),
        "instance_dim": int(model_cfg.get("instance_dim", base.instance_dim)),
        "motion_dim": int(model_cfg.get("motion_dim", base.motion_dim)),
        "dynamic_classes": int(model_cfg.get("dynamic_classes", base.dynamic_classes)),
        "semantic_classes": int(model_cfg.get("semantic_classes", base.semantic_classes)),
        "projected_semantic_classes": int(model_cfg.get("projected_semantic_classes", base.projected_semantic_classes)),
        "max_track_queries": int(model_cfg.get("max_track_queries", base.max_track_queries)),
        "new_queries": int(model_cfg.get("new_queries", base.new_queries)),
        "sparse_dynamic_anchors": int(model_cfg.get("sparse_dynamic_anchors", base.sparse_dynamic_anchors)),
        "global_latents": int(model_cfg.get("global_latents", base.global_latents)),
        "occ_samples_per_scale": int(model_cfg.get("occ_samples_per_scale", base.occ_samples_per_scale)),
        "gs_bias_scale": float(model_cfg.get("gs_bias_scale", base.gs_bias_scale)),
        "render_splat_radius": int(model_cfg.get("render_splat_radius", base.render_splat_radius)),
        "selected_layers": tuple(model_cfg.get("selected_layers", list(base.selected_layers))),
        "camera_view_ids": tuple(model_cfg.get("camera_view_ids", list(base.camera_view_ids))),
        "occ_grid": base.occ_grid,
    }
    return DVGTOccConfig(**values)


def build_dataset(cfg: dict, model_cfg: DVGTOccConfig, split: str, manifest: Path, output_root: Path) -> DVGTOccClipDataset:
    data_cfg = dict(cfg.get("data", {}))
    scene_ids = data_cfg.get("train_scene_ids") if split == "train" else data_cfg.get("val_scene_ids")
    return DVGTOccClipDataset(
        manifest_path=manifest,
        root=output_root,
        load_cache=True,
        load_supervision=True,
        load_scene_sam3_full=True,
        full_res_size=(model_cfg.image_height, model_cfg.image_width),
        projected_semantic_classes=model_cfg.projected_semantic_classes,
        cache_keys=DEFAULT_CACHE_KEYS,
        supervision_keys=DEFAULT_SUPERVISION_KEYS,
        scene_ids=scene_ids,
        limit=None,
    )


def tensor_stats(name: str, value: torch.Tensor) -> None:
    print(
        f"{name}: shape={tuple(value.shape)} mean={float(value.mean()):.6f} "
        f"min={float(value.min()):.6f} max={float(value.max()):.6f}"
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_cfg = build_model_config(cfg)
    manifest = args.manifest or Path(cfg.get("data", {}).get("manifest", "data/nuscenes_dvgt_v0/manifest_trainval.json"))
    output_root = args.output_root or Path(cfg.get("data", {}).get("output_root", "data/nuscenes_dvgt_v0"))

    device = torch.device(args.device)
    model = DVGTOccModel(config=model_cfg).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    model.set_train_target("gs_only")

    dataset = build_dataset(cfg, model_cfg, args.split, manifest, output_root)
    sample = dataset[args.index]
    batch = collate_dvgt_occ_batch([sample])
    batch["_train_mode"] = "v1-stable"
    batch["_mask_all_weight"] = 0.0
    batch = move_batch_to_device(batch, device)

    with torch.no_grad():
        outputs = model(batch)

    gaussians = outputs["gaussians"]
    render = outputs["render"]
    loss_builder = DVGTOccLossBuilder(config=model_cfg).to(device)
    valid = loss_builder._build_render_valid_mask(batch["points_conf"])

    print(f"scene_id={batch['scene_id'][0]} clip_id={batch['clip_id'][0]}")
    tensor_stats("points_conf", batch["points_conf"])
    tensor_stats("render_valid_mask", valid)
    print(f"render_valid_ratio={float(valid.mean()):.6f}")
    tensor_stats("gaussians.keep_score", gaussians.keep_score)
    tensor_stats("gaussians.opacity", gaussians.opacity)
    tensor_stats("gaussians.scale", gaussians.scale)
    tensor_stats("gaussians.feat_dc", gaussians.feat_dc)
    tensor_stats("render.render_rgb_all", render.render_rgb_all)
    tensor_stats("render.render_alpha_all", render.render_alpha_all)
    tensor_stats("render.render_alpha_dynamic", render.render_alpha_dynamic)
    valid_rgb = render.render_rgb_all * valid
    tensor_stats("render.render_rgb_all_valid_only", valid_rgb)


if __name__ == "__main__":
    main()
