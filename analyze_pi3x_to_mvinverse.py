import argparse
import json
from collections import defaultdict

import torch
from safetensors.torch import load_file

from mvinverse.models.mvinverse import MVInverse


def top_prefix(name: str) -> str:
    return name.split(".", 1)[0]


def load_checkpoint(path: str):
    if path.endswith(".safetensors"):
        checkpoint = load_file(path)
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint


def main():
    parser = argparse.ArgumentParser(description="Analyze how much of a checkpoint matches MVInverse.")
    parser.add_argument("--ckpt", required=True, help="Path to Pi3X or other checkpoint")
    parser.add_argument("--json", default="", help="Optional path to save structured results as JSON")
    args = parser.parse_args()

    model = MVInverse()
    model_state = model.state_dict()
    ckpt_state = load_checkpoint(args.ckpt)

    matched_keys = []
    shape_mismatch_keys = []
    unexpected_keys = []
    missing_keys = []

    matched_params = 0
    model_total_params = 0
    ckpt_total_params = 0
    ckpt_reused_params = 0

    model_by_prefix = defaultdict(lambda: {"keys": 0, "params": 0, "matched_keys": 0, "matched_params": 0})
    ckpt_by_prefix = defaultdict(lambda: {"keys": 0, "params": 0, "reused_keys": 0, "reused_params": 0})

    for key, tensor in model_state.items():
        numel = tensor.numel()
        model_total_params += numel
        prefix = top_prefix(key)
        model_by_prefix[prefix]["keys"] += 1
        model_by_prefix[prefix]["params"] += numel

        if key not in ckpt_state:
            missing_keys.append(key)
            continue

        if tuple(ckpt_state[key].shape) != tuple(tensor.shape):
            shape_mismatch_keys.append(key)
            continue

        matched_keys.append(key)
        matched_params += numel
        model_by_prefix[prefix]["matched_keys"] += 1
        model_by_prefix[prefix]["matched_params"] += numel

    for key, tensor in ckpt_state.items():
        numel = tensor.numel()
        ckpt_total_params += numel
        prefix = top_prefix(key)
        ckpt_by_prefix[prefix]["keys"] += 1
        ckpt_by_prefix[prefix]["params"] += numel

        if key in model_state and tuple(model_state[key].shape) == tuple(tensor.shape):
            ckpt_reused_params += numel
            ckpt_by_prefix[prefix]["reused_keys"] += 1
            ckpt_by_prefix[prefix]["reused_params"] += numel
        else:
            unexpected_keys.append(key)

    summary = {
        "model_total_keys": len(model_state),
        "ckpt_total_keys": len(ckpt_state),
        "matched_keys": len(matched_keys),
        "missing_keys": len(missing_keys),
        "shape_mismatch_keys": len(shape_mismatch_keys),
        "unexpected_keys": len(unexpected_keys),
        "model_total_params": model_total_params,
        "matched_model_params": matched_params,
        "matched_model_param_ratio": matched_params / model_total_params if model_total_params else 0.0,
        "ckpt_total_params": ckpt_total_params,
        "reused_ckpt_params": ckpt_reused_params,
        "reused_ckpt_param_ratio": ckpt_reused_params / ckpt_total_params if ckpt_total_params else 0.0,
    }

    print("=== MVInverse <- Checkpoint Compatibility ===")
    print(f"Model keys          : {summary['model_total_keys']}")
    print(f"Checkpoint keys     : {summary['ckpt_total_keys']}")
    print(f"Matched keys        : {summary['matched_keys']}")
    print(f"Missing keys        : {summary['missing_keys']}")
    print(f"Shape mismatch keys : {summary['shape_mismatch_keys']}")
    print(f"Unexpected keys     : {summary['unexpected_keys']}")
    print(f"Matched model params: {summary['matched_model_params']:,} / {summary['model_total_params']:,} ({summary['matched_model_param_ratio']:.2%})")
    print(f"Reused ckpt params  : {summary['reused_ckpt_params']:,} / {summary['ckpt_total_params']:,} ({summary['reused_ckpt_param_ratio']:.2%})")
    print()

    print("=== Model Prefix Coverage ===")
    for prefix in sorted(model_by_prefix):
        item = model_by_prefix[prefix]
        ratio = item["matched_params"] / item["params"] if item["params"] else 0.0
        print(
            f"{prefix:16s} "
            f"matched_keys={item['matched_keys']:4d}/{item['keys']:4d} "
            f"matched_params={item['matched_params']:,}/{item['params']:,} "
            f"({ratio:.2%})"
        )
    print()

    print("=== Checkpoint Prefix Reuse ===")
    for prefix in sorted(ckpt_by_prefix):
        item = ckpt_by_prefix[prefix]
        ratio = item["reused_params"] / item["params"] if item["params"] else 0.0
        print(
            f"{prefix:16s} "
            f"reused_keys={item['reused_keys']:4d}/{item['keys']:4d} "
            f"reused_params={item['reused_params']:,}/{item['params']:,} "
            f"({ratio:.2%})"
        )

    if args.json:
        payload = {
            "summary": summary,
            "model_by_prefix": dict(model_by_prefix),
            "ckpt_by_prefix": dict(ckpt_by_prefix),
            "matched_keys": matched_keys,
            "missing_keys": missing_keys,
            "shape_mismatch_keys": shape_mismatch_keys,
            "unexpected_keys": unexpected_keys,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
