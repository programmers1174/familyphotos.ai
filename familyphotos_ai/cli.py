from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_enroll_config
from .images import iter_image_files
from .pipeline import build_references, load_references_json, save_references_json, scan_images


def _cmd_enroll(args: argparse.Namespace) -> int:
    cfg = load_enroll_config(args.enroll_yaml)
    refs = build_references(cfg)
    save_references_json(args.out_refs, cfg, refs)
    print(f"Wrote references: {args.out_refs}")
    print(f"People enrolled: {len(refs)}")
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    if args.refs_json:
        cfg, refs = load_references_json(args.refs_json)
        if args.threshold is not None:
            cfg = cfg.__class__(model=cfg.model, matching=cfg.matching.__class__(threshold=float(args.threshold)), people=cfg.people)
    else:
        cfg = load_enroll_config(args.enroll_yaml)
        refs = build_references(cfg)

    image_paths = iter_image_files(args.images)
    matches = scan_images(image_paths, cfg=cfg, references=refs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2)

    print(f"Scanned images: {len(image_paths)}")
    print(f"Wrote matches: {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="familyphotos-ai", description="Stage 1: identify people in photos using DeepFace embeddings.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_enroll = sub.add_parser("enroll", help="Create reference embeddings from enrollment YAML.")
    p_enroll.add_argument("--enroll-yaml", required=True, help="Path to enrollment YAML (see enroll.sample.yaml).")
    p_enroll.add_argument("--out-refs", default="artifacts/references.json", help="Where to write reference embeddings JSON.")
    p_enroll.set_defaults(func=_cmd_enroll)

    p_scan = sub.add_parser("scan", help="Scan a folder (recursive) and output matches per image.")
    group = p_scan.add_mutually_exclusive_group(required=True)
    group.add_argument("--refs-json", help="Precomputed references JSON from `enroll`.")
    group.add_argument("--enroll-yaml", help="Enrollment YAML (will enroll on the fly).")
    p_scan.add_argument("--images", required=True, help="Folder to scan recursively, or a single image path.")
    p_scan.add_argument("--out", default="artifacts/matches.json", help="Where to write matches JSON.")
    p_scan.add_argument("--threshold", type=float, default=None, help="Override matching threshold (lower is stricter).")
    p_scan.set_defaults(func=_cmd_scan)

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

