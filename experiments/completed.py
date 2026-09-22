import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("frames", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.path.exists():
        raise SystemExit(1)
    if args.path.name.endswith(".summary.json"):
        config_path = args.path.with_name(args.path.name.replace(".summary.json", ".config.json"))
        if not config_path.exists():
            raise SystemExit(1)
        summary = json.loads(args.path.read_text())
        config = json.loads(config_path.read_text())
        logged_frames = int(summary["frames_logged"])
        complete = int(config["frames"]) == args.frames
        complete = complete and (logged_frames == args.frames if args.frames > 0 else logged_frames > 0)
    elif args.path.suffix == ".jsonl":
        complete = args.frames > 0 and sum(1 for line in args.path.open() if line.strip()) == args.frames
    else:
        complete = int(json.loads(args.path.read_text())["frames"]) == args.frames
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()
