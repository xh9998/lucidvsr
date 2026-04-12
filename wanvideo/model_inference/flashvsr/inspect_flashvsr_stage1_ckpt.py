import argparse
from collections import Counter

from diffsynth.core import load_state_dict


def main():
    parser = argparse.ArgumentParser(description="检查 FlashVSR Stage1 训练 ckpt 内容。")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    args = parser.parse_args()

    state_dict = load_state_dict(args.checkpoint_path, device="cpu")
    keys = sorted(state_dict.keys())

    lq_proj_keys = [key for key in keys if key.startswith("lq_proj_in.")]
    lora_keys = [key for key in keys if "lora_" in key]
    other_keys = [key for key in keys if key not in set(lq_proj_keys) and key not in set(lora_keys)]

    prefix_counter = Counter(key.split(".", 1)[0] for key in keys)

    print(f"checkpoint: {args.checkpoint_path}")
    print(f"total_keys: {len(keys)}")
    print(f"lq_proj_keys: {len(lq_proj_keys)}")
    print(f"lora_keys: {len(lora_keys)}")
    print(f"other_keys: {len(other_keys)}")
    print()
    print("top_level_prefix:")
    for prefix, count in sorted(prefix_counter.items()):
        print(f"  {prefix}: {count}")

    if lq_proj_keys:
        print()
        print("sample_lq_proj_keys:")
        for key in lq_proj_keys[:10]:
            print(f"  {key}")

    if lora_keys:
        print()
        print("sample_lora_keys:")
        for key in lora_keys[:20]:
            print(f"  {key}")

    if other_keys:
        print()
        print("sample_other_keys:")
        for key in other_keys[:20]:
            print(f"  {key}")


if __name__ == "__main__":
    main()
