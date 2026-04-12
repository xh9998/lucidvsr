import argparse
import csv
import os
import re


LOSS_PATTERN = re.compile(r"step=(\d+)\s+loss=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def parse_loss_points(log_path: str):
    points = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            match = LOSS_PATTERN.search(line)
            if match is None:
                continue
            step = int(match.group(1))
            loss = float(match.group(2))
            points.append((step, loss))
    return points


def write_csv(points, csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "loss"])
        writer.writerows(points)


def write_png(points, png_path: str, title: str):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable, skip png: {exc}")
        return False

    steps = [step for step, _ in points]
    losses = [loss for _, loss in points]
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, losses, marker="o", linewidth=1.5, markersize=3)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="从 FlashVSR 训练 run.log 提取 loss 并导出 csv/png。")
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--title", type=str, default="FlashVSR Training Loss")
    args = parser.parse_args()

    points = parse_loss_points(args.log_path)
    if not points:
        raise ValueError(f"No loss points found in log_path={args.log_path}")

    csv_path = os.path.join(args.output_dir, "loss_points.csv")
    png_path = os.path.join(args.output_dir, "loss_curve.png")
    write_csv(points, csv_path)
    has_png = write_png(points, png_path, args.title)

    print(f"num_points={len(points)}")
    print(f"csv={csv_path}")
    if has_png:
        print(f"png={png_path}")


if __name__ == "__main__":
    main()
