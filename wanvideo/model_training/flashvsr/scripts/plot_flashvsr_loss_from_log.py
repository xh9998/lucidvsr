import argparse
import csv
import os
import re
from xml.sax.saxutils import escape


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


def write_svg(points, svg_path: str, title: str):
    if not points:
        return False

    os.makedirs(os.path.dirname(svg_path), exist_ok=True)
    width = 960
    height = 540
    margin_left = 70
    margin_right = 20
    margin_top = 50
    margin_bottom = 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    steps = [step for step, _ in points]
    losses = [loss for _, loss in points]
    min_step, max_step = min(steps), max(steps)
    min_loss, max_loss = min(losses), max(losses)
    if max_step == min_step:
        max_step += 1
    if max_loss == min_loss:
        max_loss += 1e-6

    def x_map(step):
        return margin_left + (step - min_step) / (max_step - min_step) * plot_w

    def y_map(loss):
        return margin_top + (max_loss - loss) / (max_loss - min_loss) * plot_h

    polyline = " ".join(f"{x_map(step):.2f},{y_map(loss):.2f}" for step, loss in points)

    y_ticks = 5
    x_ticks = min(6, len(points))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-size="22" font-family="Arial, sans-serif">{escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#222" stroke-width="2"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#222" stroke-width="2"/>',
    ]

    for i in range(y_ticks + 1):
        frac = i / y_ticks
        y = margin_top + frac * plot_h
        loss = max_loss - frac * (max_loss - min_loss)
        lines.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_w}" y2="{y:.2f}" stroke="#ddd" stroke-width="1"/>')
        lines.append(f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial, sans-serif" fill="#444">{loss:.4f}</text>')

    for i in range(x_ticks):
        frac = 0 if x_ticks == 1 else i / (x_ticks - 1)
        step = min_step + frac * (max_step - min_step)
        x = margin_left + frac * plot_w
        lines.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_h}" stroke="#eee" stroke-width="1"/>')
        lines.append(f'<text x="{x:.2f}" y="{margin_top + plot_h + 22}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#444">{int(round(step))}</text>')

    lines.append(f'<polyline fill="none" stroke="#1565c0" stroke-width="2.5" points="{polyline}"/>')
    for step, loss in points:
        lines.append(f'<circle cx="{x_map(step):.2f}" cy="{y_map(loss):.2f}" r="2.5" fill="#1565c0"/>')
    lines.append(f'<text x="{width/2:.1f}" y="{height - 15}" text-anchor="middle" font-size="14" font-family="Arial, sans-serif">step</text>')
    lines.append(
        f'<text x="18" y="{height/2:.1f}" text-anchor="middle" font-size="14" font-family="Arial, sans-serif" transform="rotate(-90 18 {height/2:.1f})">loss</text>'
    )
    lines.append("</svg>")

    with open(svg_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
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
    svg_path = os.path.join(args.output_dir, "loss_curve.svg")
    write_csv(points, csv_path)
    has_png = write_png(points, png_path, args.title)
    has_svg = write_svg(points, svg_path, args.title)

    print(f"num_points={len(points)}")
    print(f"csv={csv_path}")
    if has_png:
        print(f"png={png_path}")
    if has_svg:
        print(f"svg={svg_path}")


if __name__ == "__main__":
    main()
