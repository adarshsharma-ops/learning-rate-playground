"""Compare gradient descent learning rates using only Python's standard library."""

import argparse
import csv
import math
from pathlib import Path


def train(rate, steps=20, x=5.0, target=15.0, initial_weight=2.0):
    """Return measurements before training and after every weight update."""
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    if steps < 1 or steps > 1000:
        raise ValueError("Steps must be between 1 and 1000.")
    if not all(math.isfinite(v) for v in (x, target, initial_weight)) or x == 0:
        raise ValueError("Inputs must be finite and x must be nonzero.")
    weight = initial_weight
    rows = []
    for step in range(steps + 1):
        prediction = x * weight
        error = prediction - target
        loss = error * error
        gradient = 2 * error * x
        if not all(math.isfinite(v) for v in (weight, prediction, loss, gradient)):
            break  # Keep the last finite measurement if training overflows.
        rows.append(dict(step=step, weight=weight, prediction=prediction,
                         loss=loss, gradient=gradient))
        weight = weight - rate * gradient
    return rows


def write_chart(runs, path):
    """Create a portable SVG chart with a log10(1 + loss) vertical scale."""
    colors = ["#2563eb", "#16a34a", "#9333ea", "#d97706", "#dc2626", "#0891b2"]
    max_step = max(rows[-1]["step"] for _, rows in runs)
    top = max(1, max(math.log10(1 + row["loss"]) for _, rows in runs for row in rows))
    height = 440 + len(runs) * 25
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Arial, sans-serif" fill="#172033">',
             '<text x="80" y="32" font-size="22">Learning rate and training loss</text>',
             '<text x="80" y="55" font-size="13">Lower is better. Vertical scale: log10(1 + loss).</text>']
    for tick in range(6):
        y = 360 - tick * 56
        parts.extend([f'<line x1="80" y1="{y}" x2="850" y2="{y}" stroke="#e2e8f0"/>',
                      f'<text x="65" y="{y + 5}" text-anchor="end" font-size="12">{top * tick / 5:.1f}</text>'])
        x = 80 + tick * 154
        parts.append(f'<text x="{x}" y="382" text-anchor="middle" font-size="12">{max_step * tick / 5:g}</text>')
    parts.append('<text x="465" y="408" text-anchor="middle" font-size="14">Weight updates</text>')
    for i, (rate, rows) in enumerate(runs):
        color = colors[i % len(colors)]
        points = " ".join(f'{80 + r["step"] / max_step * 770:.2f},{360 - math.log10(1 + r["loss"]) / top * 280:.2f}' for r in rows)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for r in rows:
            px = 80 + r["step"] / max_step * 770
            py = 360 - math.log10(1 + r["loss"]) / top * 280
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.5" fill="{color}"><title>rate={rate:g}, step={r["step"]}, loss={r["loss"]:.6g}</title></circle>')
        parts.append(f'<text x="80" y="{440 + i * 25}" fill="{color}" font-size="14">rate {rate:g}: final loss {rows[-1]["loss"]:.6g}</text>')
    parts.append('</g></svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.001, 0.01, 0.02, 0.04, 0.1])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--x", type=float, default=5.0)
    parser.add_argument("--target", type=float, default=15.0)
    parser.add_argument("--weight", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    try:
        runs = [(rate, train(rate, args.steps, args.x, args.target, args.weight)) for rate in args.rates]
        if any(not rows for _, rows in runs):
            raise ValueError("Initial values overflow. Choose smaller inputs.")
    except ValueError as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "measurements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rate", "step", "weight", "prediction", "loss", "gradient"])
        writer.writeheader()
        for rate, rows in runs:
            writer.writerows(dict(rate=rate, **row) for row in rows)
    write_chart(runs, args.output / "comparison.svg")
    print(f"{'Rate':>8} {'Updates':>8} {'Final weight':>16} {'Final loss':>16}")
    for rate, rows in runs:
        last = rows[-1]
        print(f"{rate:8g} {last['step']:8d} {last['weight']:16.6g} {last['loss']:16.6g}")
        if last["step"] < args.steps:
            print("  Stopped early because the next measurement overflowed.")
    print(f"\nOpen {args.output / 'comparison.svg'} to view the chart.")


if __name__ == "__main__":
    main()
