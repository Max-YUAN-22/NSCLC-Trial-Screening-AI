"""
Compute 95% Wilson CIs for main cohort metrics (accuracy, false inclusion, uncertain rate).
Output: numbers to paste into main.tex Table 2 and Results paragraphs.
"""

from pathlib import Path


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return max(0.0, centre - half), min(1.0, centre + half)


def main():
    # From main.tex: dev 30 (20 inel), held-out 90 (66 inel), combined 120 (86 inel)
    # Rule-based: dev 2/20 FI, acc 0.47; held-out 8/66 FI, acc 0.60; combined 10/86 FI, acc 0.57
    # Safety: dev 0/20 FI, acc 0.60; held-out 1/66 FI, acc 0.67; combined 1/86 FI, acc 0.65
    # Single-agent: dev 0/20, acc 0.63; held-out 0/66, acc 0.78; combined 0/86, acc 0.74, unc 7.5%
    # M1: combined 4/86 FI, acc 0.63, unc 25%; M2: 3/86 FI, acc 0.63, unc 25.8%

    rows = [
        ("Rule-based combined", "accuracy", 0.57 * 120, 120),
        ("Rule-based combined", "FI", 10, 86),
        ("Rule+Safety combined", "accuracy", 0.65 * 120, 120),
        ("Rule+Safety combined", "FI", 1, 86),
        ("Rule-based held-out", "FI", 8, 66),
        ("Rule+Safety held-out", "FI", 1, 66),
        ("Single-agent combined", "accuracy", 0.74 * 120, 120),
        ("Single-agent combined", "uncertain", 0.075 * 120, 120),
        ("Single-agent combined", "FI", 0, 86),
        ("M1 combined", "accuracy", 0.63 * 120, 120),
        ("M1 combined", "FI", 4, 86),
        ("M2 combined", "accuracy", 0.63 * 120, 120),
        ("M2 combined", "FI", 3, 86),
    ]
    print("95% Wilson CI (proportion)\n")
    for name, metric, succ, n in rows:
        s = int(round(succ))
        lo, hi = wilson(s, n)
        if "accuracy" in metric or "uncertain" in metric:
            print(f"{name} {metric}: {s}/{n} -> {lo:.3f}--{hi:.3f}")
        else:
            print(f"{name} {metric}: {s}/{n} -> {100*lo:.1f}--{100*hi:.1f}%")


if __name__ == "__main__":
    main()
