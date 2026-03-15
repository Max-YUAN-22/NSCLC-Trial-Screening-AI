"""
McNemar test for paired correctness: rule-based vs rule-based+Safety Agent on 120 pairs.
Reads final_120_reviewed_predictions.csv (columns: baseline_label, safety_label, gold_label).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis"


def mcnemar_exact(b: int, c: int) -> float:
    """McNemar p-value (exact binomial): probability of observing >= max(b,c) discordant in b+c under p=0.5."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    p = 0.0
    for i in range(k, n + 1):
        p += comb(n, i) * (0.5 ** n)
    return 2 * min(p, 1 - p)


def main():
    import pandas as pd
    path = ANALYSIS_DIR / "final_120_reviewed_predictions.csv"
    if not path.exists():
        print(f"Missing {path}")
        return
    df = pd.read_csv(path)
    gold = df["gold_label"].str.strip().str.lower()
    base = df["baseline_label"].str.strip().str.lower()
    safe = df["safety_label"].str.strip().str.lower()
    correct_base = (gold == base)
    correct_safe = (gold == safe)
    # 2x2: rows = baseline correct (T/F), cols = safety correct (T/F)
    # b = baseline wrong & safety correct, c = baseline correct & safety wrong
    b = ((~correct_base) & correct_safe).sum()
    c = (correct_base & (~correct_safe)).sum()
    a = (correct_base & correct_safe).sum()
    d = ((~correct_base) & (~correct_safe)).sum()
    print("Paired correctness: Rule-based vs Rule+Safety Agent (n=120)")
    print("                    Safety correct   Safety wrong")
    print(f"  Baseline correct   {a:>4}              {c:>4}")
    print(f"  Baseline wrong     {b:>4}              {d:>4}")
    print(f"  Discordant pairs: baseline wrong & safety correct = {b}, baseline correct & safety wrong = {c}")
    p = mcnemar_exact(int(b), int(c))
    print(f"  McNemar (exact binomial) p = {p:.4f}")
    if p < 0.05:
        print("  -> Improvement with Safety Agent is statistically significant (p < 0.05).")
    else:
        print("  -> p >= 0.05 (report exact p in manuscript).")


if __name__ == "__main__":
    main()
