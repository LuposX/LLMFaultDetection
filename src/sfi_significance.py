"""Statistical Fault Injection sizing, after Leveugle et al. (DATE 2009).

Applies equations (1) and (2) of "Statistical Fault Injection: Quantified
Error and Confidence" to this project's fault model.

Equation (1) — required sample size:
    n = N * t^2 * p*(1-p) / (e^2 * (N-1) + t^2 * p*(1-p))
Equation (2) — margin of error for a given n:
    e = t * sqrt(p*(1-p)/n) * sqrt((N-n)/(N-1))

Population N here = number of distinct fault configurations in the
stratum:
- k=1: N = number of flippable bit positions (per scope)
- k>1: N = C(bits, k) (exact-count draws; astronomically large)

Hypotheses check (paper section II.A):
- uniform sampling: yes, sample_fault draws positions uniformly (torch
  generator = Mersenne-Twister, the algorithm the paper recommends);
- sampling fraction << 5%: yes for every stratum (n/N ~ 1e-8 or smaller),
  so the finite-population correction is negligible and results match the
  infinite-population formula t^2 p(1-p)/e^2;
- p = 0.5 is the conservative worst case (paper II.C).

One caveat beyond the paper: our detection criterion evaluates a FIXED
100-probe workload, so these margins quantify fault-sampling error only.
The prompt set adds a separate (prompt-sampling) error term for the
single-prompt-rate metric; coverage-by->=1-probe has the workload built
into the detector definition and only carries fault-sampling error.

Run: python src/sfi_significance.py [--n 100]
"""

import argparse
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = "Qwen/Qwen3-0.6B"

# t quantiles for two-sided confidence levels (Normal distribution)
T = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758, 0.998: 3.0902}
ELEMENTS = 596_049_920        # unique bf16 parameters of Qwen3-0.6B
BITS = {"all": ELEMENTS * 16, "mantissa": ELEMENTS * 7,
        "exponent": ELEMENTS * 8}


def required_n(N, e, t, p=0.5):
    """Paper eq. (1): minimum sample size for margin e at confidence t."""
    return math.ceil(N * t * t * p * (1 - p)
                     / (e * e * (N - 1) + t * t * p * (1 - p)))


def margin(N, n, t, p=0.5):
    """Paper eq. (2): margin of error achieved with sample size n."""
    return t * math.sqrt(p * (1 - p) / n) * math.sqrt((N - n) / (N - 1))


def log10_comb(n_bits, k):
    """log10 of C(n_bits, k) for the k-flip population size."""
    return (math.lgamma(n_bits + 1) - math.lgamma(k + 1)
            - math.lgamma(n_bits - k + 1)) / math.log(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=385,
                    help="faults per stratum in the current campaign")
    args = ap.parse_args()

    print(f"model: {MODEL}")
    print(f"bf16 elements: {ELEMENTS:,}")
    for scope, b in BITS.items():
        print(f"  {scope:8s} bits: {b:,}")
    print(f"\nk>1 population sizes (log10 of C(bits, k)):")
    for scope in ("mantissa", "exponent"):
        for k in (2, 4, 8, 10, 20, 50):
            print(f"  k={k:2d} {scope:8s}: N ~ 10^{log10_comb(BITS[scope], k):.0f}")

    print("\n== required faults per stratum (paper eq. 1, p=0.5) ==")
    print(f"{'margin e':>9s} | " + " | ".join(f"{c*100:>4.1f}% conf" for c in T))
    for e in (0.05, 0.02, 0.01):
        row = " | ".join(f"{required_n(1e12, e, t):>8d}" for t in T.values())
        print(f"  {e*100:4.0f}%   | " + row)
    print("(N >= 1e12 makes the finite-population correction vanish; true "
          "for every stratum here, so these n are exact for our model)")

    print(f"\n== margins achieved with n={args.n} faults per stratum "
          "(eq. 2, p=0.5 worst case) ==")
    for n in (args.n, 385, 9604):
        row = " | ".join(f"{margin(1e12, n, t)*100:5.1f}%" for t in T.values())
        print(f"  n={n:>5d}: {row}")
    print("\nexample at non-worst-case p (mantissa k=1 observed coverage "
          "was ~0.95):")
    for n in (args.n,):
        print(f"  n={n}, p=0.95, 95% conf: margin = "
              f"{margin(1e12, n, 1.96, p=0.95)*100:.1f}%")


if __name__ == "__main__":
    main()
