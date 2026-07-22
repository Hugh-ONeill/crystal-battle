"""Bernoulli SPRT gate for bench series (fishtest-style sequential stopping).

Tests H0: winrate <= p0 against H1: winrate >= p1 on the stream of DECIDED
games (draws are simply not counted). The queue dispenser in par_series.sh
calls this before handing out each game and stops the series on a verdict,
so a lopsided result concludes in a fraction of the fixed-n cost: resolving
0.30-vs-0.40 at alpha=beta=0.05 expects ~125 games where a fixed-n Fisher
test needs several hundred.

This gates ONE ARM's winrate against fixed thresholds — it does not test an
A/B difference directly. For A/Bs, run two gated series, or gate the
candidate against the baseline's historical band.

Usage:
  sprt.py W L P0 P1 [ALPHA BETA]      -> "continue|accept-h0|accept-h1 llr=... bounds=[B,A] n=N"
  sprt.py --expected-n P0 P1 [ALPHA BETA]  -> worst-case expected games (int)
"""
import math
import sys


def llr(wins: int, losses: int, p0: float, p1: float) -> float:
    """Log-likelihood ratio of H1 over H0 after wins/losses."""
    return (wins * math.log(p1 / p0)
            + losses * math.log((1.0 - p1) / (1.0 - p0)))


def bounds(alpha: float, beta: float) -> tuple[float, float]:
    """(lower, upper): accept H0 at/below lower, H1 at/above upper."""
    return (math.log(beta / (1.0 - alpha)),
            math.log((1.0 - beta) / alpha))


def verdict(wins: int, losses: int, p0: float, p1: float,
            alpha: float = 0.05, beta: float = 0.05) -> str:
    lo, hi = bounds(alpha, beta)
    v = llr(wins, losses, p0, p1)
    if v >= hi:
        return "accept-h1"
    if v <= lo:
        return "accept-h0"
    return "continue"


def expected_n(p0: float, p1: float,
               alpha: float = 0.05, beta: float = 0.05) -> int:
    """Worst case of the expected sample sizes under H0 and under H1 (Wald).
    The launch gate refuses a series whose game cap is below this."""
    lo, hi = bounds(alpha, beta)
    win_step = math.log(p1 / p0)
    loss_step = math.log((1.0 - p1) / (1.0 - p0))
    n_h1 = ((1.0 - beta) * hi + beta * lo) / (p1 * win_step
                                              + (1.0 - p1) * loss_step)
    n_h0 = (alpha * hi + (1.0 - alpha) * lo) / (p0 * win_step
                                                + (1.0 - p0) * loss_step)
    return math.ceil(max(n_h0, n_h1))


def _validate(p0: float, p1: float, alpha: float, beta: float):
    if not (0.0 < p0 < p1 < 1.0):
        sys.exit(f"FATAL: need 0 < p0 < p1 < 1, got p0={p0} p1={p1}")
    if not (0.0 < alpha < 0.5 and 0.0 < beta < 0.5):
        sys.exit(f"FATAL: need alpha, beta in (0, 0.5), got {alpha}, {beta}")


def main(argv):
    if argv and argv[0] == "--expected-n":
        p0, p1 = float(argv[1]), float(argv[2])
        alpha = float(argv[3]) if len(argv) > 3 else 0.05
        beta = float(argv[4]) if len(argv) > 4 else 0.05
        _validate(p0, p1, alpha, beta)
        print(expected_n(p0, p1, alpha, beta))
        return
    wins, losses = int(argv[0]), int(argv[1])
    p0, p1 = float(argv[2]), float(argv[3])
    alpha = float(argv[4]) if len(argv) > 4 else 0.05
    beta = float(argv[5]) if len(argv) > 5 else 0.05
    _validate(p0, p1, alpha, beta)
    lo, hi = bounds(alpha, beta)
    print(f"{verdict(wins, losses, p0, p1, alpha, beta)} "
          f"llr={llr(wins, losses, p0, p1):.3f} "
          f"bounds=[{lo:.3f},{hi:.3f}] n={wins + losses}")


if __name__ == "__main__":
    main(sys.argv[1:])
