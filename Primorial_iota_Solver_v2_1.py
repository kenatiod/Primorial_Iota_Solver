#!/usr/bin/env python3
"""
Primorial_iota_Solver_v2_0.py
=============================

Exact, windowed computation of

    a(n) = iota_min(n) = min { pi(gpf(m(m+1))) : P_n <= m(m+1) < P_{n+1} }

for all primorial intervals n <= N in ONE pass, by a level-incremental
Størmer/Lehmer Pell enumeration whose continued fractions are truncated
by a rigorous size cap.

Why this is exact (the cap theorem)
-----------------------------------
Every consecutive pair (m, m+1) of p_S-smooth integers arises from a
squarefree D | P_S and an index k >= 1 with

    x_k + y_k*sqrt(D) = eps_D^k,   x_k = 2m+1,   4m(m+1) = D*y_k^2,

where eps_D = x_1 + y_1*sqrt(D) is the least positive solution of
x^2 - D y^2 = 1.  If m <= U_N (the largest m with m(m+1) < P_{N+1}) then
x_1 <= x_k = 2m+1 <= 2U_N + 1 =: CAP.  Every solution is a convergent of
sqrt(D), and convergent numerators grow at least as fast as Fibonacci
numbers, so the continued fraction can be stopped as soon as a numerator
exceeds CAP: if no solution has appeared by then, x_1 > CAP and D
contributes no pair with m <= U_N.  This is a proof, not a heuristic;
it costs about log(CAP)/log(phi) ~ (theta(p_{N+1})/2)/0.48 steps per D
(roughly 120 steps for N = 30, 190 for N = 44) instead of a period that
can be ~ sqrt(D).

Level-incremental completeness
------------------------------
Call top(D) the index of the largest prime dividing D ("D-level").  A
pair of level L = pi(gpf(m(m+1))) has top(D) <= L.  Processing D-levels
s = 1, 2, 3, ... in order, after level s is complete EVERY pair with
level <= s and m <= U_N has been found.  Hence, for each interval n <= N:

    * if a found pair of level <= s lies in interval n, then
      iota_min(n) = (least level of a found pair in n)   [EXACT]
    * otherwise iota_min(n) > s                            [LOWER BOUND]

Pairs found at D-level s may have level > s (y_k carries a larger prime
<= p_MAXLEVEL); these give upper bounds only until their level is reached.

Index cap.  By Bilu-Hanrot-Voutier the Lucas sequence u_k = y_k/y_1 has a
primitive prime divisor for k > 30, and such a divisor p satisfies
p >= 2k-1 (Lehmer); so a p_S-smooth y_k has k <= max(30, (p_S+1)/2).
Together with the window this bounds the power loop.  (The loop stops at
x_k > CAP anyway; the index cap only matters for tiny regulators.)

Cache / resume
--------------
The state directory keeps
    state.json      N, CAP, MAXLEVEL, completed levels and per-level
                    completed task masks (crash-safe checkpointing)
    hits.csv        every pair found (m, D, k, level, interval, omega, PC)
    eps_cache.jsonl fundamental solutions found (x_1 <= CAP), keyed by D
Re-running with --resume continues where it stopped.  Re-running with a
LARGER --end-r raises CAP; completed levels are then re-run in extension
mode: cached D reuse eps_D and only the power loop is repeated, uncached D
(the majority, whose x_1 exceeded the old CAP) recompute their truncated
continued fraction against the new CAP.  Since the per-D cost grows only
linearly in theta(p_{N+1}), choose N generously the first time.

Stopping rule.  The run proceeds level by level until every interval
n <= N is EXACT, or level MAXLEVEL is reached, or the time budget expires.
Every completed level is a certificate of the lower bounds it implies.

Prime-complete detection.  A found pair with omega(m(m+1)) == level has
rad(m(m+1)) = P_level and is flagged PC (these are the terms of A141399).

Kernel.  Continued fractions run in pure Python (gmpy2 accelerated when
available) or, with --kernel pari, in PARI via cypari2 using the same
pellxy() routine as A002072_Solver v13.  The CF loop is the hot spot; a
C/GMP kernel would be the next speed step and can replace cf_pell_capped().

Usage
-----
    # self-test against the 18 known values (seconds):
    python Primorial_iota_Solver_v2_0.py --selftest

    # all intervals n <= 40 in one pass, 32 workers, checkpoint every 5 min:
    python Primorial_iota_Solver_v2_0.py --end-r 40 --workers 32 --tag N40

    # continue after a stop, or after a crash:
    python Primorial_iota_Solver_v2_0.py --end-r 40 --workers 32 --tag N40 --resume

    # raise N later (extension mode; cached eps_D reused):
    python Primorial_iota_Solver_v2_0.py --end-r 48 --workers 32 --tag N40 --resume

    # limit a session to 12 hours (state is saved; --resume continues):
    python Primorial_iota_Solver_v2_0.py --end-r 40 --workers 32 --tag N40 --time-budget-hours 12

Outputs (in --outdir/--tag): iota_min.csv, iota_min_table.txt, hits.csv
(every pair found, with its interval, level, D, k, omega and PC flag),
eps_cache.jsonl, state.json.  A row is EXACT when a pair of level <=
s_done lies in the interval; LOWER means iota_min(n) > s_done is
certified; BRACKET means a pair of higher level is known but the levels
between are not yet exhausted.  pc_excluded = NO means the lower bound
already exceeds n, so no prime-complete product lies in interval n.

Ken Clements, September 2026; developed with Claude (Anthropic).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from multiprocessing import get_context, cpu_count
from typing import Dict, List, Optional, Sequence, Tuple

try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass

PROGRAM_NAME = "Primorial_iota_Solver"
PROGRAM_VERSION = "2.0"

# Reference values (K. Clements, Sept 2026) used by --selftest.
KNOWN_IOTA_MIN = [1, 2, 2, 3, 3, 4, 4, 4, 7, 6, 8, 8, 10, 10, 11, 14, 13, 14,
                  17, 18, 21, 21, 23, 24, 22, 29, 30, 32, 27]

# ---------------------------------------------------------------------------
# Optional accelerators
# ---------------------------------------------------------------------------
try:
    import gmpy2
    from gmpy2 import mpz as _mpz, isqrt as _isqrt
    HAS_GMPY2 = True
except ImportError:  # pragma: no cover
    _mpz = int
    _isqrt = math.isqrt
    HAS_GMPY2 = False

try:
    import cypari2 as _cypari2
    HAS_CYPARI2 = True
except ImportError:  # pragma: no cover
    HAS_CYPARI2 = False


def mpz(x):
    return _mpz(x)


# ---------------------------------------------------------------------------
# Basic arithmetic
# ---------------------------------------------------------------------------

def first_n_primes(n: int) -> List[int]:
    primes: List[int] = []
    x = 2
    while len(primes) < n:
        lim = math.isqrt(x)
        ok = True
        for p in primes:
            if p > lim:
                break
            if x % p == 0:
                ok = False
                break
        if ok:
            primes.append(x)
        x = 3 if x == 2 else x + 2
    return primes


def first_m_with_product_at_least(T: int) -> int:
    """Smallest m >= 1 with m(m+1) >= T."""
    if T <= 2:
        return 1
    s = math.isqrt(4 * T + 1)
    m = max(1, (s - 1) // 2)
    while m * (m + 1) < T:
        m += 1
    while m > 1 and (m - 1) * m >= T:
        m -= 1
    return m


def log_bigint(n: int) -> float:
    if n <= 0:
        raise ValueError
    b = n.bit_length()
    if b <= 53:
        return math.log(n)
    shift = b - 53
    return math.log(n >> shift) + shift * math.log(2)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------

class Intervals:
    """Half-open product intervals  P_n <= m(m+1) < P_{n+1},  n = 1..N."""

    def __init__(self, N: int):
        self.N = N
        self.primes = first_n_primes(N + 2)
        self.P: List[int] = [1]            # P[0] = 1, P[n] = primorial(n)
        for p in self.primes[: N + 1]:
            self.P.append(self.P[-1] * p)
        self.L = [0] * (N + 1)
        self.U = [0] * (N + 1)
        for n in range(1, N + 1):
            self.L[n] = first_m_with_product_at_least(self.P[n])
            self.U[n] = first_m_with_product_at_least(self.P[n + 1]) - 1
        # CAP on x = 2m+1 for m <= U_N
        self.cap = 2 * self.U[N] + 1
        self._Pcut = self.P[1: N + 2]      # P_1 .. P_{N+1}

    def interval_of(self, M: int) -> Optional[int]:
        """n with P_n <= M < P_{n+1}, or None if M >= P_{N+1} or M < P_1."""
        i = bisect.bisect_right(self._Pcut, M)   # number of P_j <= M
        if i == 0 or i > self.N:
            return None
        return i


# ---------------------------------------------------------------------------
# Pell kernel (pure Python / gmpy2)
# ---------------------------------------------------------------------------

def cf_pell_capped(D: int, cap: int) -> Tuple[Optional[Tuple[int, int]], int]:
    """
    Fundamental solution (x1, y1) of x^2 - D y^2 = 1 if x1 <= cap, else None.
    Returns ((x1, y1) | None, cf_steps).  Rigorous: every solution is a
    convergent of sqrt(D) and convergent numerators are increasing, so
    the first convergent numerator above cap certifies x1 > cap.
    """
    D = mpz(D)
    cap = mpz(cap)
    a0 = _isqrt(D)
    if a0 * a0 == D:
        raise ValueError(f"D={D} is a square")
    m = mpz(0)
    d = mpz(1)
    a = a0
    p0, p1 = mpz(1), a0
    q0, q1 = mpz(0), mpz(1)
    steps = 1
    while True:
        m = d * a - m
        d = (D - m * m) // d
        a = (a0 + m) // d
        p0, p1 = p1, a * p1 + p0
        q0, q1 = q1, a * q1 + q0
        steps += 1
        if p1 > cap:
            return None, steps
        if p1 * p1 - D * q1 * q1 == 1:
            return (int(p1), int(q1)), steps


_PELL_GP_SRC = r"""
pellxy(D, max_x=0)={
  if(D<=0, error("D<=0"));
  if(issquare(D), error("D is square"));
  my(a0=sqrtint(D), m=0, d=1, a=a0, p0=1, p1=a0, q0=0, q1=1);
  while(p1^2 - D*q1^2 != 1,
    m = d*a - m;
    d = (D - m^2)/d;
    a = (a0 + m)\d;
    my(p2=a*p1+p0, q2=a*q1+q0);
    p0=p1; p1=p2; q0=q1; q1=q2;
    if(max_x>0 && p1>max_x, return([0,0]));
  );
  if(max_x>0 && p1>max_x, return([0,0]));
  [p1, q1];
};
"""

# ---------------------------------------------------------------------------
# Worker state
# ---------------------------------------------------------------------------

_W: Dict = {}


def _worker_init(primes_all: List[int], cap: int, level: int, H: int,
                 kmax: int, P_list: List[int], N: int, eps_cache: Dict[int, Tuple[int, int]],
                 kernel: str) -> None:
    """Pool initializer for one D-level."""
    try:
        sys.set_int_max_str_digits(0)
    except AttributeError:
        pass

    pari = None
    if kernel == "pari":
        if not HAS_CYPARI2:
            raise RuntimeError("--kernel pari requested but cypari2 is not importable")
        pari = _cypari2.Pari()
        try:
            pari.allocatemem(256 * 1024 * 1024, silent=True)
        except TypeError:
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                pari.allocatemem(256 * 1024 * 1024)
        pari(_PELL_GP_SRC)
        v = pari("pellxy(46)")
        assert int(v[0]) ** 2 - 46 * int(v[1]) ** 2 == 1

    primes_mpz = [mpz(p) for p in primes_all]
    ONE = mpz(1)

    if HAS_GMPY2:
        _remove = gmpy2.remove
    else:
        def _remove(n, p):  # type: ignore[misc]
            k = 0
            while n % p == 0:
                n //= p
                k += 1
            return n, k

    def factor_smooth(n):
        """n (mpz>=1) -> (is_smooth over primes_all, gpf_index (1-based), omega)."""
        gpf_idx = 0
        omega = 0
        for i, p in enumerate(primes_mpz):
            if n % p == 0:
                n, _ = _remove(n, p)
                gpf_idx = i + 1
                omega += 1
                if n <= ONE:
                    return True, gpf_idx, omega
        return n == ONE, gpf_idx, omega

    lower = primes_all[: level - 1]          # primes below p_level
    Lc = len(lower) - H
    _W.update({
        "primes_all": primes_all,
        "p_level": primes_all[level - 1],
        "level": level,
        "cap": cap,
        "capm": mpz(cap),
        "T": cap * cap,                      # prefilter: D > cap^2 => x1 > cap
        "H": H,
        "primes_low": lower[:Lc],
        "primes_high": lower[Lc:],
        "kmax": kmax,
        "P_list": P_list,                    # P_1..P_{N+1}
        "N": N,
        "factor_smooth": factor_smooth,
        "eps_cache": eps_cache,
        "pari": pari,
        "kernel": kernel,
        "ONE": ONE,
    })


def _fundamental(D: int) -> Tuple[Optional[Tuple[int, int]], int]:
    cache = _W["eps_cache"]
    if D in cache:
        return cache[D], 0
    if _W["kernel"] == "pari":
        v = _W["pari"](f"pellxy({D}, {_W['cap']})")
        x1, y1 = int(v[0]), int(v[1])
        if x1 == 0:
            return None, -1
        return (x1, y1), -1
    return cf_pell_capped(D, _W["cap"])


def _process_D(D: int, agg: Dict) -> None:
    factor_smooth = _W["factor_smooth"]
    capm = _W["capm"]
    kmax = _W["kmax"]
    P_list = _W["P_list"]
    N = _W["N"]

    sol, steps = _fundamental(D)
    if steps > 0:
        agg["cf_steps"] += steps
    if sol is None:
        agg["n_cut"] += 1
        return
    x1, y1 = sol
    Dm, x1m, y1m = mpz(D), mpz(x1), mpz(y1)
    if x1m * x1m - Dm * y1m * y1m != 1:
        raise RuntimeError(f"Pell identity failed for D={D}")
    agg["n_solved"] += 1
    agg["eps_found"].append((int(D), int(x1), int(y1)))

    # y1 gate: y1 | y_k for all k, so a non-smooth y1 kills the branch.
    sm, _, _ = factor_smooth(y1m)
    if not sm:
        agg["n_gate"] += 1
        return

    Dy1 = Dm * y1m
    x, y = x1m, y1m
    k = 1
    while x <= capm and k <= kmax:
        if x & 1:                                    # x odd <=> m integer
            sm_y, _, _ = factor_smooth(y)
            if sm_y:
                mm = (x - 1) >> 1
                M = mm * (mm + 1)
                # independent verification and level/omega of the pair
                sa, ga, oa = factor_smooth(mm)
                sb, gb, ob = factor_smooth(mm + 1)
                if not (sa and sb):
                    raise RuntimeError(f"identity violated at D={D}, k={k}")
                level = max(ga, gb)
                omega = oa + ob
                i = bisect.bisect_right(P_list, M)      # number of P_j <= M
                n = i if 1 <= i <= N else None
                if n is not None:
                    agg["hits"].append((n, level, int(mm), int(D), k, omega,
                                        int(omega == level)))
        x, y = x1m * x + Dy1 * y, x1m * y + y1m * x
        k += 1


def _task(high_mask: int):
    """One task: fixed membership over the top-H lower primes; DFS over the rest."""
    try:
        primes_low = _W["primes_low"]
        primes_high = _W["primes_high"]
        T = _W["T"]
        p_level = _W["p_level"]
        Lc = len(primes_low)

        base = p_level
        for i, p in enumerate(primes_high):
            if (high_mask >> i) & 1:
                base *= p

        agg: Dict = {"n_d": 0, "pref": 0, "n_cut": 0, "n_gate": 0, "n_solved": 0,
                     "cf_steps": 0, "hits": [], "eps_found": [], "n_err": 0,
                     "err_samples": []}

        def run_D(D: int) -> None:
            agg["n_d"] += 1
            try:
                _process_D(D, agg)
            except Exception as e:  # keep going; report samples
                agg["n_err"] += 1
                if len(agg["err_samples"]) < 5:
                    agg["err_samples"].append(f"D={D}: {e!r}")

        if base > T:
            agg["pref"] = 1 << Lc
        else:
            run_D(base)
            def rec(i: int, prod: int) -> None:
                for j in range(i, Lc):
                    np_ = prod * primes_low[j]
                    if np_ > T:
                        agg["pref"] += (1 << (Lc - j)) - 1
                        break
                    run_D(np_)
                    rec(j + 1, np_)
            rec(0, base)
        return (high_mask, agg)
    except Exception as e:  # pragma: no cover
        return (high_mask, {"n_d": 0, "pref": 0, "n_cut": 0, "n_gate": 0,
                            "n_solved": 0, "cf_steps": 0, "hits": [],
                            "eps_found": [], "n_err": 1,
                            "err_samples": [f"task {high_mask}: {e!r}"]})


# ---------------------------------------------------------------------------
# Results / state
# ---------------------------------------------------------------------------

HIT_FIELDS = ["level_done_at", "n", "level", "m", "D", "k", "omega", "pc", "found_utc"]


class Results:
    def __init__(self, iv: Intervals, maxlevel: int):
        self.iv = iv
        self.maxlevel = maxlevel
        N = iv.N
        # best hit per interval: (level, m, D, k, omega, pc)
        self.best: List[Optional[Tuple]] = [None] * (N + 1)
        self.n_hits = [0] * (N + 1)
        self.pc_hits: List[List[int]] = [[] for _ in range(N + 1)]
        self.levels_done: List[int] = []      # sorted list of completed D-levels
        self.level_stats: Dict[int, Dict] = {}
        self.seen: set = set()                # m values already recorded (dedup)

    @property
    def s_done(self) -> int:
        """Largest s with all D-levels 1..s complete."""
        s = 0
        for x in self.levels_done:
            if x == s + 1:
                s = x
            else:
                break
        return s

    def add_hit(self, h: Tuple) -> bool:
        """Record a pair; return False if this m was already recorded."""
        n, level, m, D, k, omega, pc = h
        if m in self.seen:
            return False
        self.seen.add(m)
        self.n_hits[n] += 1
        if pc:
            self.pc_hits[n].append(m)
        b = self.best[n]
        if b is None or (level, m) < (b[0], b[1]):
            self.best[n] = (level, m, D, k, omega, pc)
        return True

    def status(self, n: int) -> Tuple[str, int, Optional[int]]:
        """(status, lower_bound, exact_or_upper)."""
        s = self.s_done
        b = self.best[n]
        if b is not None and b[0] <= s:
            return "EXACT", b[0], b[0]
        if b is not None:
            return "BRACKET", s + 1, b[0]
        return "LOWER", s + 1, None

    def all_exact(self) -> bool:
        return all(self.status(n)[0] == "EXACT" for n in range(1, self.iv.N + 1))


def write_table(res: Results, path: Optional[str] = None) -> str:
    iv = res.iv
    lines = []
    lines.append(f"levels complete through s = {res.s_done}   (CAP = 2*U_{iv.N}+1, "
                 f"log CAP = {log_bigint(iv.cap):.2f})")
    lines.append(" n   p_n   status   iota_min  lower  upper  PC?  pairs  pc_hits   witness m (level, D digits, k)")
    lines.append("---  ----  -------  --------  -----  -----  ---  -----  -------   -------------------------------")
    for n in range(1, iv.N + 1):
        st, lo, up = res.status(n)
        b = res.best[n]
        exact = str(up) if st == "EXACT" else "-"
        ups = "-" if up is None else str(up)
        pc = "NO" if lo > n else "?"
        w = "-"
        if b is not None:
            w = f"{b[1]} ({b[0]}, {len(str(b[2]))}d, k={b[3]})"
        lines.append(f"{n:3d}  {iv.primes[n-1]:4d}  {st:7s}  {exact:>8}  {lo:5d}  {ups:>5}  {pc:3s}   "
                     f"{res.n_hits[n]:5d}  {len(res.pc_hits[n]):7d}  {w:<31s}  ")
    lines.append("PC? = NO means iota_min(n) > n is certified, so no prime-complete "
                 "product lies in interval n.")
    text = "\n".join(lines)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return text


def save_state(path: str, args, iv: Intervals, res: Results,
               level_progress: Dict[int, List[int]], extra: Dict) -> None:
    payload = {
        "program": f"{PROGRAM_NAME} v{PROGRAM_VERSION}",
        "saved_utc": utc_now(),
        "N": iv.N,
        "cap": str(iv.cap),
        "log_cap": log_bigint(iv.cap),
        "maxlevel": res.maxlevel,
        "levels_done": res.levels_done,
        "level_stats": {str(k): v for k, v in res.level_stats.items()},
        "level_progress": {str(k): v for k, v in level_progress.items()},
        "iota_min": {str(n): (res.status(n)[2] if res.status(n)[0] == "EXACT" else None)
                     for n in range(1, iv.N + 1)},
        "lower_bounds": {str(n): res.status(n)[1] for n in range(1, iv.N + 1)},
        "best": {str(n): ([str(x) for x in res.best[n]] if res.best[n] else None)
                 for n in range(1, iv.N + 1)},
        "n_hits": res.n_hits[1:],
        "pc_hits": {str(n): [str(m) for m in res.pc_hits[n]] for n in range(1, iv.N + 1)
                    if res.pc_hits[n]},
        "command": " ".join(sys.argv),
        **extra,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def load_hits(path: str, res: Results) -> int:
    if not os.path.isfile(path):
        return 0
    n = 0
    with open(path, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            h = (int(rec["n"]), int(rec["level"]), int(rec["m"]), int(rec["D"]),
                 int(rec["k"]), int(rec["omega"]), int(rec["pc"]))
            if 1 <= h[0] <= res.iv.N and res.add_hit(h):
                n += 1
    return n


def append_hits(path: str, hits: Sequence[Tuple], level_done_at: int) -> None:
    if not hits:
        return
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HIT_FIELDS)
        if not exists:
            w.writeheader()
        now = utc_now()
        for (n, level, m, D, k, omega, pc) in hits:
            w.writerow({"level_done_at": level_done_at, "n": n, "level": level,
                        "m": m, "D": D, "k": k, "omega": omega, "pc": pc,
                        "found_utc": now})


def load_eps_cache(path: str) -> Dict[int, Tuple[int, int]]:
    cache: Dict[int, Tuple[int, int]] = {}
    if not os.path.isfile(path):
        return cache
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cache[int(d["D"])] = (int(d["x1"]), int(d["y1"]))
    return cache


def append_eps_cache(path: str, entries: Sequence[Tuple[int, int, int]],
                     cache: Dict[int, Tuple[int, int]]) -> int:
    new = [(D, x1, y1) for (D, x1, y1) in entries if D not in cache]
    if not new:
        return 0
    with open(path, "a", encoding="utf-8") as f:
        for D, x1, y1 in new:
            cache[D] = (x1, y1)
            f.write(json.dumps({"D": str(D), "x1": str(x1), "y1": str(y1)}) + "\n")
    return len(new)


# ---------------------------------------------------------------------------
# Level driver
# ---------------------------------------------------------------------------

def run_level(level: int, iv: Intervals, res: Results, args, primes_all: List[int],
              eps_cache: Dict[int, Tuple[int, int]], level_progress: Dict[int, List[int]],
              paths: Dict[str, str], deadline: Optional[float]) -> bool:
    """Process all D with top(D) = level.  Returns True if the level completed."""
    n_lower = level - 1
    want_tasks = max(1, args.workers * args.tasks_per_worker)
    H = min(n_lower, max(0, math.ceil(math.log2(want_tasks)))) if n_lower > 0 else 0
    n_tasks = 1 << H
    done_masks = set(level_progress.get(level, []))
    todo = [m for m in range(n_tasks) if m not in done_masks]

    p_S = primes_all[res.maxlevel - 1]
    kmax = max(30, (p_S + 1) // 2)
    P_list = iv.P[1: iv.N + 2]

    t0 = time.time()
    print(f"[level {level:2d}] p={primes_all[level-1]}  D-count=2^{n_lower}  tasks={n_tasks} "
          f"(H={H}, {len(todo)} to do)  kmax={kmax}  kernel={args.kernel}", flush=True)

    totals = {"n_d": 0, "pref": 0, "n_cut": 0, "n_gate": 0, "n_solved": 0,
              "cf_steps": 0, "n_hits": 0, "n_err": 0}
    if not todo:
        completed = True
    else:
        ctx = get_context(args.mp_start_method) if args.mp_start_method != "auto" else get_context()
        pool = ctx.Pool(processes=args.workers, initializer=_worker_init,
                        initargs=(primes_all, iv.cap, level, H, kmax, P_list, iv.N,
                                  eps_cache, args.kernel),
                        maxtasksperchild=(args.maxtasks_per_child or None))
        completed = True
        pending_hits: List[Tuple] = []
        pending_eps: List[Tuple[int, int, int]] = []
        last_ckpt = time.time()
        n_done_now = 0
        try:
            for (mask, agg) in pool.imap_unordered(_task, todo, chunksize=1):
                n_done_now += 1
                for key in ("n_d", "pref", "n_cut", "n_gate", "n_solved", "cf_steps", "n_err"):
                    totals[key] += agg[key]
                totals["n_hits"] += len(agg["hits"])
                for h in agg["hits"]:
                    if res.add_hit(h):          # new pair (dedup across runs)
                        pending_hits.append(h)
                pending_eps.extend(agg["eps_found"])
                done_masks.add(mask)
                if agg["n_err"]:
                    print(f"  !! errors in task {mask}: {agg['err_samples']}", flush=True)

                now = time.time()
                if now - last_ckpt > args.checkpoint_seconds or n_done_now == len(todo):
                    append_hits(paths["hits"], pending_hits, level_done_at=level)
                    append_eps_cache(paths["eps"], pending_eps, eps_cache)
                    pending_hits, pending_eps = [], []
                    level_progress[level] = sorted(done_masks)
                    save_state(paths["state"], args, iv, res, level_progress,
                               {"in_progress_level": level})
                    last_ckpt = now
                    frac = n_done_now / len(todo)
                    el = now - t0
                    print(f"  [level {level}] {n_done_now}/{len(todo)} tasks, "
                          f"{el/60:.1f} min elapsed, ETA {el/frac*(1-frac)/60:.1f} min, "
                          f"hits so far {totals['n_hits']}", flush=True)
                if deadline is not None and now > deadline:
                    print(f"  [level {level}] time budget reached; stopping after current tasks",
                          flush=True)
                    completed = False
                    pool.terminate()
                    break
        finally:
            pool.close()
            pool.join()
        append_hits(paths["hits"], pending_hits, level_done_at=level)
        append_eps_cache(paths["eps"], pending_eps, eps_cache)
        level_progress[level] = sorted(done_masks)

    if completed and len(done_masks) == n_tasks:
        expected = 1 << n_lower
        if totals["n_d"] + totals["pref"] != expected and todo == list(range(n_tasks)):
            raise RuntimeError(f"level {level}: accounting failed: "
                               f"{totals['n_d']} + {totals['pref']} != {expected}")
        res.levels_done = sorted(set(res.levels_done) | {level})
        elapsed = time.time() - t0
        res.level_stats[level] = {**totals, "seconds": round(elapsed, 1),
                                  "completed_utc": utc_now()}
        per_d = (elapsed / totals["n_d"]) if totals["n_d"] else 0.0
        print(f"[level {level:2d}] done in {elapsed/60:.2f} min: D={totals['n_d']} "
              f"pref={totals['pref']} cut={totals['n_cut']} solved={totals['n_solved']} "
              f"y1-gate={totals['n_gate']} hits={totals['n_hits']} "
              f"cf-steps/D={totals['cf_steps']/max(1,totals['n_d']):.1f} "
              f"({per_d*1e3:.3f} ms/D); next level ~{2*elapsed/60:.1f} min", flush=True)
        level_progress.pop(level, None)
    save_state(paths["state"], args, iv, res, level_progress, {"in_progress_level": None})
    return completed and len(done_masks) == n_tasks


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Windowed, level-incremental exact solver for iota_min(n) "
                    "over primorial intervals n <= N.")
    ap.add_argument("--end-r", type=int, default=40,
                    help="largest interval N; sets CAP = 2*U_N+1. Choose generously.")
    ap.add_argument("--max-level", type=int, default=0,
                    help="largest D-level to process (0 => N+30)")
    ap.add_argument("--start-level", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count()))
    ap.add_argument("--tasks-per-worker", type=int, default=32)
    ap.add_argument("--kernel", choices=["auto", "python", "pari"], default="auto")
    ap.add_argument("--mp-start-method", default="auto",
                    help="auto | fork | spawn | forkserver")
    ap.add_argument("--maxtasks-per-child", type=int, default=0)
    ap.add_argument("--checkpoint-seconds", type=float, default=300.0)
    ap.add_argument("--time-budget-hours", type=float, default=0.0,
                    help="stop (with checkpoint) after this many hours; 0 = unlimited")
    ap.add_argument("--no-stop-when-resolved", action="store_true",
                    help="keep processing levels even after every interval is EXACT")
    ap.add_argument("--outdir", default="Primorial_iota_v2_runs")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the state in --outdir/--tag")
    ap.add_argument("--selftest", action="store_true",
                    help="run N=18 and compare with the known values")
    args = ap.parse_args()

    if args.selftest:
        args.end_r = 18
        args.tag = "selftest"
        args.max_level = 24

    if args.kernel == "auto":
        args.kernel = "pari" if HAS_CYPARI2 else "python"

    N = args.end_r
    maxlevel = args.max_level if args.max_level > 0 else N + 30
    iv = Intervals(N)
    primes_all = first_n_primes(maxlevel)
    res = Results(iv, maxlevel)

    base = os.path.join(args.outdir, args.tag)
    os.makedirs(base, exist_ok=True)
    paths = {"state": os.path.join(base, "state.json"),
             "hits": os.path.join(base, "hits.csv"),
             "eps": os.path.join(base, "eps_cache.jsonl"),
             "table": os.path.join(base, "iota_min_table.txt"),
             "csv": os.path.join(base, "iota_min.csv")}

    eps_cache = load_eps_cache(paths["eps"])
    level_progress: Dict[int, List[int]] = {}
    stale_levels: List[int] = []

    if args.resume and os.path.isfile(paths["state"]):
        with open(paths["state"], encoding="utf-8") as f:
            st = json.load(f)
        old_cap = int(st["cap"])
        old_levels = list(st.get("levels_done", []))
        old_progress = {int(k): v for k, v in st.get("level_progress", {}).items()}
        if old_cap == iv.cap:
            res.levels_done = sorted(old_levels)
            level_progress = old_progress
            res.level_stats = {int(k): v for k, v in st.get("level_stats", {}).items()}
        elif old_cap < iv.cap:
            # Extension: previously completed levels must be re-run against the
            # larger CAP (cached eps_D are reused; uncached D recompute).
            stale_levels = sorted(old_levels)
            print(f"[resume] CAP raised (old N gave log CAP {st.get('log_cap'):.2f}, "
                  f"new {log_bigint(iv.cap):.2f}); levels {stale_levels} will be "
                  f"re-run in extension mode using {len(eps_cache)} cached eps_D.")
        else:
            print("[resume] new CAP is smaller than the saved one; starting a fresh "
                  "state for this N (hits below the new CAP are still valid).")
        n = load_hits(paths["hits"], res)
        print(f"[resume] loaded {n} prior hits, {len(eps_cache)} cached fundamental "
              f"solutions, levels done {res.levels_done}")
    elif args.resume:
        print("[resume] no state found; starting fresh")

    print(f"{PROGRAM_NAME} v{PROGRAM_VERSION}: N={N}, log CAP={log_bigint(iv.cap):.2f} "
          f"(~{log_bigint(iv.cap)/math.log((1+5**0.5)/2):.0f} CF steps/D max), "
          f"max D-level {maxlevel} (p={primes_all[-1]}), workers={args.workers}, "
          f"kernel={args.kernel}, gmpy2={HAS_GMPY2}, cypari2={HAS_CYPARI2}")

    deadline = time.time() + 3600 * args.time_budget_hours if args.time_budget_hours > 0 else None

    # Extension mode first: re-run stale levels (they are cheap where cached).
    for lv in stale_levels:
        if lv in res.levels_done:
            continue
        ok = run_level(lv, iv, res, args, primes_all, eps_cache, level_progress, paths, deadline)
        print(write_table(res, paths["table"]), flush=True)
        if not ok:
            print("stopped (time budget) during extension; rerun with --resume")
            return 0

    s = max(args.start_level, res.s_done + 1)
    while s <= maxlevel:
        if not args.no_stop_when_resolved and res.all_exact():
            print("All intervals EXACT; stopping.")
            break
        ok = run_level(s, iv, res, args, primes_all, eps_cache, level_progress, paths, deadline)
        print(write_table(res, paths["table"]), flush=True)
        write_csv(paths["csv"], res)
        if not ok:
            print(f"stopped (time budget) inside level {s}; rerun with --resume to continue")
            break
        s += 1

    write_csv(paths["csv"], res)
    print(write_table(res, paths["table"]))

    if args.selftest:
        got = [res.status(n)[2] if res.status(n)[0] == "EXACT" else None
               for n in range(1, N + 1)]
        want = KNOWN_IOTA_MIN[:N]
        ok = got == want
        print(f"selftest: {'PASS' if ok else 'FAIL'}\n  got  {got}\n  want {want}")
        return 0 if ok else 1
    return 0


def write_csv(path: str, res: Results) -> None:
    iv = res.iv
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "p_n", "P_n", "L_n", "U_n", "status", "iota_min", "lower",
                    "upper", "pc_excluded", "witness_m", "witness_level", "witness_D",
                    "witness_k", "witness_omega", "pairs_found", "pc_hits", "levels_done_through"])
        for n in range(1, iv.N + 1):
            st, lo, up = res.status(n)
            b = res.best[n]
            w.writerow([n, iv.primes[n-1], iv.P[n], iv.L[n], iv.U[n], st,
                        up if st == "EXACT" else "", lo, up if up is not None else "",
                        "NO" if lo > n else "?",
                        b[1] if b else "", b[0] if b else "", b[2] if b else "",
                        b[3] if b else "", b[4] if b else "",
                        res.n_hits[n], " ".join(str(m) for m in res.pc_hits[n]),
                        res.s_done])
    os.replace(tmp, path)


if __name__ == "__main__":
    raise SystemExit(main())
