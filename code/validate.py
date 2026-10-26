"""
validate.py -- independent verification of every closed-form result used in
the report.  Writes  generated/validation.json  and  generated/validation_table.tex
Run:  python3 code/validate.py        (from the project root)
"""
import itertools, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensembles import *  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "generated")
os.makedirs(GEN, exist_ok=True)

results = []   # (id, description, max deviation, tolerance, passed)


def record(cid, desc, dev, tol):
    ok = bool(dev <= tol)
    results.append(dict(id=cid, desc=desc, dev=float(dev), tol=float(tol), ok=ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {cid:4s} {desc:78s} dev={dev:.3e} (tol {tol:.0e})")


# ---------------------------------------------------------------------------
# V1  Model A: enumerate all 2^M occupation patterns
# ---------------------------------------------------------------------------
M = 14
occ = ((np.arange(2 ** M)[:, None] >> np.arange(M)) & 1).sum(axis=1)
cnt = np.bincount(occ, minlength=M + 1)
exact = np.array([math.comb(M, n) for n in range(M + 1)])
record("V1", "Model A: brute-force 2^M enumeration vs C(M,N)  (M=14, all N)",
       float(np.abs(cnt - exact).max()), 0.0)


# ---------------------------------------------------------------------------
# V2/V3  Model B: brute-force enumeration of ALL microstates (explicit
#        occupation + explicit quantum numbers of every vibrational mode)
# ---------------------------------------------------------------------------
def site_states(g, Qc):
    st = [(0, 0)]
    for qs in itertools.product(range(Qc + 1), repeat=g):
        if sum(qs) <= Qc:
            st.append((1, sum(qs)))
    return np.array(st, dtype=np.int16)


def enumerate_all(M, g, Qc):
    s = site_states(g, Qc)
    Nt, Qt = s[:, 0].copy(), s[:, 1].copy()
    for _ in range(M - 1):
        Nt = (Nt[:, None] + s[None, :, 0]).ravel()
        Qt = (Qt[:, None] + s[None, :, 1]).ravel()
    return Nt, Qt


configs = [(4, 1, 30, 20), (3, 2, 14, 20), (3, 3, 8, 20)]  # M, g, Qcut, r
for (Mb, gb, Qc, rb) in configs:
    Nt, Qt = enumerate_all(Mb, gb, Qc)
    # (V2) multiplicities for total quanta <= Qc  (complete set of states)
    keep = Qt <= Qc
    H = np.zeros((Mb + 1, Qc + 1), dtype=np.int64)
    np.add.at(H, (Nt[keep], Qt[keep]), 1)
    maxdev = 0
    for n in range(1, Mb + 1):
        for q in range(Qc + 1):
            f = math.comb(Mb, n) * math.comb(q + gb * n - 1, q)
            maxdev = max(maxdev, abs(int(H[n, q]) - f))
    record("V2", f"Model B: enumerated Omega(N,Q) vs C(M,N)C(Q+gN-1,Q)  (M={Mb}, g={gb}, Q<={Qc})",
           maxdev, 0.0)
    # (V3) canonical and grand-canonical sums over the enumerated states
    beta = 3.0
    E = -rb * Nt.astype(float) + Qt
    dev3 = 0.0
    for n in range(0, Mb + 1):
        Zbf = np.exp(-beta * E[Nt == n]).sum()
        Zcf = math.comb(Mb, n) * np.exp(beta * rb * n + n * ln_zv(beta, gb))
        dev3 = max(dev3, abs(Zbf / Zcf - 1))
    record("V3a", f"canonical Z_N: brute-force sum vs C(M,N)e^(b N eps) z_v^N  (M={Mb}, g={gb})",
           dev3, 1e-8)
    bmu = -45.0
    Xi_bf = np.exp(-beta * E + bmu * Nt).sum()
    lam = np.exp(bmu + beta * rb + ln_zv(beta, gb))
    Xi_cf = (1 + lam) ** Mb
    pbf = np.array([np.exp(-beta * E[Nt == n] + bmu * n).sum() for n in range(Mb + 1)]) / Xi_bf
    pcf = np.array([math.comb(Mb, n) * lam ** n for n in range(Mb + 1)]) / Xi_cf
    record("V3b", f"grand Xi=(1+lambda)^M and P(N): brute force vs closed form  (M={Mb}, g={gb})",
           max(abs(Xi_bf / Xi_cf - 1), np.abs(pbf - pcf).max()), 1e-8)

# ---------------------------------------------------------------------------
# V4  Stirling-level T and mu equal the derivatives of the Stirling entropy
# ---------------------------------------------------------------------------
g, r = 3, 20
Mx, Nx, Qx = 1.0e6, 3.0e5, 9.0e4
S_E = lambda N, E: stirling_entropy(Mx, N, E + r * N, g)       # S(E,N)
E0 = -r * Nx + Qx
h = 1e-2
dSdE = (S_E(Nx, E0 + h) - S_E(Nx, E0 - h)) / (2 * h)
dSdN = (S_E(Nx + h, E0) - S_E(Nx - h, E0)) / (2 * h)
b_st, bmu_st = micro_stirling(Mx, Nx, Qx, g, r)
record("V4a", "Stirling: 1/kT = dS/dE|_N  analytic vs numerical derivative",
       abs(dSdE / b_st - 1), 1e-7)
record("V4b", "Stirling: -mu/kT = dS/dN|_E  analytic vs numerical derivative",
       abs(-dSdN / bmu_st - 1), 1e-7)

# ---------------------------------------------------------------------------
# V5  exact algebraic identities of the finite-difference matching
# ---------------------------------------------------------------------------
Mx, Nx, Qx = 1000, 300, 90
G = g * Nx
bb, _ = micro_exact(Mx, Nx, Qx, g, r, "backward")
bf, _ = micro_exact(Mx, Nx, Qx, g, r, "forward")
cb = canonical(Mx, Nx, bb, g, r)["Q"]
cf = canonical(Mx, Nx, bf, g, r)["Q"]
record("V5a", "canonical <Q> at backward-difference T equals Q*G/(G-1), G=gN",
       abs(cb / (Qx * G / (G - 1)) - 1), 1e-12)
record("V5b", "canonical <Q> at forward-difference T equals G(Q+1)/(G-1)",
       abs(cf / (G * (Qx + 1) / (G - 1)) - 1), 1e-12)
dev = 0.0
for (Mm, Nn) in [(50, 7), (1000, 300), (10 ** 6, 3 * 10 ** 5)]:
    xb = Nn / (Mm - Nn + 1.0)       # backward  : x = N/(M-N+1)
    xf = (Nn + 1.0) / (Mm - Nn)     # forward   : x = (N+1)/(M-N)
    dev = max(dev, abs(grand_A(Mm, xb) / (Mm * Nn / (Mm + 1.0)) - 1),
              abs(grand_A(Mm, xf) / (Mm * (Nn + 1.0) / (Mm + 1.0)) - 1))
record("V5c", "Model A: <N>_GC = M N/(M+1) (backward) and M(N+1)/(M+1) (forward), exact",
       dev, 1e-12)
# exact bracket in Model B (numerical statement)
ok = True
for j in [4, 16, 64, 256, 1024]:
    Mm, Nn, Qn = 100 * j, 30 * j, 9 * j
    vals = []
    for sch in ["backward", "central", "forward"]:
        b, bm = micro_exact(Mm, Nn, Qn, g, r, sch)
        vals.append(grand(Mm, b, bm, g, r)["N"])
    ok &= (vals[0] < Nn < vals[2])
record("V5d", "Model B: backward/forward matching bracket N:  <N>_back < N < <N>_fwd  (M=400..1e5)",
       0.0 if ok else 1.0, 0.0)

# ---------------------------------------------------------------------------
# V6  Covariance (Hessian of ln Xi) -- determinant, Model B versus Model A
# ---------------------------------------------------------------------------
beta0, bmu0, Mh = 2.4, -49.0, 1000.0
def lnXi(b, a):
    return Mh * np.log1p(np.exp(a + b * r + ln_zv(b, g)))
hh = 1e-4
fEE = (lnXi(beta0 + hh, bmu0) - 2 * lnXi(beta0, bmu0) + lnXi(beta0 - hh, bmu0)) / hh ** 2
fNN = (lnXi(beta0, bmu0 + hh) - 2 * lnXi(beta0, bmu0) + lnXi(beta0, bmu0 - hh)) / hh ** 2
fEN = (lnXi(beta0 + hh, bmu0 + hh) - lnXi(beta0 + hh, bmu0 - hh)
       - lnXi(beta0 - hh, bmu0 + hh) + lnXi(beta0 - hh, bmu0 - hh)) / (4 * hh ** 2)
Gd = grand(Mh, beta0, bmu0, g, r)
uu = -r + g / math.expm1(beta0)
cov_closed = Mh * Gd["theta"] * (1 - Gd["theta"]) * uu          # Cov(E,N)
record("V6a", "Model B: Cov(E,N) numerical Hessian vs closed form M th(1-th) u",
       abs(-fEN / cov_closed - 1), 1e-6)
record("V6e", "Model B: det Cov = Var E Var N - Cov^2 equals M^2 th^2(1-th) sig_h^2 (algebra)",
       abs((Gd["varE"] * Gd["varN"] - cov_closed ** 2) / Gd["detcov"] - 1), 1e-9)
record("V6b", "Model B: Var(N) numerical vs M th (1-th)", abs(fNN / Gd["varN"] - 1), 1e-5)
record("V6c", "Model B: Var(E) numerical vs closed form", abs(fEE / Gd["varE"] - 1), 1e-5)
# Model A (energy -eps*N): det must vanish identically
thA, epsA, MA = 0.3, 1.0, 1000.0
varN_A = MA * thA * (1 - thA); varE_A = epsA ** 2 * varN_A; cov_A = -epsA * varN_A
record("V6d", "Model A: det Cov(E,N) = 0 (E and N perfectly locked)",
       abs(varE_A * varN_A - cov_A ** 2) / (varE_A * varN_A), 1e-14)

# ---------------------------------------------------------------------------
# V7  normalisation / moments of the exact GC and canonical distributions
# ---------------------------------------------------------------------------
Mx = 400
bt, bm = 2.3979, -49.09
lam = np.exp(bm + bt * r + ln_zv(bt, g))
n = np.arange(Mx + 1)
lnP = lnC(Mx, n) + n * np.log(lam) - Mx * np.log1p(lam)
P = np.exp(lnP)
mean = (n * P).sum(); var = (n ** 2 * P).sum() - mean ** 2
Gd = grand(Mx, bt, bm, g, r)
record("V7a", "GC: sum_N P(N) = 1, <N>, Var N from exact distribution vs formulas",
       max(abs(P.sum() - 1), abs(mean / Gd["N"] - 1), abs(var / Gd["varN"] - 1)), 1e-10)
# canonical vibrational energy distribution: Q-distribution P(Q) ~ C(Q+G-1,Q) e^{-beta Q}
Nn = 120; Gg = g * Nn
qq = np.arange(0, 4000)
lnPq = lnC(qq + Gg - 1, qq) - bt * qq
lnPq -= lnPq.max(); Pq = np.exp(lnPq); Pq /= Pq.sum()
mq = (qq * Pq).sum(); vq = (qq ** 2 * Pq).sum() - mq ** 2
cc = canonical(Mx, Nn, bt, g, r)
record("V7b", "canonical: <Q> and Var Q from exact P(Q) vs g N/(e^b-1), gN e^b/(e^b-1)^2",
       max(abs(mq / cc["Q"] - 1), abs(vq / (cc["varE"]) - 1)), 1e-9)

# ---------------------------------------------------------------------------
# V8  entropy: S_ensemble - k ln Omega  (Gaussian-fluctuation offsets)
# ---------------------------------------------------------------------------
Mx, Nx, Qx = 10 ** 6, 3 * 10 ** 5, 9 * 10 ** 4
b_st, bmu_st = micro_stirling(Mx, Nx, Qx, g, r)
lnOm = ln_omega_ratio(Mx, Nx, Qx, 1, 0, g) + ln_omega(Mx, 1, 0, g)   # ln Omega exactly
can = canonical(Mx, Nx, b_st, g, r)
gr = grand(Mx, b_st, bmu_st, g, r)
sig2_Q = g * Nx * math.exp(b_st) / math.expm1(b_st) ** 2
pred_can = 0.5 * math.log(2 * math.pi * sig2_Q)
pred_gc = math.log(2 * math.pi) + 0.5 * math.log(gr["detcov"])
record("V8a", "S_can - k ln(Omega) = (1/2) ln(2 pi sigma_Q^2) + O(1/N)   (M=1e6)",
       abs((can["S"] - lnOm) - pred_can), 1e-3)
record("V8b", "S_GC - k ln(Omega) = ln 2pi + (1/2) ln det Cov(N,Q) + O(1/N)   (M=1e6)",
       abs((gr["S"] - lnOm) - pred_gc), 1e-3)

# ---------------------------------------------------------------------------
# V9  finite-size convergence rate (log-log slope of deviations vs M)
# ---------------------------------------------------------------------------
Ms, dQ, dN = [], [], []
for j in [2 ** k for k in range(10, 22)]:
    Mm, Nn, Qn = 100 * j, 30 * j, 9 * j
    b, bm = micro_exact(Mm, Nn, Qn, g, r, "central")
    dQ.append(abs(canonical(Mm, Nn, b, g, r)["Q"] / Qn - 1))
    dN.append(abs(grand(Mm, b, bm, g, r)["N"] / Nn - 1))
    Ms.append(Mm)
sl_Q = np.polyfit(np.log(Ms[-6:]), np.log(dQ[-6:]), 1)[0]
sl_N = np.polyfit(np.log(Ms[-6:]), np.log(dN[-6:]), 1)[0]
record("V9a", "central matching: slope d ln|<Q>/Q-1| / d ln M -> -1 (largest six M)",
       abs(sl_Q + 1), 2e-2)
record("V9b", "central matching: slope d ln|<N>/N-1| / d ln M -> -1 (largest six M)",
       abs(sl_N + 1), 5e-2)

# ---------------------------------------------------------------------------
# write outputs
# ---------------------------------------------------------------------------
json.dump(results, open(os.path.join(GEN, "validation.json"), "w"), indent=1)

def sci(x):
    if x == 0:
        return "$0$"
    e = int(math.floor(math.log10(abs(x))))
    m = round(x / 10 ** e, 1)
    if m >= 10:
        m, e = 1.0, e + 1
    return rf"${m:.1f}\times10^{{{e}}}$"

import re
TEX = {
"V1": r"Model A: all $2^M$ occupation patterns counted vs.\ $\binom{M}{N}$",
"V2": r"Model B: enumerated $\Omega(N,Q)$ vs.\ $\binom{M}{N}\binom{Q+gN-1}{Q}$",
"V3a": r"Canonical $Z_N$: brute-force sum vs.\ $\binom{M}{N}e^{\beta N\varepsilon}z_v^{N}$",
"V3b": r"Grand $\Xi=(1+\lambda)^M$ and $P(N)$: brute-force sum vs.\ closed form",
"V4a": r"Stirling: $1/k_BT=\partial S/\partial E|_N$, analytic vs.\ numerical derivative",
"V4b": r"Stirling: $-\mu/k_BT=\partial S/\partial N|_E$, analytic vs.\ numerical derivative",
"V5a": r"Canonical $\langle Q\rangle$ at backward-difference $T$ equals $QG/(G-1)$, $G=gN$",
"V5b": r"Canonical $\langle Q\rangle$ at forward-difference $T$ equals $G(Q+1)/(G-1)$",
"V5c": r"Model A: $\langle N\rangle_{\rm GC}=MN/(M+1)$ (backward), $M(N+1)/(M+1)$ (forward)",
"V5d": r"Model B: $\langle N\rangle^{\rm back}_{\rm GC}<N<\langle N\rangle^{\rm fwd}_{\rm GC}$, $M=400$--$10^5$ (0 = holds)",
"V6a": r"Model B: $\mathrm{Cov}(E,N)$, numerical Hessian of $\ln\Xi$ vs.\ closed form",
"V6b": r"Model B: $\mathrm{Var}(N)$ numerical vs.\ $M\theta(1-\theta)$",
"V6c": r"Model B: $\mathrm{Var}(E)$ numerical vs.\ closed form",
"V6d": r"Model A: $\det\mathrm{Cov}(E,N)=0$ (relative to $\mathrm{Var}E\,\mathrm{Var}N$)",
"V6e": r"Model B: $\det\mathrm{Cov}=M^2\theta^2(1-\theta)\sigma_h^2$ (algebraic identity)",
"V7a": r"Grand: normalisation, $\langle N\rangle$, $\mathrm{Var}\,N$ from exact $P(N)$",
"V7b": r"Canonical: $\langle Q\rangle$, $\mathrm{Var}\,Q$ from exact $P(Q)$",
"V8a": r"$S_{\rm can}-k_B\ln\Omega=\tfrac12\ln(2\pi\sigma_Q^2)$ at $M=10^6$ (deviation of prediction)",
"V8b": r"$S_{\rm GC}-k_B\ln\Omega=\ln 2\pi+\tfrac12\ln\det\mathrm{Cov}$ at $M=10^6$ (deviation)",
"V9a": r"Central matching: slope of $\ln|\langle Q\rangle/Q-1|$ vs.\ $\ln M$ is $-1$ (deviation)",
"V9b": r"Central matching: slope of $\ln|\langle N\rangle/N-1|$ vs.\ $\ln M$ is $-1$ (deviation)",
}
rows = []
for c in results:
    d = TEX[c["id"]]
    m = re.search(r"\((M=\d+(?:, g=\d)?(?:, Q<=\d+)?)\)", c["desc"])
    if m:
        cfg = m.group(1).replace(", ", r",\ ").replace("Q<=", r"Q\leq ")
        d += rf" [${cfg}$]"
    rows.append(rf"{c['id']} & {d} & {sci(c['dev'])} & {'pass' if c['ok'] else 'FAIL'} \\")
open(os.path.join(GEN, "validation_rows.tex"), "w").write("\n".join(rows) + "\n")
json.dump(dict(slope_Q=float(sl_Q), slope_N=float(sl_N)), open(os.path.join(GEN, "slopes.json"), "w"))
print("\nALL PASS" if all(c["ok"] for c in results) else "\nSOME CHECKS FAILED")
sys.exit(0 if all(c["ok"] for c in results) else 1)
