"""
make_figures.py -- produces every figure (figures/*.pdf) and every auto-generated
number/table used by the report (generated/numbers.tex, generated/*.tex).
Run from the project root:  python3 code/make_figures.py
"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensembles import *  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures"); GEN = os.path.join(ROOT, "generated")
os.makedirs(FIG, exist_ok=True); os.makedirs(GEN, exist_ok=True)

# ------------------------------------------------------------ constants (SI)
kB = 1.380649e-23; hP = 6.62607015e-34; amu = 1.66053906660e-27
NA_YEAR = 3.15576e7                              # s in a Julian year
# --- illustrative model parameters (NOT fitted to a specific species) -------
g, r = 3, 20                                      # vibrational modes, eps/(hbar omega)
HW_K = 60.0                                       # hbar*omega / k_B  [K]
EPS_K = r * HW_K                                  # eps / k_B = 1200 K
MASS = 28.0 * amu                                 # molecular mass used for the gas
NU0 = 1.0e12                                      # attempt frequency [1/s] (order of magnitude)

# ------------------------------------------------------------ plotting style
BLUE, VERM, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#6b6b6b"
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
    "axes.labelsize": 9.5, "axes.titlesize": 9.5, "legend.fontsize": 8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
})
SCH = [("backward", "backward", VERM, "v", "--"),
       ("central", "central", BLUE, "o", "-"),
       ("forward", "forward", GREEN, "^", ":")]


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


def panel(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight="bold",
            ha="right", va="bottom")


NUM = {}        # macro name -> string


# =========================================================== Figure 1: model
fig = plt.figure(figsize=(6.6, 3.1))
ax = fig.add_axes([0.0, 0.0, 0.58, 1.0]); ax.set_xlim(0, 10); ax.set_ylim(0, 7.2); ax.axis("off")
cx, cy, R = 2.9, 3.5, 1.35
ax.add_patch(Circle((cx, cy), R, fc="#c9c9c9", ec="#555555", lw=1.0))
ax.text(cx, cy, "dust grain\n(substrate)", ha="center", va="center", fontsize=8)
rng = np.random.RandomState(3)
nsite = 18; occ = rng.rand(nsite) < 0.5
for k in range(nsite):
    a = 2 * np.pi * k / nsite
    px, py = cx + (R + 0.17) * np.cos(a), cy + (R + 0.17) * np.sin(a)
    ax.add_patch(Circle((px, py), 0.13, fc=BLUE if occ[k] else "white", ec=BLUE, lw=1.0, zorder=3))
    if occ[k]:
        ax.add_patch(Circle((cx + (R + 0.52) * np.cos(a), cy + (R + 0.52) * np.sin(a)), 0.1,
                            fc=VERM, ec="none", zorder=3))
ax.add_patch(Circle((cx, cy), 2.3, fc="none", ec="#222222", lw=1.0, ls=(0, (4, 3))))
ax.text(cx, 0.35, "system: $M$ sites + $N$ adsorbed molecules", ha="center", fontsize=8)
ax.add_patch(FancyBboxPatch((7.6, 1.2), 2.3, 4.6, boxstyle="round,pad=0.05",
                            fc="#eef3f8", ec="#555555", lw=0.9))
ax.text(8.75, 5.15, "gas phase", ha="center", fontsize=8.5)
ax.text(8.75, 4.55, "(reservoir)", ha="center", fontsize=8)
ax.text(8.75, 3.35, "$T,\\ \\mu$", ha="center", fontsize=10)
for (x, y) in [(8.0, 2.2), (9.1, 2.55), (8.5, 1.7), (9.4, 1.6)]:
    ax.add_patch(Circle((x, y), 0.1, fc=VERM, ec="none"))
ax.add_patch(FancyArrowPatch((5.35, 3.5), (7.5, 3.5), arrowstyle="<->", mutation_scale=10,
                             lw=1.1, color="#222222"))
ax.text(6.42, 3.82, "exchange", ha="center", fontsize=7.5)
ax.text(6.42, 3.05, "(GC only)", ha="center", fontsize=7.5)
ax.add_patch(Circle((0.55, 6.75), 0.13, fc=BLUE, ec=BLUE)); ax.text(0.8, 6.68, "occupied site", fontsize=7.5, va="center")
ax.add_patch(Circle((3.0, 6.75), 0.13, fc="white", ec=BLUE)); ax.text(3.25, 6.68, "empty site", fontsize=7.5, va="center")
ax.add_patch(Circle((5.2, 6.75), 0.1, fc=VERM, ec="none")); ax.text(5.4, 6.68, "molecule", fontsize=7.5, va="center")
ax.text(0.0, 7.2, "(a)", fontsize=10, fontweight="bold", va="top")

ax2 = fig.add_axes([0.66, 0.14, 0.32, 0.80]); ax2.set_xlim(0, 6); ax2.set_ylim(-23.5, 3.5)
ax2.axis("off"); ax2.text(-0.05, 1.0, "(b)", transform=ax2.transAxes, fontsize=10, fontweight="bold", va="bottom")
ax2.plot([0.3, 3.6], [0, 0], color="#222222", lw=1.3)
ax2.text(3.75, 0, "gas, at rest: $E=0$", va="center", fontsize=7.5)
ax2.plot([0.3, 3.6], [0.0, 0.0], color="#222222")
for q in range(5):
    ax2.plot([0.3, 2.0], [-r + q, -r + q], color=BLUE, lw=1.1)
    ax2.text(2.1, -r + q, f"$q={q}$", va="center", fontsize=6.8, color="#222222")
ax2.annotate("", xy=(0.9, -r), xytext=(0.9, 0), arrowprops=dict(arrowstyle="<->", lw=1.0, color=VERM))
ax2.text(1.05, -9.5, "$\\varepsilon$", color=VERM, fontsize=11, va="center")
ax2.text(3.0, -r + 2.0, "spacing $\\hbar\\omega$", fontsize=7.3, va="center")
ax2.plot([0.3, 3.6], [-r - 1.6, -r - 1.6], color="white")
ax2.text(0.3, -22.4, "bound level $E=-\\varepsilon+q\\hbar\\omega$", fontsize=7.3)
save(fig, "fig_model")

# ===================================== Figure 2: microcanonical (M = 1000)
M2 = 1000
fig, axs = plt.subplots(1, 3, figsize=(6.7, 2.45), gridspec_kw=dict(wspace=0.55))
ax = axs[0]
th = np.linspace(0.02, 0.98, 49); N = np.round(th * M2).astype(int)
for qv, col, ls, lab in [(0.3, BLUE, "-", "$Q/N=0.3$"), (1.0, VERM, "--", "$Q/N=1$")]:
    Q = np.round(qv * N).astype(int)
    ex = ln_omega(M2, N, Q, g) / M2
    st = stirling_entropy(M2, N.astype(float), Q.astype(float), g) / M2
    ax.plot(N / M2, ex, color=col, ls=ls, label=lab)
    ax.plot(N[::6] / M2, st[::6], marker="o", ls="none", color=col, mfc="white", ms=3.8)
ax.set_xlabel("coverage $\\theta=N/M$"); ax.set_ylabel("$S/(Mk_B)$")
ax.legend(frameon=False, loc="lower center", handlelength=1.6); panel(ax, "(a)")
ax = axs[1]
Nn = 300; Qs = np.arange(5, 901, 5)
Tex = []; Tst = []
for q in Qs:
    b, _ = micro_exact(M2, Nn, int(q), g, r, "central"); Tex.append(HW_K / b)
    Tst.append(HW_K / micro_stirling(M2, Nn, q, g, r)[0])
ax.plot(Qs / Nn, Tex, color=BLUE, label="exact")
ax.plot(Qs[::6] / Nn, np.array(Tst)[::6], "o", ls="none", color=VERM, mfc="white", ms=3.8, label="Stirling")
ax.set_xlabel("quanta per molecule $Q/N$"); ax.set_ylabel("$T$ (K)")
ax.legend(frameon=False, loc="lower right"); panel(ax, "(b)")
ax = axs[2]
thv = np.arange(0.10, 0.96, 0.05); mex = []; mst = []
for t in thv:
    Nv = int(round(t * M2)); Qv = int(round(0.3 * Nv))
    b, bm = micro_exact(M2, Nv, Qv, g, r, "central"); mex.append(HW_K * bm / b)
    b2, bm2 = micro_stirling(M2, Nv, Qv, g, r); mst.append(HW_K * bm2 / b2)
ax.plot(thv, mex, color=BLUE, label="exact")
ax.plot(thv, mst, "o", color=VERM, mfc="white", ms=3.8, label="Stirling")
ax.axhline(-EPS_K, color=GREY, lw=0.8, ls=":"); ax.text(0.97, -EPS_K - 20, "$\\mu=-\\varepsilon$", ha="right", fontsize=7.5, color=GREY)
ax.set_xlabel("coverage $\\theta$"); ax.set_ylabel("$\\mu/k_B$ (K)")
ax.legend(frameon=False, loc="lower right"); panel(ax, "(c)")
save(fig, "fig_micro")

# ======================================= Figure 3: canonical check (M=1000)
M3, N3 = 1000, 300
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.55), gridspec_kw=dict(wspace=0.38))
Qs = np.arange(20, 1201, 10)
ax = axs[0]
Ec = []; E0 = []
for q in Qs:
    b, _ = micro_exact(M3, N3, int(q), g, r, "central")
    Ec.append(HW_K * canonical(M3, N3, b, g, r)["E"] / M3); E0.append(HW_K * (q - r * N3) / M3)
ax.plot([min(E0), max(E0)], [min(E0), max(E0)], color="#bbbbbb", lw=5, label="$y=x$")
ax.plot(E0, Ec, color=BLUE, label="canonical $\\langle E\\rangle$ at $T_{\\rm micro}$")
ax.set_xlabel("microcanonical $E/(Mk_B)$ (K)"); ax.set_ylabel("canonical $\\langle E\\rangle/(Mk_B)$ (K)")
ax.legend(frameon=False, loc="upper left"); panel(ax, "(a)")
ax = axs[1]
for (sch, lab, col, mk, ls) in SCH:
    d = []
    for q in Qs:
        b, _ = micro_exact(M3, N3, int(q), g, r, sch)
        d.append(canonical(M3, N3, b, g, r)["Q"] / q - 1)
    ax.semilogy(Qs / N3, d, color=col, ls=ls, label=lab)
G3 = g * N3
ax.semilogy(Qs / N3, np.full_like(Qs, 1.0 / (G3 - 1), dtype=float), color="black", lw=0.8, ls="-.")
ax.set_ylim(6e-4, 1e-1)
ax.text(0.3, 1.0 / (G3 - 1) * 0.8, "$1/(gN-1)$", ha="left", va="top", fontsize=7.5)
ax.set_xlabel("$Q/N$"); ax.set_ylabel("$(\\langle Q\\rangle_{\\rm can}-Q)/Q$")
ax.legend(frameon=False, loc="upper right", title="matching of $T$", title_fontsize=7.5)
panel(ax, "(b)")
save(fig, "fig_canonical")

# ================================ Figure 4: grand canonical check (M=10^4)
M4 = 10 ** 4
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.55), gridspec_kw=dict(wspace=0.38))
ax = axs[0]
Nv = np.arange(500, 9501, 250); Ng = []
for n_ in Nv:
    b, bm = micro_exact(M4, int(n_), int(round(0.3 * n_)), g, r, "central")
    Ng.append(grand(M4, b, bm, g, r)["N"])
ax.plot(Nv / M4, np.array(Ng) / M4, color=BLUE, label="grand-canonical $\\langle N\\rangle$")
ax.plot([0, 1], [0, 1], color=GREY, ls=":", lw=1.0, label="$y=x$")
ax.set_xlabel("microcanonical $N/M$"); ax.set_ylabel("$\\langle N\\rangle_{\\rm GC}/M$")
ax.legend(frameon=False, loc="upper left"); panel(ax, "(a)")
ax = axs[1]
th = np.arange(0.05, 0.951, 0.025)
for (sch, lab, col, mk, ls) in SCH:
    d = []
    for t in th:
        n_ = int(round(t * M4)); q = int(round(0.3 * n_))
        b, bm = micro_exact(M4, n_, q, g, r, sch)
        d.append(grand(M4, b, bm, g, r)["N"] / n_ - 1)
    ax.plot(th, d, color=col, ls=ls, label=lab)
ax.axhline(0, color=GREY, lw=0.8)
ax.set_xlabel("coverage $\\theta=N/M$"); ax.set_ylabel("$(\\langle N\\rangle_{\\rm GC}-N)/N$")
ax.legend(frameon=False, loc="upper right", title="matching of $T,\\mu$", title_fontsize=7.5)
panel(ax, "(b)")
save(fig, "fig_grand")

# ==================================== Figure 5: finite-size scaling (θ=0.3)
js = [2 ** k for k in range(2, 24)]
Mv = np.array([100 * j for j in js], dtype=float)
dQ = {s[0]: [] for s in SCH}; dN = {s[0]: [] for s in SCH}
for j in js:
    Mm, Nn, Qn = 100 * j, 30 * j, 9 * j
    for (sch, *_r) in SCH:
        b, bm = micro_exact(Mm, Nn, Qn, g, r, sch)
        dQ[sch].append(abs(canonical(Mm, Nn, b, g, r)["Q"] / Qn - 1))
        dN[sch].append(abs(grand(Mm, b, bm, g, r)["N"] / Nn - 1))
fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.5), gridspec_kw=dict(wspace=0.55))
for ax, dd, ylab, s in [(axs[0], dQ, "$|\\langle Q\\rangle_{\\rm can}/Q-1|$", "(a)"),
                        (axs[1], dN, "$|\\langle N\\rangle_{\\rm GC}/N-1|$", "(b)")]:
    for (sch, lab, col, mk, ls) in SCH:
        ax.loglog(Mv, dd[sch], color=col, ls=ls, label=lab)
    ax.loglog(Mv, 1.5e1 / Mv * (1 if ax is axs[1] else 0.5e-1), color="black", lw=0.8, ls="-.")
    ax.set_xlabel("number of sites $M$"); ax.set_ylabel(ylab); panel(ax, s)
axs[0].legend(frameon=False, loc="lower left", fontsize=7.3)
ax = axs[2]
Ms2 = np.array([100 * j for j in [2 ** k for k in range(2, 17)]], dtype=float)
dc = []; dg = []; pc = []; pg = []
for Mm in Ms2:
    Mm = int(Mm); Nn = int(0.3 * Mm); Qn = int(0.09 * Mm)
    b, bm = micro_stirling(Mm, Nn, Qn, g, r)
    lnO = ln_omega(Mm, Nn, Qn, g)
    c = canonical(Mm, Nn, b, g, r); gr = grand(Mm, b, bm, g, r)
    dc.append(c["S"] - lnO); dg.append(gr["S"] - lnO)
ax.semilogx(Ms2, dc, color=BLUE, label="canonical")
ax.semilogx(Ms2, dg, color=VERM, ls="--", label="grand-canonical")
ax.set_xlabel("number of sites $M$"); ax.set_ylabel("$(S_{\\rm ens}-k_B\\ln\\Omega)/k_B$")
ax.legend(frameon=False, loc="center right"); panel(ax, "(c)")
save(fig, "fig_scaling")
NUM["entCanSmall"] = f"{dc[0]:.2f}"; NUM["entCanLarge"] = f"{dc[-1]:.2f}"
NUM["entGCLarge"] = f"{dg[-1]:.2f}"

# ===================================================== Figure 6: fluctuations
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.55), gridspec_kw=dict(wspace=0.38))
ax = axs[0]
lam_ = 0.3 / 0.7; z = np.linspace(-4, 4, 400)
ax.plot(z, np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi), color=GREY, lw=0.9, ls=":", label="Gaussian")
for (Mm, col, mk) in [(30, VERM, "^"), (300, GREEN, "s"), (3000, BLUE, "o")]:
    n = np.arange(Mm + 1); lnP = lnC(Mm, n) + n * np.log(lam_) - Mm * np.log1p(lam_)
    P = np.exp(lnP); mu_ = Mm * 0.3; sg = math.sqrt(Mm * 0.21)
    ax.plot((n - mu_) / sg, P * sg, color=col, marker=mk, ms=3.0, lw=0.8, label=f"$M={Mm}$")
ax.set_xlim(-4, 4); ax.set_xlabel("$(N-\\langle N\\rangle)/\\sigma_N$"); ax.set_ylabel("$\\sigma_N\\,P(N)$")
ax.legend(frameon=False, loc="upper left", fontsize=7.3); panel(ax, "(a)")
ax = axs[1]
Mf = np.logspace(1, 8, 30)
th0 = 0.3; x0 = math.log1p(g / 0.3); gG = grand(1.0, x0, math.log(th0 / (1 - th0)) - x0 * r - ln_zv(x0, g), g, r)
sigN = np.sqrt(Mf * th0 * (1 - th0)) / (Mf * th0)
nbar = g / math.expm1(x0); sh2 = g * math.exp(x0) / math.expm1(x0) ** 2
sigQ = np.sqrt(th0 * Mf * sh2) / (th0 * Mf * nbar)
ax.loglog(Mf, sigN, color=BLUE, label="$\\sigma_N/\\langle N\\rangle$ (GC)")
ax.loglog(Mf, sigQ, color=VERM, ls="--", label="$\\sigma_Q/\\langle Q\\rangle$ (can.)")
ax.loglog(Mf, 1 / np.sqrt(Mf), color="black", lw=0.8, ls="-.")
ax.set_xlabel("number of sites $M$"); ax.set_ylabel("relative fluctuation")
ax.legend(frameon=False, loc="lower left"); panel(ax, "(b)")
save(fig, "fig_fluct")

# ======================== Figure 7: astrophysical context (equilibrium only)
def ln_lambda_site(Tk, n_cm3):
    n = n_cm3 * 1e6
    lamT = hP / math.sqrt(2 * math.pi * MASS * kB * Tk)
    x = HW_K / Tk
    return math.log(n * lamT ** 3) + EPS_K / Tk + ln_zv(x, g)

Tg = np.linspace(8, 60, 400)
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw=dict(wspace=0.42))
ax = axs[0]
for (n_, col, ls) in [(1e2, GREEN, ":"), (1e4, BLUE, "-"), (1e6, VERM, "--")]:
    th_ = [1 / (1 + math.exp(-ln_lambda_site(t, n_))) for t in Tg]
    ax.plot(Tg, th_, color=col, ls=ls, label=f"$n=10^{{{int(round(math.log10(n_)))}}}$ cm$^{{-3}}$")
ax.set_xlabel("temperature $T$ (K)"); ax.set_ylabel("equilibrium coverage $\\theta$")
ax.legend(frameon=False, loc="upper right"); panel(ax, "(a)")
ax = axs[1]
tdes = NU0 ** -1 * np.exp(EPS_K / Tg)
ax.semilogy(Tg, tdes / NA_YEAR, color=BLUE)
for yr, lab in [(1.0, "1 yr"), (1e10, "$10^{10}$ yr")]:
    ax.axhline(yr, color=GREY, lw=0.8, ls=":"); ax.text(59, yr * 1.5, lab, ha="right", fontsize=7.5, color="#444444")
ax.set_xlabel("temperature $T$ (K)"); ax.set_ylabel("desorption time $t_{\\rm des}$ (yr)")
ax.set_ylim(1e-12, 1e40); panel(ax, "(b)")
save(fig, "fig_astro")
T_yr = EPS_K / math.log(1.0 * NA_YEAR * NU0)
T_gyr = EPS_K / math.log(1e10 * NA_YEAR * NU0)
NUM["TdesOneYr"] = f"{T_yr:.1f}"; NUM["TdesTenGyr"] = f"{T_gyr:.1f}"


def wrap_tab(spec, header, rows):
    return ("\\begin{tabular}{" + spec + "}\n\\toprule\n" + header + "\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ============================================================ numbers/tables
# representative state:  theta=0.3, Q/N=0.3
def rep(Mm, sch):
    Nn = int(round(0.3 * Mm)); Qn = int(round(0.09 * Mm))
    b, bm = micro_exact(Mm, Nn, Qn, g, r, sch)
    c = canonical(Mm, Nn, b, g, r); gr = grand(Mm, b, bm, g, r)
    E_mc = HW_K * (Qn - r * Nn) / Mm
    lnO = float(ln_omega(Mm, Nn, Qn, g))
    return dict(M=Mm, N=Nn, Q=Qn, b=b, bm=bm, c=c, gr=gr, E_mc=E_mc, lnO=lnO)

# Stirling-level (large-system) parameters for the representative state
b_s, bm_s = micro_stirling(10 ** 6, 3 * 10 ** 5, 9 * 10 ** 4, g, r)
Tst = HW_K / b_s; mu_K = HW_K * bm_s / b_s
NUM["Tstir"] = f"{Tst:.2f}"; NUM["muKstir"] = f"{mu_K:.0f}"
NUM["bhwstir"] = f"{b_s:.4f}"; NUM["bmustir"] = f"{bm_s:.2f}"
NUM["epsK"] = f"{EPS_K:.0f}"; NUM["hwK"] = f"{HW_K:.0f}"
NUM["epsJ"] = f"{EPS_K * kB:.3e}".replace("e-", r"\times10^{-") + "}"
NUM["epsEV"] = f"{EPS_K * kB / 1.602176634e-19:.3f}"
NUM["hwJ"] = f"{HW_K * kB:.3e}".replace("e-", r"\times10^{-") + "}"
NUM["nuHz"] = f"{HW_K * kB / hP / 1e12:.2f}"
mu_J = mu_K * kB
NUM["muJ"] = f"{mu_J:.3e}".replace("e-", r"\times10^{-").replace("-1", "-1") + "}"
# gas density that realises this mu (structureless ideal gas, z_int = 1)
lamT = hP / math.sqrt(2 * math.pi * MASS * kB * Tst)
n_gas = math.exp(bm_s) / lamT ** 3
NUM["lamT"] = f"{lamT * 1e10:.2f}"
NUM["ngas"] = f"{n_gas / 1e6:.0f}"
NUM["tdesRep"] = f"{NU0 ** -1 * math.exp(EPS_K / Tst) / NA_YEAR:.0f}"
NUM["lamsite"] = f"{(0.3 / 0.7):.4f}"
NUM["zvRep"] = f"{math.exp(ln_zv(b_s, g)):.4f}"
NUM["nbarmode"] = f"{1 / math.expm1(b_s):.3f}"

R6 = rep(10 ** 6, "central")
def pct(a, b_):
    return (a / b_ - 1)
lines = []
def fx(v):
    if v == 0:
        return "$0$"
    sgn = "+" if v > 0 else "-"
    e = int(math.floor(math.log10(abs(v)))); m = abs(v) / 10 ** e
    if round(m, 1) >= 10:
        m, e = 1.0, e + 1
    return rf"${sgn}{m:.1f}\!\times\!10^{{{e}}}$"

def row(label, mc, can, gc, fmt="{:.6g}"):
    d_c = can / mc - 1 if mc != 0 else 0.0
    d_g = gc / mc - 1 if mc != 0 else 0.0
    return rf"{label} & {fmt.format(mc)} & {fmt.format(can)} & {fmt.format(gc)} & {fx(d_c)} & {fx(d_g)} \\"
c6, g6 = R6["c"], R6["gr"]
Mm = R6["M"]
tab = []
tab.append(row(r"$N$ (adsorbed molecules)", R6["N"], R6["N"], g6["N"], "{:,.1f}".replace(",", "")))
tab.append(row(r"$Q$ (vibrational quanta)", R6["Q"], c6["Q"], g6["Q"], "{:.1f}"))
tab.append(row(r"$E/(Mk_B)$ (K)", R6["E_mc"], HW_K * c6["E"] / Mm, HW_K * g6["E"] / Mm, "{:.5f}"))
tab.append(row(r"$S/(Mk_B)$", R6["lnO"] / Mm, c6["S"] / Mm, g6["S"] / Mm, "{:.6f}"))
tab = [t.replace("& -", "& $-$") for t in tab]
open(os.path.join(GEN, "table_compare_1e6.tex"), "w").write(wrap_tab('@{}l r r r r r@{}', 'Quantity & micro & canonical & grand-can. & rel.\\ diff.\\ (can.) & rel.\\ diff.\\ (GC) \\\\', tab))

# micro parameters table: T and mu for every matching convention
rowsP = []
for Mm_ in [10 ** 3, 10 ** 6]:
    Nn_ = int(round(0.3 * Mm_)); Qn_ = int(round(0.09 * Mm_))
    lab = rf"$10^{{{int(round(math.log10(Mm_)))}}}$"
    bS, bmS = micro_stirling(Mm_, Nn_, Qn_, g, r)
    entries = [("Stirling (large-$N$)", bS, bmS)]
    for sch in ["backward", "central", "forward"]:
        b_, bm_ = micro_exact(Mm_, Nn_, Qn_, g, r, sch); entries.append((sch, b_, bm_))
    for k_, (name, b_, bm_) in enumerate(entries):
        rowsP.append(rf"{lab if k_ == 0 else ''} & {name} & {HW_K / b_:.5f} & $-${abs(HW_K * bm_ / b_):.3f} \\")
    rowsP.append(r"\midrule") if Mm_ == 10 ** 3 else None
open(os.path.join(GEN, "table_micro_params.tex"), "w").write(wrap_tab('@{}c l r r@{}', '$M$ & definition & $T$ (K) & $\\mu/\\kB$ (K)\\\\', rowsP))


# finite-size table : deviations for several M (central) ; backward/forward bracket
rows2 = []
for Mm_ in [10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]:
    outs = []
    for sch in ["backward", "central", "forward"]:
        Rr = rep(Mm_, sch)
        outs.append((Rr["c"]["Q"] / Rr["Q"] - 1, Rr["gr"]["N"] / Rr["N"] - 1))
    def f(v):
        s = f"{v:+.1e}".replace("e-0", "e-").replace("e+0", "e+")
        mant, ex = s.split("e")
        return rf"${mant}\!\times\!10^{{{int(ex)}}}$"
    rows2.append(rf"$10^{{{int(round(math.log10(Mm_)))}}}$ & " + " & ".join(f(o[0]) for o in outs)
                 + " & " + " & ".join(f(o[1]) for o in outs) + r" \\")
open(os.path.join(GEN, "table_finite_size.tex"), "w").write(wrap_tab('@{}c ccc ccc@{}', ' & \\multicolumn{3}{c}{$(\\ave{Q}_{\\rm can}-Q)/Q$} & \\multicolumn{3}{c}{$(\\ave{N}_{\\rm GC}-N)/N$}\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n$M$ & bwd & central & fwd & bwd & central & fwd\\\\', rows2))

# other macros
Rk = rep(10 ** 3, "central")
NUM["devNkilo"] = f"{Rk['gr']['N'] / Rk['N'] - 1:+.3f}"
NUM["devQmega"] = f"{c6['Q'] / R6['Q'] - 1:.2e}"
NUM["devNmega"] = f"{g6['N'] / R6['N'] - 1:.2e}"
NUM["relQpred"] = f"{1 / (g * 300000 - 1):.2e}"
sl = json.load(open(os.path.join(GEN, "slopes.json")))
NUM["slopeQ"] = f"{sl['slope_Q']:.3f}"; NUM["slopeN"] = f"{sl['slope_N']:.3f}"
v = json.load(open(os.path.join(GEN, "validation.json")))
NUM["nchecks"] = str(len(v)); NUM["npass"] = str(sum(1 for c in v if c["ok"]))
# fluctuation numbers at M = 1e6
NUM["relfluctN"] = f"{math.sqrt(1e6 * 0.21) / (3e5):.2e}"
# determinant / covariance of the representative state
NUM["sighsq"] = f"{sh2:.4f}"
NUM["thetaRep"] = "0.30"
# scale for r^2 regime
NUM["rsq"] = str(r * r)

with open(os.path.join(GEN, "numbers.tex"), "w") as fh:
    for k, val in NUM.items():
        fh.write(f"\\newcommand{{\\num{k}}}{{{val}}}\n")
print(json.dumps(NUM, indent=1))
