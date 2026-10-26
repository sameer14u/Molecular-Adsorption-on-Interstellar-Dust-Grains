"""
ensembles.py -- exact and asymptotic results for the three ensembles of the
independent-site adsorption model with vibrational structure.

MODEL B (the main model of the report)
--------------------------------------
* M distinguishable, independent adsorption sites (fixed lattice).
* Each site is empty (energy 0) or holds one molecule (n_i = 1).
* An adsorbed molecule sits in its lowest bound state at energy -eps
  (zero reference: molecule at rest, far away, in the gas phase) and carries
  g harmonic vibrational modes of common quantum  hbar*omega.
  Level energy of an occupied site:  -eps + (q_1+...+q_g) hbar*omega.
* N = number of ADSORBED molecules, Q = total number of vibrational quanta,
  E = -N*eps + Q*hbar*omega.

UNITS: energies in units of hbar*omega, temperature in units of hbar*omega/k_B,
so beta = 1/(k_B T) is the dimensionless number  beta*hbar*omega, entropy in
units of k_B.  eps = r (integer, so that fixed-E particle exchange is exact).

MODEL A (bare binary lattice): see functions with suffix _A.
"""
import math
import numpy as np
from scipy.special import gammaln


# ----------------------------------------------------------------- helpers
def lnC(n, k):
    """ln of the binomial coefficient C(n,k) for real arguments (gammaln)."""
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)
    return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)


def ln_zv(beta, g):
    """ln of the vibrational partition function of ONE adsorbed molecule,
    z_v = (1 - exp(-beta))^(-g)   (levels q*hbar*omega, zero point in eps)."""
    return -g * np.log1p(-np.exp(-np.asarray(beta, dtype=float)))


# --------------------------------------------------------- microcanonical
def ln_omega(M, N, Q, g):
    """Exact ln Omega(M,N,Q) = ln C(M,N) + ln C(Q+gN-1, Q), N >= 1."""
    N = np.asarray(N, dtype=float)
    Q = np.asarray(Q, dtype=float)
    return lnC(M, N) + lnC(Q + g * N - 1.0, Q)


def stirling_entropy(M, N, Q, g):
    """Leading-order (Stirling) entropy S/k_B (continuous N, Q)."""
    G = g * N
    conf = M * np.log(M) - N * np.log(N) - (M - N) * np.log(M - N)
    vib = (Q + G) * np.log(Q + G) - Q * np.log(Q) - G * np.log(G)
    return conf + vib


def micro_stirling(M, N, Q, g, r):
    """Large-system microcanonical T and mu from the Stirling entropy.
    Returns beta (=beta*hbar*omega), beta*mu."""
    N = float(N)
    Q = float(Q)
    beta = np.log1p(g * N / Q)
    bmu = np.log(N / (M - N)) - beta * r - ln_zv(beta, g)
    return beta, bmu


def lnfr(a, b):
    """ln(a!/b!) for non-negative integers, as an exact sum of logarithms
    (no cancellation between huge Gamma functions)."""
    a = int(a); b = int(b)
    if a == b:
        return 0.0
    if a > b:
        return math.fsum(math.log(j) for j in range(b + 1, a + 1))
    return -math.fsum(math.log(j) for j in range(a + 1, b + 1))


def lnC_ratio(n1, k1, n2, k2):
    """ln[ C(n1,k1) / C(n2,k2) ] for integer arguments, evaluated stably."""
    return lnfr(n1, n2) - lnfr(k1, k2) - lnfr(n1 - k1, n2 - k2)


def ln_omega_ratio(M, N1, Q1, N2, Q2, g):
    """ln[ Omega(M,N1,Q1) / Omega(M,N2,Q2) ], exact and cancellation-free."""
    return (lnC_ratio(M, N1, M, N2)
            + lnC_ratio(Q1 + g * N1 - 1, Q1, Q2 + g * N2 - 1, Q2))


def micro_exact(M, N, Q, g, r, scheme="central"):
    """Temperature and chemical potential from the EXACT multiplicity using
    finite differences on the integer lattice of microstates:
        beta     = d lnOmega / dE    at fixed N   (E-step = one quantum)
        -beta*mu = d lnOmega / dN    at fixed E   (adding/removing a molecule
                                                  at fixed E changes Q by +-r)
    scheme: 'central', 'backward' (remove one), 'forward' (add one).
    Returns beta, beta*mu."""
    M, N, Q, g, r = int(M), int(N), int(Q), int(g), int(r)
    R = lambda n1, q1, n2, q2: ln_omega_ratio(M, n1, q1, n2, q2, g)
    nan = float("nan")          # mu undefined when removing a molecule needs Q < r
    if scheme == "central":
        beta = 0.5 * R(N, Q + 1, N, Q - 1)
        a = 0.5 * R(N + 1, Q + r, N - 1, Q - r) if Q >= r else nan
    elif scheme == "backward":
        beta = R(N, Q, N, Q - 1)
        a = R(N, Q, N - 1, Q - r) if Q >= r else nan
    elif scheme == "forward":
        beta = R(N, Q + 1, N, Q)
        a = R(N + 1, Q + r, N, Q)
    else:
        raise ValueError(scheme)
    return beta, -a


# -------------------------------------------------------------- canonical
def canonical(M, N, beta, g, r):
    """Canonical ensemble at fixed (M, N, T).  Returns dict."""
    lnZ = lnC(M, N) + beta * r * N + N * ln_zv(beta, g)
    Qm = g * N / np.expm1(beta)
    Em = -r * N + Qm
    varE = g * N * np.exp(beta) / np.expm1(beta) ** 2
    S = lnZ + beta * Em
    return dict(lnZ=lnZ, Q=Qm, E=Em, varE=varE, S=S)


# ----------------------------------------------------------- grand canonical
def grand(M, beta, bmu, g, r):
    """Grand-canonical ensemble at fixed (M, T, mu).  Returns dict."""
    lam = np.exp(bmu + beta * r + ln_zv(beta, g))     # activity of one site
    theta = lam / (1.0 + lam)
    Nm = M * theta
    nbar = g / np.expm1(beta)                          # mean quanta / molecule
    sig2 = g * np.exp(beta) / np.expm1(beta) ** 2      # variance of quanta
    u = -r + nbar
    Em = Nm * u
    Qm = Nm * nbar
    varN = M * theta * (1.0 - theta)
    varE = M * (theta * (1.0 - theta) * u ** 2 + theta * sig2)
    lnXi = M * np.log1p(lam)
    S = lnXi + beta * Em - bmu * Nm
    detcov = (M * M) * theta ** 2 * (1.0 - theta) * sig2  # det Cov(N,E)
    return dict(lam=lam, theta=theta, N=Nm, Q=Qm, E=Em, varN=varN, varE=varE,
                lnXi=lnXi, S=S, detcov=detcov)


# ------------------------------------------------------------------ MODEL A
def lnomega_A(M, N):
    return lnC(M, N)


def grand_A(M, x):
    """Model A grand canonical; x = exp(beta*(mu+eps)).  <N> = M x/(1+x)."""
    return M * x / (1.0 + x)
