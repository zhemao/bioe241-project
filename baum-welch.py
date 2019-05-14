import argparse
import pandas as pd
import numpy as np
import scipy.stats as spst
import itertools

FP_SCALE = 1e300

def gen_prob_tables(p, r, uv, sv, ur, sr):
    U = np.array([p, 1 - p])
    T = np.array([[p, 1-p], [1 -r, r]])
    norms = [spst.norm(uv, sv), spst.norm(ur, sr)]

    return U, T, norms

def forward_table(y, U, T, norms):
    m = len(norms)
    L = len(y)
    F = np.zeros((L, m), dtype=np.float64)

    for i in range(0, m):
        F[0, i] = norms[i].pdf(y[0]) * U[i] * FP_SCALE

    for n in range(1, L):
        for i in range(0, m):
            e = norms[i].pdf(y[n])
            f = F[n - 1]
            t = T[:,i]
            F[n, i] = e * np.dot(f, t)

    return F

def backward_table(y, U, T, norms):
    m = len(norms)
    L = len(y)
    B = np.zeros((len(y), 2), dtype=np.float64)

    for i in range(0, m):
        B[L-1, i] = FP_SCALE

    for n in range(L-2, -1, -1):
        for j in range(0, m):
            t = T[j]
            e = np.array([norms[i].pdf(y[n+1]) for i in range(0, m)])
            b = B[n+1]
            B[n, j] = np.sum(t * e * b)

    return B

def ecij_table(y, F, B, T, norms, PY):
    C = np.zeros((2, 2), dtype=np.float64)

    m = len(norms)

    for (i, j) in itertools.product(range(0, m), range(0, m)):
        t = T[i, j]
        f = F[:-1, i]
        b = B[1:, j]
        e = np.array([norms[j].pdf(yn) for yn in y[1:]])
        c = f * b * t * e
        C[i, j] = np.sum(c)

    return C / (FP_SCALE * PY)

def moment_table(k, y, Pxi):
    m = Pxi.shape[1]
    M = np.zeros(m)
    yk = y ** k

    for i in range(0, m):
        Px = Pxi[:,i]
        M[i] = np.dot(yk, Px)

    return M / FP_SCALE

def convergence_factor(p, r, u, s, pn, rn, un, sn):
    prat = abs(pn - p) / p
    rrat = abs(rn - r) / r
    urat = [abs(uni - ui) / ui for (uni, ui) in zip(un, u)]
    srat = [abs(sni - si) / si for (sni, si) in zip(sn, s)]

    return prat + rrat + sum(urat + srat)

def calc_next_params(y, p, r, uv, sv, ur, sr):
    U, T, norms = gen_prob_tables(p, r, uv, sv, ur, sr)

    F = forward_table(y, U, T, norms)
    B = backward_table(y, U, T, norms)

    PY = np.sum(F[-1])
    Pxi = F * B / PY
    C = ecij_table(y, F, B, T, norms, PY)
    M0 = moment_table(0, y, Pxi)
    M1 = moment_table(1, y, Pxi)
    M2 = moment_table(2, y, Pxi)

    un = M1 / M0
    sn = np.sqrt(M2 / M0 - M1**2 / M0**2)

    Xp = p + C[0,0]
    Yp = (1 - p) + C[0,1]
    Xr = C[1,1]
    Yr = C[1,0]

    pn = Xp / (Xp + Yp)
    rn = Xr / (Xr + Yr)

    conv = convergence_factor(p, r, [uv, ur], [sv, sr], pn, rn, un, sn)

    return pn, rn, un[0], sn[0], un[1], sn[1], conv

def calc_final_prob(y, p, r, uv, sv, ur, sr):
    U, T, norms = gen_prob_tables(p, r, uv, sv, ur, sr)
    m = len(norms)

    F = forward_table(y, U, T, norms)
    B = backward_table(y, U, T, norms)

    PY = np.sum(F[-1])
    Px = F[-1] / PY
    ybar = np.mean(y)
    indices = itertools.product(range(0, m), range(0, m))
    Pext = sum([Px[i] * T[i, j] * norms[j].cdf(ybar) for (i, j) in indices])

    return Px, Pext

def main():
    parser = argparse.ArgumentParser(description="Baum-Welch training for Dog Race model")
    parser.add_argument("-c", dest="conv_threshold", type=float, default=0.001)
    parser.add_argument("inputfile")
    args = parser.parse_args()

    df = pd.read_csv(args.inputfile, sep=' ')
    y = df["Time"]
    L = int(len(y) / 2)

    p = 0.5
    r = 0.5
    uv = np.mean(y[:L])
    sv = np.std(y[:L])
    ur = np.mean(y[L:])
    sr = np.std(y[L:])
    conv = float("inf")
    niter = 0

    print("Compute EM for dataset of {} points".format(len(y)))

    while conv > args.conv_threshold:
        (p, r, uv, sv, ur, sr, conv) = calc_next_params(y, p, r, uv, sv, ur, sr)
        niter += 1

    print("Finished EM in {} cycles".format(niter))

    print("p  = {}".format(p))
    print("r  = {}".format(r))
    print("uv = {}".format(uv))
    print("tv = {}".format(1 / sv ** 2))
    print("ur = {}".format(ur))
    print("tr = {}".format(1 / sr ** 2))

    Px, Pext = calc_final_prob(y, p, r, uv, sv, ur, sr)
    print("Prob. of X on last day: {}".format(Px))
    print("Prob. of smaller-than-average run on additional day: {}".format(Pext))

if __name__ == "__main__":
    main()
