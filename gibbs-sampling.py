import random
import argparse
import pandas as pd
import numpy as np
import scipy.stats
from scipy.integrate import quad

class trans_prob_dist(scipy.stats.rv_continuous):
    def __init__(self, c0, c1):
        self.c0 = c0
        self.c1 = c1
        (self.A, err) = quad(self._numer, 0.0, 1.0)
        super(trans_prob_dist, self).__init__(a = 0.0, b = 1.0)

    def _numer(self, p):
        return p ** self.c0 * (1 - p) ** self.c1

    def _pdf(self, p):
        return self._numer(p) / self.A

def trans_count(x, i, j):
    cnt = 0
    for (xn, xnp) in zip(x[:-1], x[1:]):
        if (xn == i) and (xnp == j):
            cnt += 1
    return cnt

def moment(x, y, i, k):
    m = 0.0
    for (xn, yn) in zip(x, y):
        if xn == i:
            m += yn ** k
    return m

def gibbs_sample(p, r, uv, tv, ur, tr, x, y, niters):
    ps = np.zeros(niters, dtype=np.float64)
    rs = np.zeros(niters, dtype=np.float64)
    uvs = np.zeros(niters, dtype=np.float64)
    tvs = np.zeros(niters, dtype=np.float64)
    urs = np.zeros(niters, dtype=np.float64)
    trs = np.zeros(niters, dtype=np.float64)

    for itr in range(0, niters):
        c00 = trans_count(x, 0, 0) + int(x[0] == 0)
        c01 = trans_count(x, 0, 1) + int(x[0] == 1)
        c10 = trans_count(x, 1, 0)
        c11 = trans_count(x, 1, 1)
        p = trans_prob_dist(c00, c01).rvs()
        r = trans_prob_dist(c11, c10).rvs()

        m00 = moment(x, y, 0, 0)
        m01 = moment(x, y, 0, 1)
        m02 = moment(x, y, 0, 2)
        a0 = m00 / 2
        b0 = 0.5 * (m02 - m01 ** 2 / m00)
        tv = scipy.stats.gamma(a0, scale=1/b0).rvs()
        e0 = m01 / m00
        s0 = 1 / np.sqrt(m00 * tv)
        uv = scipy.stats.norm(e0, s0).rvs()

        m10 = moment(x, y, 1, 0)
        m11 = moment(x, y, 1, 1)
        m12 = moment(x, y, 1, 2)
        a1 = m10 / 2
        b1 = 0.5 * (m12 - m11 ** 2 / m10)
        tr = scipy.stats.gamma(a1, scale=1/b1).rvs()
        e1 = m11 / m10
        s1 = 1 / np.sqrt(m10 * tr)
        ur = scipy.stats.norm(e1, s1).rvs()

        sv = 1 / np.sqrt(tv)
        sr = 1 / np.sqrt(tr)
        normv = scipy.stats.norm(uv, sv)
        normr = scipy.stats.norm(ur, sr)

        t0 = p if x[1] == 0 else (1 - p)
        t1 = (1 - r) if x[1] == 0 else r
        px0 = p * normv.pdf(y[0]) * t0
        px1 = (1 - p) * normr.pdf(y[0]) * t1
        x[0] = random.choices([0, 1], weights=[px0, px1])[0]

        for i in range(1, len(x)-1):
            tin0 = p if x[i-1] == 0 else (1 - r)
            tin1 = (1 - p) if x[i-1] == 0 else r
            tout0 = p if x[i+1] == 0 else (1 - p)
            tout1 = (1 - r) if x[i+1] == 0 else r
            px0 = tin0 * normv.pdf(y[i]) * tout0
            px1 = tin1 * normr.pdf(y[i]) * tout1
            x[i] = random.choices([0, 1], weights=[px0, px1])[0]

        n = len(y)
        t0 = p if x[-2] == 0 else (1 - r)
        t1 = (1 - p) if x[-2] == 0 else  r
        px0 = t0 * normv.pdf(y[n-1])
        px1 = t1 * normr.pdf(y[n-1])
        x[-1] = random.choices([0, 1], weights=[px0, px1])[0]

        ps[itr] = p
        rs[itr] = r
        uvs[itr] = uv
        tvs[itr] = tv
        urs[itr] = ur
        trs[itr] = tr

    return ps, rs, uvs, tvs, urs, trs

def print_range(name, values):
    print("{} = {} +/- {}".format(name, np.mean(values), np.std(values)))

def main():
    parser = argparse.ArgumentParser(description="Gibbs sampling for Dog Race model")
    parser.add_argument("-i", dest="niters", type=int, default=100)
    parser.add_argument("inputfile")
    args = parser.parse_args()

    df = pd.read_csv(args.inputfile, sep=' ')
    y = df["Time"]

    # Initial values
    p = 0.5
    r = 0.5

    x = [random.choice([0, 1]) for _ in y]

    vvalues = [y[i] for (i, x) in enumerate(x) if x == 0]
    rvalues = [y[i] for (i, x) in enumerate(x) if x == 1]

    uv = np.mean(vvalues)
    sv = np.mean(vvalues)
    tv = 1.0 / sv ** 2

    ur = np.mean(rvalues)
    sr = np.mean(rvalues)
    tr = 1.0 / sr ** 2

    (ps, rs, uvs, tvs, urs, trs) = gibbs_sample(p, r, uv, tv, ur, tr, x, y, args.niters)

    print_range("p", ps)
    print_range("r", rs)
    print_range("uv", uvs)
    print_range("tv", tvs)
    print_range("ur", urs)
    print_range("tr", trs)

if __name__ == "__main__":
    main()
