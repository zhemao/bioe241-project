import sys
import argparse
import random

def mag(nx, x, ny, y):
    return (nx - x) ** 2 + (ny - y) ** 2

def expected_results(p, r, uv, tv, ur, tr):
    rv = 1 / (1 - p)
    rr = 1 / (1 - r)

    n = 0
    piv = p
    pir = 1 - p
    npiv = piv * p + pir * (1 - r)
    npir = piv * (1 - p) + pir * r

    while abs(npir - pir) > 0.001:
        piv = npiv
        pir = npir
        npiv = piv * p + pir * (1 - r)
        npir = piv * (1 - p) + pir * r
        n += 1

    pv = (1 - r) / (2 - r - p)
    pr = (1 - p) / (2 - r - p)
    at = pv * uv + pr * ur

    return rv, rr, n, pr, at

def gentime(state, uv, tv, ur, tr):
    if state == 0:
        return random.gauss(uv, math.sqrt(1.0 / tv))
    return random.gauss(ur, math.sqrt(1.0 / tr))

def simulate(p, r, uv, tv, ur, tr, n):
    states = [0] * n
    curstate = 0 if random.random() < p else 1
    states[0] = curstate

    for i in range(1, n):
        if curstate == 0:
            curstate = 0 if random.random() < p else 1
        else:
            curstate = 1 if random.random() < r else 0
        states[i] = curstate

    times = [gentime(state, uv, tv, ur, tr) for state in states]

    return states, times

def calc_avgrunlen(states):
    n = 1
    cur_state = states[0]
    runlens = [[], []]

    for state in states[1:]:
        if state == cur_state:
            n += 1
        else:
            runlens[cur_state].append(n)
            cur_state = state
            n = 1

    runlens[cur_state].append(n)

    avglens = [sum(lens) / len(lens) for lens in runlens]

    return tuple(avglens)

def state_ratio(states, cs):
    return len([state for state in states if state == cs]) / len(states)

def calc_convergence(states):
    pv = 1.0 if states[0] == 0 else 0.0
    n = len(states)

    for i in range(1, n):
        npv = state_ratio(states[:i+1], 0)
        if abs(npv - pv) < 0.001:
            return i
        pv = npv
    return n

def main():
    parser = argparse.ArgumentParser(description="Generate dog race model")
    parser.add_argument("-p", type=float)
    parser.add_argument("-r", type=float)
    parser.add_argument("--uv", type=float)
    parser.add_argument("--tv", type=float)
    parser.add_argument("--ur", type=float)
    parser.add_argument("--tr", type=float)
    parser.add_argument("-n", type=int)
    args = parser.parse_args()

    (rv, rr, cn, pr, at) = expected_results(
            args.p, args.r, args.uv, args.tv, args.ur, args.tr)

    print("Expectation: ")
    print("  Avg. Vladimir run: {}".format(rv))
    print("  Avg. Ringer run: {}".format(rr))
    print("  Days til convergence: {}".format(cn))
    print("  Ringer proportion: {}".format(pr))
    print("  Avg. time: {}\n".format(at))

    states, times = simulate(args.p, args.r, args.uv, args.tv, args.ur, args.tr, args.n)
    (rv, rr) = calc_avgrunlen(states)
    cn = calc_convergence(states)
    pr = state_ratio(states, 1)
    at = sum(times) / len(times)

    print("Simulated: ")
    print("  Avg. Vladimir run: {}".format(rv))
    print("  Avg. Ringer run: {}".format(rr))
    print("  Days til convergence: {}".format(cn))
    print("  Ringer proportion: {}".format(pr))
    print("  Avg. time: {}\n".format(at))

if __name__ == "__main__":
    main()
