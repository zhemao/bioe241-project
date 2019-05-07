import sys
import argparse
import random
import csv
import math

def gentime(state, uv, sv, ur, sr):
    if state == 0:
        return random.gauss(uv, sv)
    return random.gauss(ur, sr)

def simulate(p, r, uv, sv, ur, sr, n):
    states = [0] * n
    curstate = 0 if random.random() < p else 1
    states[0] = curstate

    for i in range(1, n):
        if curstate == 0:
            curstate = 0 if random.random() < p else 1
        else:
            curstate = 1 if random.random() < r else 0
        states[i] = curstate

    times = [gentime(state, uv, sv, ur, sr) for state in states]

    return states, times

def main():
    parser = argparse.ArgumentParser(description="Generate dog race data")
    parser.add_argument("-p", type=float)
    parser.add_argument("-r", type=float)
    parser.add_argument("--uv", type=float)
    parser.add_argument("--tv", type=float)
    parser.add_argument("--ur", type=float)
    parser.add_argument("--tr", type=float)
    parser.add_argument("-n", type=int)
    parser.add_argument("outfile")
    args = parser.parse_args()

    sv = 1 / math.sqrt(args.tv)
    sr = 1 / math.sqrt(args.tr)
    states, times = simulate(args.p, args.r, args.uv, sv, args.ur, sr, args.n)

    with open(args.outfile, "w") as f:
        writer = csv.writer(f, delimiter=' ')

        writer.writerow(["Day", "Time"])

        for i, t in enumerate(times):
            writer.writerow([str(i+1), str(t)])

if __name__ == "__main__":
    main()
