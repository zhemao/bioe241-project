import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser("Dump statistics about dataset")
    parser.add_argument("inputfile")
    args = parser.parse_args()

    df = pd.read_csv(args.inputfile, sep=' ')
    y = df["Time"]

    print("Average time: {}".format(np.mean(y)))
    print("Stdev time: {}".format(np.std(y)))

if __name__ == "__main__":
    main()
