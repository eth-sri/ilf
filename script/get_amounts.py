import os
import json
import argparse
import numpy as np
from collections import Counter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', dest='train_dir', type=str, required=True)
    args = parser.parse_args()
    return args


args = get_args()


def main():
    amounts = Counter()
    for f in os.listdir(args.train_dir):
        if not f.endswith('.data'):
            continue

        with open(os.path.join(args.train_dir, f)) as fin:
            for line in fin:
                d = json.loads(line)
                if d['type'] != 'tx':
                    continue

                method = d['tx']['method']

                if d['features']['methods'][method][4] > 0:
                    amount = d['tx']['amount']
                    amounts[amount] += 1

    print('hex_value value: frequency')
    for k, v in sorted(amounts.items(), key=lambda i: i[1], reverse=True):
        print('{} {}: {}'.format(hex(k), k, v))


if __name__ == '__main__':
    main()