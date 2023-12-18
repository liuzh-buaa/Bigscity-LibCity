"""
    python analyse_testing_result.py --exp_id 11529 (BDCRNNVariableDecoder; METR_LA )
    python analyse_testing_result.py --exp_id 53968 (BDCRNNVariableDecoderShared; PEMS_BAY )
"""
import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    plt.rc('font', family='Times New Roman', size=20)
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, default=53968, choices=[11529, 53968], help='id of experiment')
    parser.add_argument('--data', type=int, default=0, help='data index')
    parser.add_argument('--ow', type=int, default=-1, help='output window')
    parser.add_argument('--nn', type=int, default=6, help='node number')
    parser.add_argument('--od', type=int, default=0, help='output dim')
    parser.add_argument('--test_nn', type=int, default=None, help='')

    # 解析参数
    args = parser.parse_args()

    if args.ow == -1:
        args.ow = [2, 5, 11]
    else:
        args.ow = [args.ow]

    if args.test_nn is None:
        args.test_nn = args.nn

    testing_res_dir = './libcity/cache/{}/testing_cache'.format(args.exp_id)
    assert os.path.exists(testing_res_dir)

    for ow in args.ow:
        filename = 'ps_testing_{}_{}_{}_{}_{}.npy'.format(args.data, ow, args.nn, args.od, args.test_nn)
        read_data = np.load(os.path.join(testing_res_dir, filename))

        print('Reading testing results of DATA {}: Output window {} of Node {} wrt Node {} - {}'.format(
            args.data, ow, args.nn, args.test_nn, read_data.shape))
        print(read_data)  # (12, 2)
