"""
    根据所有预测结果，评估某个传感器对其他传感器的影响平均值
"""
import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np

from analyse_testing_result import get_exp_id
from libcity.utils import ensure_dir
from visualize_sensor import visualize_sensor_varying

if __name__ == '__main__':
    plt.rc('font', family='Times New Roman', size=20)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='METR_LA', choices=['METR_LA', 'PEMS_BAY'])
    parser.add_argument('--ow', type=int, default=2, help='output window')
    parser.add_argument('--od', type=int, default=0, help='output dim')

    # 解析参数
    args = parser.parse_args()

    sensors = 207 if args.dataset == 'METR_LA' else 325
    indices = list(range(0, 25))
    res_dir = f'analyse_testing_result5/{args.dataset}'
    ensure_dir(res_dir)

    ext = {nn: 0 for nn in range(sensors)}
    for index in indices:
        exp_id = get_exp_id(args.dataset, index)
        testing_res_dir = './libcity/cache/{}/testing_cache'.format(exp_id)
        assert os.path.exists(testing_res_dir)
        for nn in range(sensors):
            for test_nn in range(sensors):
                filename = 'ps_testing_{}_{}_{}_{}_{}.npy'.format(index, 2, nn, args.od, test_nn)
                read_data = np.load(os.path.join(testing_res_dir, filename))

                if test_nn == nn:
                    ext[test_nn] += read_data[2, 0]
                else:
                    ext[test_nn] += read_data[0, 0]

    max_value = max(ext.values())
    normalized_ext = {key: value / max_value for key, value in ext.items()}

    visualize_sensor_varying('METR_LA', args.test_nn, filename=f'{res_dir}/significance.html', ext=ext)
