"""
    生成 METR-LA 或 PEMS-BAY 数据集所有预测点对【自己传感器】【过去12个时刻】的检测结果
"""
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyse_testing_result import get_exp_id
from libcity.utils import ensure_dir

if __name__ == '__main__':
    plt.rc('font', family='Times New Roman', size=20)

    dataset = 'METR_LA'
    indices = list(range(0, 25))
    sensors = 207
    res_dir = 'analyse_testing_result3/METR_LA'

    # dataset = 'PEMS_BAY'
    # indices = []
    # sensors = 325
    # res_dir = 'analyse_testing_result1/PEMS_BAY'

    ensure_dir(res_dir)

    for sensor in range(sensors):
        tot_data = []
        for index in indices:
            exp_id = get_exp_id(dataset, index)

            testing_res_dir = './libcity/cache/{}/testing_cache'.format(exp_id)
            assert os.path.exists(testing_res_dir)

            for test_sensor in range(sensors):
                filename = 'ps_testing_{}_{}_{}_{}_{}.npy'.format(index, 2, sensor, 0, test_sensor)
                read_data = 1 - np.load(os.path.join(testing_res_dir, filename))
                if sensor == test_sensor:
                    tot_data.append(read_data[2, 0])
                else:
                    tot_data.append(read_data[0, 0])

        tot_data = np.array(tot_data).reshape((len(indices), sensors))   # (len(indices), 207)
        df = pd.DataFrame(tot_data, index=indices)
        df.to_csv(f'{res_dir}/{sensor}.csv')
