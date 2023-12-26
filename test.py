import h5py
import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_comparison():
    x1 = [0.25, 0.5, 1]
    x2 = [0.25, 0.5, 1, 1.5, 2]

    y_dcrnn_paper = [[2.77, 3.15, 3.60], [5.38, 6.45, 7.59], [7.3, 8.8, 10.5]]
    y_dcrnn_libcity = [[2.76, 3.12, 3.56, 3.86, 4.09], [5.35, 6.37, 7.5, 8.2, 8.68], [7.2, 8.6, 10.3, 11.3, 12.1]]
    y_bdcrnn = [[2.89, 3.24, 3.65, 3.94, 4.13], [5.64, 6.58, 7.6, 8.23, 8.65], [7.7, 9.1, 10.8, 12, 12.7]]

    metrics = ['MAE', 'RMSE', 'MAPE']

    for i in range(3):
        plt.plot(x1, y_dcrnn_paper[i], 's-', color='r', label='Paper')
        plt.plot(x2, y_dcrnn_libcity[i], 'o-', color='g', label='DCRNN')
        plt.plot(x2, y_bdcrnn[i], '^-', color='b', label='BDCRNN')

        plt.xlabel('hour')
        plt.ylabel(metrics[i])

        plt.legend(loc='best')
        plt.savefig(f'METR_LA_{metrics[i]}.svg', bbox_inches='tight')
        plt.close()

    y_dcrnn_paper = [[1.38, 1.74, 2.07], [2.95, 3.97, 4.74], [2.9, 3.9, 4.9]]
    y_dcrnn_libcity = [[1.55, 1.93, 2.30, 2.53, 2.74], [3.18, 4.21, 5.17, 5.64, 6.06], [3.4, 4.5, 5.7, 6.3, 6.8]]
    y_bdcrnn = [[1.61, 1.93, 2.26, 2.48, 2.69], [3.41, 4.30, 5.23, 5.76, 6.22], [3.7, 4.6, 5.7, 6.4, 6.9]]

    for i in range(3):
        plt.plot(x1, y_dcrnn_paper[i], 's-', color='r', label='Paper')
        plt.plot(x2, y_dcrnn_libcity[i], 'o-', color='g', label='DCRNN')
        plt.plot(x2, y_bdcrnn[i], '^-', color='b', label='BDCRNN')

        plt.xlabel('hour')
        plt.ylabel(metrics[i])

        plt.legend(loc='best')
        plt.savefig(f'PEMS_BAY_{metrics[i]}.svg', bbox_inches='tight')
        plt.close()


def read_h5_file():
    # store = pd.HDFStore('raw_data/METR_LA/metr-la.h5')
    # axis0 = store['axis0']
    # axis1 = store['axis1']

    f = h5py.File('raw_data/METR_LA/metr-la.h5', 'r')
    data = f['array']
    f.close()

    f = h5py.File('raw_data/PEMS_BAY/pems-bay.h5', 'r')
    keys = f.keys()
    data = f['data']
    f.close()


if __name__ == '__main__':
    read_h5_file()
