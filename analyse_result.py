import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from libcity.config import ConfigParser
from libcity.utils import get_logger, ensure_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='traffic_state_pred')
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--exp_id', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)

    args = parser.parse_args()
    task, model_name, dataset_name, exp_id = args.task, args.model, args.dataset, args.exp_id

    # load config
    config = ConfigParser(task, model_name, dataset_name,
                          config_file=None, saved_model=False, train=False, other_args=None)
    config['exp_id'] = exp_id
    # logger
    logger = get_logger(config)
    logger.info('Begin analyzing result, task={}, model_name={}, dataset_name={}, exp_id={}'.
                format(str(task), str(model_name), str(dataset_name), str(exp_id)))
    logger.info(config.config)

    evaluate_cache_dir = './libcity/cache/{}/evaluate_cache'.format(exp_id)
    analyze_cache_dir = './libcity/cache/{}/analyze_cache'.format(exp_id)
    images_cache_dir = '{}/images'.format(analyze_cache_dir)
    assert os.path.exists(evaluate_cache_dir)
    ensure_dir(analyze_cache_dir)
    ensure_dir(images_cache_dir)

    visited = False
    for filename in os.listdir(evaluate_cache_dir):
        if filename[-4:] == '.npz':
            assert not visited
            arr = np.load(f'{evaluate_cache_dir}/{filename}')
            visited = True

    assert visited
    prediction, truth, outputs, sigmas = arr['prediction'], arr['truth'], arr['outputs'], arr['sigmas']
    logger.info(f'prediction shape: {prediction.shape}')
    logger.info(f'truth shape: {truth.shape}')
    logger.info(f'outputs shape: {outputs.shape}')
    logger.info(f'sigmas shape: {sigmas.shape}')

    evaluate_rep, num_data, output_window, num_nodes, output_dim = outputs.shape
    assert outputs.shape == sigmas.shape and output_dim == 1
    prediction, truth, outputs, sigmas = prediction[..., 0], truth[..., 0], outputs[..., 0], sigmas[..., 0]
    for i in range(10):
        logger.info(f'Analyzing node {i}...')
        # (num_data, output_window), (evaluate_rep, num_data, output_window)
        prediction_node, truth_node, outputs_node, sigmas_node = prediction[:, :, i], truth[:, :, i], \
                                                                 outputs[:, :, :, i], sigmas[:, :, :, i]
        sigmas_node_2 = sigmas_node * sigmas_node
        outputs_node_2 = outputs_node * outputs_node
        prediction_node_2 = prediction_node * prediction_node
        writer = pd.ExcelWriter(f'{analyze_cache_dir}/{dataset_name}_node_{i}.xlsx')
        for j, time in zip([2, 5, 11], ['15min', '30min', '1h']):
            p, t, o, s = prediction_node[:, j], truth_node[:, j], outputs_node[:, :, j], sigmas_node[:, :, j]
            p2, s2, o2 = prediction_node_2[:, j], sigmas_node_2[:, :, j], outputs_node_2[:, :, j]
            error = p - t
            a_uncertainty = 1 / evaluate_rep * np.sum(s2, axis=0)
            e_uncertainty = 1 / evaluate_rep * np.sum(o2, axis=0) - p2
            uncertainty = a_uncertainty + e_uncertainty
            res = np.stack((p, t, error, uncertainty, a_uncertainty, e_uncertainty), axis=1)
            res = np.concatenate((res, o.T, s.T), axis=1)
            columns_name = ['pred', 'truth', 'error', 'uncertainty', 'a_uncertainty', 'e_uncertainty']
            columns_name.extend(['output_{}'.format(i) for i in range(evaluate_rep)])
            columns_name.extend(['sigma_{}'.format(i) for i in range(evaluate_rep)])
            pd_data = pd.DataFrame(res, columns=columns_name)
            pd_data.to_excel(writer, sheet_name=time, float_format='%.4f')
            for k in range(0, num_data, args.batch_size):
                t_num = min(args.batch_size, num_data - k)
                x = np.arange(k, k + t_num)
                plt.plot(x, error[k:k + t_num], 'ro-', label='error')
                plt.plot(x, a_uncertainty[k:k + t_num], 'bv-', label='a_uncertainty')
                plt.plot(x, e_uncertainty[k:k + t_num], 'g^-', label='e_uncertainty')
                plt.plot(x, uncertainty[k:k + t_num], 'ys-', label='uncertainty')
                plt.legend()
                plt.savefig(f'{images_cache_dir}/{dataset_name}_node_{i}_batch_{k // args.batch_size}_{time}.svg',
                            bbox_inches='tight')
                plt.close()
        writer.close()
