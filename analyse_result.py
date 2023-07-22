import argparse
import os.path

import numpy as np

from libcity.config import ConfigParser
from libcity.utils import get_logger, ensure_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='traffic_state_pred')

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
    assert os.path.exists(evaluate_cache_dir)
    ensure_dir(analyze_cache_dir)

    visited = False
    for filename in os.listdir(evaluate_cache_dir):
        if filename[-4:] == '.npz':
            assert not visited
            arr = np.load(f'{evaluate_cache_dir}/{filename}')
            visited = True

    assert visited
    prediction, truth, pre_outputs = arr['prediction'], arr['truth'], arr['pre_outputs']
    logger.info('pre_outputs shape: ', pre_outputs.shape)
    logger.info('prediction shape: ', prediction.shape)
    logger.info('truth shape: ', truth.shape)

    evaluate_rep, num_data, output_window, num_nodes, output_dim = pre_outputs.shape
    assert output_dim == 1
    prediction, truth, pre_outputs = prediction[..., 0], truth[..., 0], pre_outputs[..., 0]
    for i in range(num_nodes):
        # (num_data, output_window), (num_data, output_window), (evaluate_rep, num_data, output_window)
        prediction_node, truth_node, pre_outputs_node = prediction[:, :, i], truth[:, :, i], pre_outputs[:, :, :, i]
        print('ok')