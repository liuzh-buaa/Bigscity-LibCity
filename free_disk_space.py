import os
import shutil

if __name__ == '__main__':
    cache_dir = './libcity/cache/'
    for exp_id in os.listdir(cache_dir):
        if exp_id in ['dataset_cache', 'hyper_tune']:
            continue
        exp_cache_dir = cache_dir + exp_id + '/'
        evaluate_cache_dir = exp_cache_dir + 'evaluate_cache/'
        model_cache_dir = exp_cache_dir + 'model_cache/'
        if not os.listdir(evaluate_cache_dir):
            print('Deleting {}'.format(exp_cache_dir))
            shutil.rmtree(exp_cache_dir)
        else:
            for filename in os.listdir(model_cache_dir):
                if filename[-4:] == '.tar':
                    print('Deleting {}{}'.format(model_cache_dir, filename))
                    os.remove('{}{}'.format(model_cache_dir, filename))
