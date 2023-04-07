import torch

exp_id = 91971
model_name = 'DCRNN'
dataset_name = 'METR_LA'

torch.cuda.set_device(2)

model_cache_file = './libcity/cache/{}/model_cache/{}_{}.m'.format(exp_id, model_name, dataset_name)
model_state, optimizer_state = torch.load(model_cache_file, map_location='cuda:2')
print('ok')
