python run_model.py --task traffic_state_pred --dataset PEMS_BAY --model DCRNN --gpu_id 3

**30363-DCRNN-PEMS_BAY-May-26-2023_20-02-54.log 原始版本**
56334-BDCRNNRegConstant-PEMS_BAY-May-26-2023_12-25-07.log 0.01,0.01,0.01 (close)
65906-BDCRNNRegConstantShared-PEMS_BAY-May-26-2023_20-27-59.log 0.01,0.01,0.01 (close)
34628-BDCRNNRegVariable-PEMS_BAY-May-28-2023_08-37-08.log (worst)
44732-BDCRNNRegVariableShared-PEMS_BAY-May-28-2023_16-37-44.log (worse)
`建模log_sigma_0改为建模sigma_0，再通过类ReLU函数变换取1e-4作为最小值，正则化系数考虑建模sigma_0的参数`
19216-BDCRNNRegVariableShared-PEMS_BAY-Jun-01-2023_13-58-00.log 0.01, 0.01 (worse)
59503-BDCRNNRegVariable-PEMS_BAY-Jun-02-2023_11-51-40.log 0.01, 0.01 (worse)
`正则化系数不考虑建模sigma_0的参数`
80794-BDCRNNRegVariableShared-PEMS_BAY-Jun-02-2023_22-24-30.log (worst)
44278-BDCRNNRegVariable-PEMS_BAY-Jun-04-2023_13-28-34.log (worst)
75689-BDCRNNRegVariableSharedLayer-PEMS_BAY-Jun-04-2023_22-52-39.log (worse)
`所有的训练损失采用mse，修复之前relu函数梯度不为0以及应该加log_sigma0的bug，在config文件中指定正则项是否包括编解码sigma_0的参数以及relu函数的粒度`
17891-BDCRNNConstant-PEMS_BAY-Jun-07-2023_14-15-01.log(close to paper, a little worse than libcity) MSE训练
38225-BDCRNNVariableLayer-PEMS_BAY-Jun-08-2023_05-18-35.log(worst) MSE训练，sigma_0也一起inverse_transform了，参数初始0.1
73236-BDCRNNVariableSharedLayer-PEMS_BAY-Jun-08-2023_05-18-43.log(worst) MSE训练，sigma_0也一起inverse_transform了，参数初始0.1
**41268-BDCRNNConstant-PEMS_BAY-Jun-09-2023_10-54-45.log(close & better) MAE训练**
**6496-BDCRNNConstantShared-PEMS_BAY-Jun-09-2023_12-24-46.log(close & better) MAE训练**
44659-BDCRNNLogVariableLayer-PEMS_BAY-Jun-09-2023_12-41-20.log(worst)
72567-BDCRNNLogVariableSharedLayer-PEMS_BAY-Jun-10-2023_14-41-40.log(worst)
36039-BDCRNNLogVariable-PEMS_BAY-Jun-10-2023_23-12-48.log(worse) reg false
92935-BDCRNNLogVariableShared-PEMS_BAY-Jun-11-2023_15-06-23.log(worse) reg false
23171-BDCRNNLogVariableFC-PEMS_BAY-Jun-12-2023_10-41-19.log(worst)
64449-BDCRNNLogVariableSharedFC-PEMS_BAY-Jun-14-2023_18-27-19.log(worst)
98415-BDCRNNLogVariable-PEMS_BAY-Jun-12-2023_08-49-20.log(worst) reg true
8110-BDCRNNLogVariableShared-PEMS_BAY-Jun-12-2023_08-50-03.log(worst) reg true
76047-BDCRNNVariableDecoderShared-PEMS_BAY-Jun-22-2023_07-00-34.log(close) no reg 
51771-BDCRNNVariableDecoder-PEMS_BAY-Jun-24-2023_02-09-54.log(close) no reg
33178-BDCRNNVariableDecoderSharedFC-PEMS_BAY-Jun-23-2023_06-29-54.log(close) no reg
94169-BDCRNNVariableDecoderFC-PEMS_BAY-Jun-24-2023_15-04-59.log(worse) no reg
63420-BDCRNNLogVariableDecoder-PEMS_BAY-Jul-02-2023_22-27-31.log(worse) no reg
31113-BDCRNNLogVariableDecoderShared-PEMS_BAY-Jul-02-2023_22-28-26.log(worse) no reg
11235-BDCRNNVariableDecoder-PEMS_BAY-Jul-21-2023_11-37-15.log copy 51771
32087-BDCRNNVariableDecoder-PEMS_BAY-Jul-24-2023_15-41-38.log clip=true, reg sigma_0=false(close)
***98286-BDCRNNVariableDecoder-PEMS_BAY-Jul-26-2023_13-23-01.log clip=true, reg sigma_0=true (close)***
92553-BDCRNNVariableDecoderShared-PEMS_BAY-Jul-28-2023_08-54-39.log reg sigma_0=false (close)
28885-BDCRNNVariableDecoderShared-PEMS_BAY-Jul-28-2023_08-59-16.log reg sigma_0=true (early stopping)
96111-BDCRNNVariableLayer-PEMS_BAY-Jul-31-2023_02-45-42.log no reg (close)
5599-BDCRNNVariableSharedLayer-PEMS_BAY-Jul-31-2023_02-45-51.log no reg (false)
75311-BDCRNNVariableDecoderShared-PEMS_BAY-Aug-26-2023_21-46-57.log no reg (Early Stopping) (worse)
6335-BDCRNNVariableDecoder-PEMS_BAY-Aug-26-2023_21-32-12.log no reg (close)
98245-DCRNN-PEMS_BAY-Aug-28-2023_22-26-16.log input_window=output_window=24
49384-BDCRNNVariableDecoder-PEMS_BAY-Aug-26-2023_13-34-11.log input_window=output_window=24
29898-BDCRNNVariableDecoder-PEMS_BAY-Oct-19-2023_02-55-45.log shuffle=False (worst)
82244-DCRNN-PEMS_BAY-Oct-21-2023_15-01-27.log shuffle=False (worse)
36283-BDCRNNVariableDecoder-PEMS_BAY-Oct-21-2023_14-54-36.log shuffle=False (worst)
---------------------------------------------------------------------------------------------------------------------
95500-DCRNN-PEMS_BAY-Nov-02-2023_15-30-19.log shuffle=True (2.78)

42704-BDCRNNVariableDecoder-PEMS_BAY-Nov-02-2023_15-36-35.log shuffle=True, sigma_pi=sigma_sigma_pi=1 sigma_start=sigma_sigma_start=0.01 lr=0.005 lr_decay_ratio=0.5 (2.81)

9422-BDCRNNVariableDecoderShared-PEMS_BAY-Nov-08-2023_11-10-14.log shuffle=True, sigma_pi=sigma_sigma_pi=1 sigma_start=sigma_sigma_start=0.01 lr=0.005 lr_decay_ratio=0.5