python run_model.py --task traffic_state_pred --dataset METR_LA --model DCRNN --gpu_id 2

**91971-DCRNN-METR_LA-Mar-17-2023_21-06-42.log 跑通原始版本的DRCNN**
35739-BDCRNN-METR_LA-Mar-19-2023_21-07-43.log 修改得到evaluate=1下的BDRCNN，无正则化
31078-BDCRNN-METR_LA-Mar-19-2023_22-01-16.log 修改得到evaluate=5下的BDRCNN，无正则化，**之后的实验都是建立在rep=5的基础上**
`前面3种没有输出参数的先验分布，默认是均值为0，标准差为1，初始标准差也为1`
62240-BDCRNN-METR_LA-Mar-21-2023_21-43-08.log 参数先验分布均值为0，标准差为0.1，初始标准差为0.1，数据噪声标准差固定为0.1 (equal)
24036-BDCRNN-METR_LA-Mar-21-2023_21-51-04.log 参数先验分布均值为0，标准差为0.1，初始标准差为0.01，数据噪声标准差为0.1（better）
`上面两种数据噪声的大小并没有影响，因为还没有添加正则化项`
53948-BDCRNNRegConstant-METR_LA-Mar-22-2023_19-52-57.log 参数先验分布均值为0，标准差0.1，初始标准差0.1，数据噪声标准差固定为0.1 (equal)
71907-BDCRNNRegConstant-METR_LA-Mar-22-2023_19-55-36.log 参数先验分布均值为0，标准差0.1，初始标准差0.1，数据噪声标准差固定为0.01 (better & worse)
65773-BDCRNNRegConstantShared-METR_LA-Mar-23-2023_11-47-30.log 参数先验分布均值为0，标准差0.1，初始标准差0.1，数据噪声标准差固定为0.01 (better & worse)
84751-BDCRNNRegConstantShared-METR_LA-Mar-23-2023_11-49-31.log 参数先验分布均值为0，标准差0.1，初始标准差0.1，数据噪声标准差固定为0.1 (worse)
24925-BDCRNNRegVariable-METR_LA-Mar-23-2023_22-15-24.log 参数先验分布均值为0，标准差0.1，初始标准差0.1 (better & worse)
79308-BDCRNNRegVariableShared-METR_LA-Mar-23-2023_22-22-44.log 参数先验分布均值为0，标准差0.1，初始标准差0.1 (未收敛)
`之前的实验evaluate阶段重复5次，之后在模型config文件中设置参数evaluate_rep，默认为30；此外，之后的实验修改训练流程，同时输出加正则化项后的损失和加之前的损失，方便对比`
22527-BDCRNNRegConstant-METR_LA-Mar-30-2023_17-56-30.log 参数先验分布均值为0，标准差0.1，初始标准差0.1，数据噪声标准差固定为0.01 (worse)
5855-BDCRNNRegConstantShared-METR_LA-Mar-30-2023_18-04-05.log 参数先验分布均值为0，标准差0.1，初始标准差0.1，数据噪声标准差固定为0.01 (worse)
17989-BDCRNNRegVariable-METR_LA-Mar-30-2023_18-04-46.log 参数先验分布均值为0，标准差0.1，初始标准差0.1 (worse)
8746-BDCRNNRegVariableShared-METR_LA-Mar-30-2023_18-05-59.log 参数先验分布均值为0，标准差0.1，初始标准差0.1 (worse)
13674-BDCRNNRegVariableShared-METR_LA-Apr-01-2023_10-10-22.log 参数先验分布均值为0，标准差1，初始标准差1 (worse)
23085-BDCRNNRegVariableShared-METR_LA-Mar-31-2023_14-49-08.log 参数先验分布均值为0，标准差1，初始标准差0.1 (worst)
99850-BDCRNNRegVariableShared-METR_LA-Apr-01-2023_10-06-48.log 参数先验分布均值为0，标准差0.1，初始标准差0.1，lr_decay 0.5 (未收敛)
41241-BDCRNNRegVariableShared-METR_LA-Apr-01-2023_10-13-17.log 参数先验分布均值为0，标准差0.1，初始标准差0.1，lr 0.1 (worst)
25142-BDCRNNRegVariableShared-METR_LA-Apr-01-2023_20-40-47.log 参数先验分布均值为0，标准差0.01，初始标准差0.01 (worse)
`fix bug!之前的reg正则化项部分有bug，没有考虑除以y的大小，导致mae/mse和正则化项尺度不匹配`
748-BDCRNNRegVariableShared-METR_LA-Apr-02-2023_16-41-23.log 参数先验分布均值为0，标准差0.1，初始标准差0.1 (worse)
3796-BDCRNNRegVariable-METR_LA-Apr-02-2023_18-11-26.log 参数先验分布均值为0，标准差0.1，初始标准差0.1 (worse)
96135-BDCRNNRegConstant-METR_LA-Apr-02-2023_19-36-05.log 参数先验分布均值为0，标准差0.1，初始标准差0.1，数据噪声标准差固定为0.01 (close)
94727-BDCRNNRegConstantShared-METR_LA-Apr-03-2023_09-22-32.log 参数先验分布均值为0，标准差0.1，初始标准差0.1，数据噪声标准差固定为0.01 (worse)
86046-BDCRNNRegVariableShared-METR_LA-Apr-03-2023_14-58-19.log 参数先验分布均值为0，标准差0.1，初始标准差0.1，lr_decay 0.5 (worse)
`fix bug?之前的异方差的reg正则化项没有考虑建模sigma_0的模型参数；计算真值非0的数据大小错误`
60526-BDCRNNRegVariable-METR_LA-Apr-04-2023_11-12-38.log 参数先验分布均值为0，标准差0.1，初始标准差0.1 (worse)
80573-BDCRNNRegVariableShared-METR_LA-Apr-04-2023_11-13-23.log 参数先验分布均值为0，标准差0.1，初始标准差0.1 (worse)
5596-BDCRNNRegVariableShared-METR_LA-Apr-05-2023_10-38-53.log 参数先验分布均值为0，标准差1，初始标准差1 (worst)
23593-BDCRNNRegVariableShared-METR_LA-Apr-05-2023_20-48-08.log 参数先验分布均值为0，标准差0.1，初始标准差0.1, lr_decay 0.5 (worse)
45889-BDCRNNRegVariableShared-METR_LA-Apr-06-2023_10-25-03.log 参数先验分布均值为0，标准差0.01，初始标准差0.01 (worse)
`把异方差的模型中对y和sigma建模的两部分模型初始化sigma_pi和sigma_start拆开`
97583-BDCRNNRegVariableShared-METR_LA-Apr-07-2023_08-50-11.log 参数1,2先验分布均值为0，标准差1，初始标准差0.01, lr_decay 0.5 (未收敛)
`fix bug?之前的decoder模型中有一个Linear层没有替换为RandLinear，所有模型都存在这个问题`
22389-BDCRNNRegVariableShared-METR_LA-Apr-07-2023_18-51-18.log 参数1,2先验分布均值为0，标准差0.1，初始标准差0.1，仅测试 (worst)
96188-BDCRNNRegVariableShared-METR_LA-Apr-07-2023_18-51-51.log 参数1,2先验分布均值为0，标准差0.1，初始标准差0.1, 初始化mu加载dcrnn训练好的参数，仅测试 (worse)
85260-BDCRNNRegVariableShared-METR_LA-Apr-07-2023_21-42-30.log 参数1,2先验分布均值为0，标准差0.1，初始标准差0.01, 初始化mu加载dcrnn训练好的参数，仅测试 (close)
93824-BDCRNNRegVariableShared-METR_LA-Apr-07-2023_21-46-31.log 参数1,2先验分布均值为0，标准差0.1，初始标准差0.001, 初始化mu加载dcrnn训练好的参数，仅测试 (close)
`fix bug?separate sigma_pi and sigma_start`
11559-BDCRNNRegVariableShared-METR_LA-Apr-07-2023_23-14-47.log 参数1,2先验分布均值为0，标准差0.01，初始标准差0.01，初始化mu加载dcrnn训练好的参数 (worst)
86800-BDCRNNRegVariableShared-METR_LA-Apr-07-2023_23-18-12.log 参数1,2先验分布均值为0，标准差0.1，初始标准差0.01，初始化mu加载dcrnn训练好的参数 (worse)
2479-BDCRNNRegConstantShared-METR_LA-Apr-11-2023_19-58-35.log 参数先验分布均值为0，标准差0.1，初始标准差0.1，数据噪声标准差固定为0.01 (worse)
69410-BDCRNNRegConstant-METR_LA-Apr-11-2023_20-01-46.log 参数先验分布均值为0，标准差0.1，初始标准差0.1，数据噪声标准差固定为0.01 (worse)
14950-BDCRNNRegConstant-METR_LA-Apr-13-2023_09-50-30.log 参数先验分布均值为0，标准差0.05，初始标准差0.05，数据噪声标准差固定为0.01 (close)
13243-BDCRNNRegVariableShared-METR_LA-Apr-13-2023_09-50-42.log 参数1,2先验分布均值为0，标准差0.05，初始标准差0.05 (worst)
95415-BDCRNNRegVariableShared-METR_LA-Apr-14-2023_21-50-17.log 参数1,2先验分布均值为0，标准差0.005，初始标准差0.005 (worst)
87897-BDCRNNRegConstant-METR_LA-Apr-14-2023_21-51-39.log 参数先验分布均值为0，标准差0.01，初始标准差0.01，数据噪声标准差固定为0.01 (close)
92453-BDCRNNRegConstantShared-METR_LA-Apr-16-2023_09-27-25.log 参数先验分布均值为0，标准差0.01，初始标准差0.01，数据噪声标准差固定为0.01 (close)
80293-BDCRNNRegVariableShared-METR_LA-Apr-16-2023_15-09-18.log 参数1,2先验分布均值为0，标准差0.01，初始标准差0.01，初始化参数2的模型参数为0.1的初始化 (worst)
99905-BDCRNNRegVariable-METR_LA-Apr-18-2023_00-02-41.log 参数1,2先验分布均值为0，标准差0.01，初始标准差0.01 (worst)
35903-BDCRNNRegVariableShared-METR_LA-Apr-20-2023_12-10-41.log 参数先验分布均值为0，标准差0.1，初始标准差0.1，lr_decay 0.5，训练loss改为mse (worse)
91505-BDCRNNRegVariable-METR_LA-Apr-20-2023_23-12-05.log 参数先验分布均值为0，标准差0.1，初始标准差0.1，lr_decay 0.5，训练loss改为mse (worse)
`更改模型架构，编码器采用相同架构，解码器不同`
82491-BDCRNNRegVariableDecoder-METR_LA-May-26-2023_14-26-34.log 标准差0.01, 0.01，考虑所有参数的KL散度 (worse)
58290-BDCRNNRegVariableDecoder-METR_LA-May-27-2023_19-20-21.log 标准差0.01, 0.01，不考虑建模sigma_0模型的KL散度 (worst)
`大改动，所有的正则化系数不考虑编解码sigma_0的参数，将建模log_sigma_0改为建模sigma_0，再通过类ReLU函数变换取1e-4作为最小值`
23010-BDCRNNRegVariableShared-METR_LA-May-30-2023_12-02-26.log 0.01, 0.01 (worst)
77460-BDCRNNRegVariable-METR_LA-May-30-2023_04-05-16.log 0.01, 0.01 (worse & close)
`正则化系数考虑编解码sigma_0的参数`
48916-BDCRNNRegVariableShared-METR_LA-Jun-01-2023_13-54-44.log 0.01, 0.01 (未收敛)
82428-BDCRNNRegVariable-METR_LA-Jun-01-2023_05-55-53.log 0.01, 0.01 (worst)
`由于效果太差，所以之后正则化系数就不考虑编解码sigma_0参数了`
89044-BDCRNNRegVariableFC-METR_LA-Jun-02-2023_14-03-55.log (worse)
19857-BDCRNNRegVariableSharedFC-METR_LA-Jun-02-2023_22-06-38.log (worse)
485-BDCRNNRegVariableSharedLayer-METR_LA-Jun-04-2023_16-54-41.log (worst)
60820-BDCRNNRegVariableLayer-METR_LA-Jun-04-2023_14-50-11.log (worse)
`所有的训练损失采用mse，修复之前relu函数梯度不为0以及应该加log_sigma0的bug，在config文件中指定正则项是否包括编解码sigma_0的参数以及relu函数的粒度`
57420-BDCRNNConstant-METR_LA-Jun-07-2023_22-12-49.log(worse) MSE训练
6309-BDCRNNConstantShared-METR_LA-Jun-07-2023_22-13-13.log(worse) MSE训练
94702-BDCRNNVariable-METR_LA-Jun-08-2023_13-17-40.log(worst) MSE训练，参数初始0.1
9839-BDCRNNVariableShared-METR_LA-Jun-08-2023_13-17-45.log(worst) MSE训练，参数初始0.1
**26112-BDCRNNConstant-METR_LA-Jun-08-2023_17-08-41.log(close) MAE训练**
**38127-BDCRNNConstantShared-METR_LA-Jun-08-2023_18-58-42.log(close) MAE训练**
48124-BDCRNNLogVariableLayer-METR_LA-Jun-09-2023_20-36-00.log(worst)
60376-BDCRNNLogVariableSharedLayer-METR_LA-Jun-09-2023_21-00-16.log(worst)
79611-BDCRNNLogVariable-METR_LA-Jun-10-2023_13-42-57.log(worst) reg false
93471-BDCRNNLogVariableShared-METR_LA-Jun-10-2023_22-02-54.log(worst worst) reg false
93682-BDCRNNLogVariableFC-METR_LA-Jun-11-2023_07-46-42.log(worst)
73710-BDCRNNLogVariableSharedFC-METR_LA-Jun-11-2023_15-07-30.log(worst)
58469-BDCRNNLogVariable-METR_LA-Jun-12-2023_03-21-07.log(worst) reg true
61569-BDCRNNLogVariableShared-METR_LA-Jun-12-2023_03-22-56.log(worst) reg true
66054-BDCRNNVariableSharedLayer-METR_LA-Jun-15-2023_10-29-36.log(worst) relu_0.0001
12868-BDCRNNVariableSharedLayer-METR_LA-Jun-16-2023_09-25-36.log(worst) Softplus_1
96628-BDCRNNVariableSharedLayer-METR_LA-Jun-17-2023_10-05-00.log(worst) relu_0.001
89632-BDCRNNVariableSharedLayer-METR_LA-Jun-17-2023_10-08-37.log(不收敛) relu_0.0001 sigma_pi=sigma_start=0.001
97415-BDCRNNVariableSharedLayer-METR_LA-Jun-18-2023_02-23-47.log(worst) relu_0.0001 sigma_pi=sigma_start=0.1
20526-BDCRNNVariableShared-METR_LA-Jun-18-2023_10-05-53.log(worst) relu_0.0001, reg false
75133-BDCRNNVariableShared-METR_LA-Jun-18-2023_10-06-48.log(worst) relu_0.0001, reg true
22001-BDCRNNVariableSharedLayer-METR_LA-Jun-19-2023_21-17-25.log(a little worse) no reg
`fix all decoder_sigma function`
7663-BDCRNNVariableDecoderShared-METR_LA-Jun-18-2023_14-43-53.log(worst) reg true
88368-BDCRNNVariableDecoderShared-METR_LA-Jun-19-2023_02-44-51.log(worst) reg false
17503-BDCRNNVariableDecoderShared-METR_LA-Jun-20-2023_08-34-49.log(worst) reg false, sigma_pi=1
9491-BDCRNNVariableDecoderShared-METR_LA-Jun-20-2023_08-35-50.log(worst) reg true, sigma_pi=1
89907-BDCRNNVariableDecoderShared-METR_LA-Jun-20-2023_16-46-40.log(close) no reg
18213-BDCRNNVariableDecoderShared-METR_LA-Jun-21-2023_02-49-55.log(worst) reg false, sigma_pi=100
38251-BDCRNNVariableDecoderShared-METR_LA-Jun-21-2023_14-39-12.log(不收敛) reg false, sigma_pi=10000
97168-BDCRNNVariableDecoderSharedFC-METR_LA-Jun-22-2023_15-13-35.log(worst) 
26230-BDCRNNVariableDecoderSharedFC-METR_LA-Jun-22-2023_07-15-13.log(close) no reg
98537-BDCRNNVariableDecoderShared-METR_LA-Jun-23-2023_10-30-54.log(close) don't care, only try if sigma_pi=10000 \approx no reg without log(sigma_0)
31579-BDCRNNVariableDecoder-METR_LA-Jun-23-2023_10-33-30.log(close) no reg
81186-BDCRNNVariableDecoderFC-METR_LA-Jun-24-2023_15-04-14.log(close) no reg
64676-BDCRNNLogVariableDecoder-METR_LA-Jul-01-2023_10-56-52.log(close) no reg
60973-BDCRNNLogVariableDecoderShared-METR_LA-Jul-01-2023_10-57-59.log(close) no reg
10047-BDCRNNVariableDecoder-METR_LA-Jul-21-2023_11-33-36.log copy 31579
14548-BDCRNNVariableDecoder-METR_LA-Jul-22-2023_12-37-30.log clip=false, reg sigma_0=false (worst)
82888-BDCRNNVariableDecoder-METR_LA-Jul-22-2023_12-39-01.log clip=false, reg sigma_0=true (worst)
22957-BDCRNNVariableDecoder-METR_LA-Jul-22-2023_13-24-25.log clip=true, reg sigma_0=false (close)
***75994-BDCRNNVariableDecoder-METR_LA-Jul-22-2023_21-25-54.log clip=true, reg sigma_0=true (close)***
93092-BDCRNNVariableDecoderShared-METR_LA-Jul-22-2023_20-40-43.log clip=false, reg sigma_0=false(worst)
44957-BDCRNNVariableDecoderShared-METR_LA-Jul-22-2023_20-42-48.log clip=false, reg sigma_0=true(worst)
61630-BDCRNNVariableDecoderShared-METR_LA-Jul-28-2023_16-51-35.log reg sigma_0=false (close)
43664-BDCRNNVariableDecoderShared-METR_LA-Jul-28-2023_16-57-32.log reg sigma_0=true (close)
127-BDCRNNVariableDecoder-METR_LA-Aug-01-2023_10-00-04.log no reg (close)
57986-BDCRNNVariableDecoderShared-METR_LA-Aug-01-2023_10-00-15.log no reg (close)
69162-BDCRNNVariableDecoder-METR_LA-Aug-26-2023_13-31-16.log reg sigma_0=true
42591-DCRNN-METR_LA-Aug-28-2023_14-24-23.log input_window=output_window=24
2727-BDCRNNVariableDecoder-METR_LA-Oct-17-2023_14-39-34.log shuffle=True (worse)
75111-BDCRNNVariableDecoder-METR_LA-Oct-13-2023_06-57-48.log shuffle=True (worse)
9488-DCRNN-METR_LA-Oct-14-2023_04-02-24.log shuffle=True (close)
98946-DCRNN-METR_LA-Oct-16-2023_08-33-51.log shuffle=False (worse)
79081-BDCRNNVariableDecoder-METR_LA-Oct-21-2023_14-52-43.log shuffle=False (不收敛)
26233-BDCRNNVariableDecoder-METR_LA-Oct-28-2023_12-32-24.log shuffle=False 从75994加载参数（不收敛）
44854-BDCRNNVariableDecoderShared-METR_LA-Oct-28-2023_12-56-35.log reg_decoder_sigma_0=False shuffle=False (不收敛)
58823-BDCRNNVariableDecoderShared-METR_LA-Oct-28-2023_12-57-39.log reg_decoder_sigma_0=True shuffle=False (不收敛)
20183-BDCRNNVariableDecoder-METR_LA-Oct-29-2023_08-29-07.log consistent_loss but no abs （不收敛）
46820-BDCRNNConstant-METR_LA-Oct-29-2023_08-04-17.log shuffle=False worst
---------------------------------------------------------------------------------------------------------------------
87635-DCRNN-METR_LA-Oct-29-2023_15-11-18.log shuffle=False (6.56)
30260-DCRNN-METR_LA-Oct-29-2023_15-12-17.log shuffle=True (5.27)
58466-DCRNN-METR_LA-Oct-30-2023_03-56-18.log shuffle=False max_norm=1 (5.98)
18303-DCRNN-METR_LA-Oct-30-2023_03-58-08.log shuffle=False learning_rate=0.005 (5.42)
34773-DCRNN-METR_LA-Oct-30-2023_14-32-19.log shuffle=False learning_rate=0.001 (5.76)
28632-DCRNN-METR_LA-Oct-30-2023_14-33-20.log shuffle=False learning_rate=0.005 max_norm=1 (5.40)
35510-DCRNN-METR_LA-Oct-31-2023_02-58-18.log shuffle=False learning_rate=0.005 lr_decay_ratio=0.5 (5.30)
33514-DCRNN-METR_LA-Jan-06-2024_14-36-41.log shuffle=True delete node 26 (5.22)

52984-BDCRNNVariableDecoderShared-METR_LA-Oct-29-2023_15-15-52.log shuffle=True sigma_pi=sigma_sigma_pi=1, sigma_start=sigma_sigma_start=0.01 (5.44)
56441-BDCRNNVariableDecoderShared-METR_LA-Oct-29-2023_15-17-09.log shuffle=True sigma_pi=sigma_sigma_pi=sigma_start=sigma_sigma_start=1 (30.23)
46635-BDCRNNVariableDecoderShared-METR_LA-Oct-30-2023_14-35-41.log shuffle=True sigma_pi=sigma_sigma_pi=0.1, sigma_start=sigma_sigma_start=0.01 (5.39)
66101-BDCRNNVariableDecoderShared-METR_LA-Oct-30-2023_14-36-51.log shuffle=True sigma_pi=sigma_sigma_pi=1, sigma_start=sigma_sigma_start=0.01 learning_rate=0.005 (5.48)
12243-BDCRNNVariableDecoderShared-METR_LA-Oct-31-2023_13-55-04.log shuffle=True sigma_pi=sigma_sigma_pi=1, sigma_start=sigma_sigma_start=0.01 max_norm=1 (5.57)
17280-BDCRNNVariableDecoderShared-METR_LA-Nov-01-2023_02-42-30.log shuffle=True sigma_pi=sigma_sigma_pi=1, sigma_start=sigma_sigma_start=0.01 learning_rate=0.005 lr_decay_ratio=0.5 (5.31)
86153-BDCRNNVariableDecoderShared-METR_LA-Nov-14-2023_09-02-02.log shuffle=True sigma_pi=sigma_sigma_pi=1, sigma_start=sigma_sigma_start=0.01 learning_rate=0.005 lr_decay_ratio=0.5 max_norm=1 (5.32)

41105-BDCRNNVariableDecoder-METR_LA-Oct-31-2023_08-06-59.log shuffle=True sigma_pi=sigma_sigma_pi=1 sigma_start=sigma_sigma_start=0.01 (5.34)
18135-BDCRNNVariableDecoder-METR_LA-Oct-31-2023_08-08-06.log shuffle=True sigma_pi=sigma_sigma_pi=1 sigma_start=sigma_sigma_start=0.01 learning_rate=0.005 (5.46)
97341-BDCRNNVariableDecoder-METR_LA-Oct-31-2023_13-56-05.log shuffle=True sigma_pi=sigma_sigma_pi=1 sigma_start=sigma_sigma_start=0.01 max_norm=1 (5.36)
51791-BDCRNNVariableDecoder-METR_LA-Nov-01-2023_07-01-08.log shuffle=True sigma_pi=sigma_sigma_pi=0.1 sigma_start=sigma_sigma_start=0.01 (5.35)
77290-BDCRNNVariableDecoder-METR_LA-Nov-01-2023_14-28-56.log shuffle=True sigma_pi=sigma_sigma_pi=1 sigma_start=sigma_sigma_start=0.01 learning_rate=0.005 lr_decay_ratio=0.5 (5.27)
38397-BDCRNNVariableDecoder-METR_LA-Nov-01-2023_14-31-22.log shuffle=False sigma_pi=sigma_sigma_pi=1 sigma_start=sigma_sigma_start=0.01 learning_rate=0.005 lr_decay_ratio=0.5 (failed)
89148-BDCRNNVariableDecoder-METR_LA-Dec-05-2023_14-55-45.log only evaluating test_data based on 77290
87083-BDCRNNVariableDecoder-METR_LA-Dec-06-2023_02-57-05.log only evaluating train_data based on 77290
51598-BDCRNNVariableDecoder-METR_LA-Dec-08-2023_07-29-18.log train based on MSE (5.26, but worse MAE and MAPE)
11529-BDCRNNVariableDecoder-METR_LA-Dec-12-2023_11-42-53.log only testing test_data based on 77290 (0-9) (l27415-l27424, ) (04:25-05:10)
15333-BDCRNNVariableDecoder-METR_LA-Dec-12-2023_11-46-03.log only testing test_data based on 77290 (10-19) (l27425-l27434, ) (05:15-06:00)
98337-BDCRNNVariableDecoder-METR_LA-Dec-28-2023_03-41-21.log only testing test_data based on 77290 (20-25) (l27435-l27440, ) (06:05-06:30)
80588-BDCRNNVariableDecoder-METR_LA-Dec-26-2023_09-50-05.log only testing test_data based on 77290 (26) (l27441, l27444, ) (06:35-06:50)
63269-BDCRNNVariableDecoder-METR_LA-Dec-28-2023_03-44-00.log only testing test_data based on 77290 (27-31) (l27442-l27446, l27444, ) (06:40-07:00)
90374-BDCRNNVariableDecoder-METR_LA-Dec-28-2023_03-49-05.log (32-36) (l27447-l27451) (07:05-07:25)
70110-BDCRNNVariableDecoder-METR_LA-Dec-28-2023_03-50-01.log (37-41) (l27452-l27456) (07:30-07:50)
72784-BDCRNNVariableDecoder-METR_LA-Dec-28-2023_03-51-57.log (42-47) (l27457-l27462) (07:55-08:20)
27302-BDCRNNVariableDecoder-METR_LA-Dec-28-2023_03-52-45.log (48-53) (l27463-l27468) (08:25-08:50)
31246-BDCRNNVariableDecoder-METR_LA-Jan-02-2024_13-41-14.log (152-157) (l27567-l27572) (17:05-17:30)
3555-BDCRNNVariableDecoder-METR_LA-Jan-02-2024_13-42-21.log (158-163) (l27573-l27578) (17:35-18:00)
5112-BDCRNNVariableDecoder-METR_LA-Jan-02-2024_13-43-14.log (164-169) (l27579-l27584) (18:05-18:30)
36239-BDCRNNVariableDecoder-METR_LA-Jan-02-2024_13-43-54.log (170-174) (l27585-l27589) (18:35-18:55)
21982-BDCRNNVariableDecoder-METR_LA-Dec-28-2023_14-46-39.log (175-181) (l27590-l27596) (19:00-19:30)
57707-BDCRNNVariableDecoder-METR_LA-Dec-28-2023_14-47-31.log (182-187) (l27597-l27602) (19:35-20:00)

51066-BDCRNNVariableDecoder-METR_LA-Jan-06-2024_14-35-21.log delete node 26  data (5.32)
86188-BDCRNNVariableDecoder-METR_LA-Jan-06-2024_14-41-15.log delete node 126 data (5.28)
82923-BDCRNNVariableDecoder-METR_LA-Jan-06-2024_14-42-00.log delete node 26, 126 data (5.30)
38420-BDCRNNVariableDecoder-METR_LA-Jan-07-2024_12-48-25.log delete node 26 data and adj (5.31)
92103-BDCRNNVariableDecoder-METR_LA-Jan-07-2024_12-51-00.log delete node 126 data and adj (5.28)
95143-BDCRNNVariableDecoder-METR_LA-Jan-07-2024_12-52-17.log delete node 26, 126 data and adj (5.32)
47951-BDCRNNVariableDecoder-METR_LA-Jan-09-2024_13-01-03.log delete geo_id 26
51977-BDCRNNVariableDecoder-METR_LA-Jan-09-2024_12-59-44.log delete geo_id 9, 131, 146, 158 (181 better)
80131-BDCRNNVariableDecoder-METR_LA-Jan-10-2024_12-42-02.log delete geo_id 131, 146, 158 (181 better)
45316-BDCRNNVariableDecoder-METR_LA-Jan-09-2024_13-05-19.log delete geo_id 190, 205 (149 better)
80663-BDCRNNVariableDecoder-METR_LA-Jan-09-2024_13-06-43.log delete geo_id 61, 72, 190, 205 (149 worse)
85216-BDCRNNVariableDecoder-METR_LA-Jan-10-2024_12-43-05.log delete geo_id 72, 190, 205 (149 worse)
18630-BDCRNNVariableDecoder-METR_LA-Jan-10-2024_12-47-48.log delete geo_id 61, 190, 205 (149 worse)
12487-BDCRNNVariableDecoder-METR_LA-Jan-10-2024_12-46-19.log delete geo_id 61, 72, 190 (149 worse)
64425-BDCRNNVariableDecoder-METR_LA-Jan-14-2024_15-12-05.log delete geo_id 24, 53, 61, 72, 110, 194, 205
69298-BDCRNNVariableDecoder-METR_LA-Jan-15-2024_21-09-52.log delete geo_id 24, 53, 61, 72, 110, 112, 194, 205
06753-BDCRNNVariableDecoder-METR_LA-Jan-16-2024_09-44-43.log delete geo_id 24, 61, 110, 112, 194, 205
46924-BDCRNNVariableDecoder-METR_LA-Jan-16-2024_11-55-17.log delete geo_id 24, 61, 110, 112
15739-BDCRNNVariableDecoder-METR_LA-Jan-16-2024_22-25-01.log delete geo_id 13: 773906, 20: 769403, 25: 716960, 67: 773880, 112: 716968, 115: 717573, 118: 717570, 120: 718089, 199: 774204, 205: 718141
28631-BDCRNNVariableDecoder-METR_LA-Jan-17-2024_01-34-46.log delete geo_id 26: 717804, 110: 717099, 115: 717573, 181: 717513
80630-BDCRNNVariableDecoder-METR_LA-Jan-17-2024_09-42-52.log delete geo_id 2: 767542, 11: 767471, 26: 717804, 108: 767454, 110: 717099, 112: 716968, 115: 717573, 126: 769867, 133: 716953, 173: 717585, 181: 717513
05689-BDCRNNVariableDecoder-METR_LA-Jan-17-2024_13-21-25.log 2: 767542, 11: 767471, 26: 717804, 44: 774012, 61: 773939, 71: 717491, 85: 767621, 108: 767454, 110: 717099, 112: 716968, 115: 717573, 126: 769867, 133: 716953, 171: 717587, 173: 717585, 174: 716939, 181: 717513
66454-BDCRNNVariableDecoder-METR_LA-Jan-17-2024_13-24-11.log 2: 767542, 11: 767471, 26: 717804, 44: 774012, 61: 773939, 71: 717491, 85: 767621, 108: 767454, 109: 761599, 110: 717099, 112: 716968, 115: 717573, 126: 769867, 133: 716953, 140: 718496, 153: 773995, 164: 769345, 171: 717587, 173: 717585, 174: 716939, 181: 717513, 184: 767494, 194: 759772, 199: 774204
03621-BDCRNNVariableDecoder-METR_LA-Jan-17-2024_21-27-02.log 2: 767542, 11: 767471, 26: 717804, 44: 774012, 51: 761604, 61: 773939, 71: 717491, 85: 767621, 108: 767454, 109: 761599, 110: 717099, 112: 716968, 115: 717573, 126: 769867, 133: 716953, 140: 718496, 153: 773995, 164: 769345, 171: 717587, 173: 717585, 174: 716939, 179: 773974, 181: 717513, 184: 767494, 194: 759772, 199: 774204
29726-BDCRNNVariableDecoder-METR_LA-Jan-17-2024_13-30-45.log 2: 767542, 11: 767471, 13: 773906, 26: 717804, 44: 774012, 46: 767609, 49: 716956, 51: 761604, 53: 716554, 61: 773939, 68: 764766, 69: 717497, 71: 717491, 85: 767621, 108: 767454, 109: 761599, 110: 717099, 112: 716968, 115: 717573, 123: 767523, 126: 769867, 133: 716953, 139: 716958, 140: 718496, 153: 773995, 164: 769345, 171: 717587, 173: 717585, 174: 716939, 179: 773974, 181: 717513, 184: 767494, 194: 759772, 199: 774204, 203: 717595
39781-BDCRNNVariableDecoder-METR_LA-Jan-18-2024_01-11-28.log 2: 767542, 11: 767471, 13: 773906, 24: 717578, 26: 717804, 27: 767572, 44: 774012, 46: 767609, 49: 716956, 51: 761604, 53: 716554, 58: 773954, 61: 773939, 62: 774067, 68: 764766, 69: 717497, 71: 717491, 72: 717492, 85: 767621, 103: 717488, 108: 767454, 109: 761599, 110: 717099, 111: 773916, 112: 716968, 115: 717573, 118: 717570, 123: 767523, 126: 769867, 129: 759591, 132: 762329, 133: 716953, 139: 716958, 140: 718496, 143: 718499, 147: 759602, 153: 773995, 164: 769345, 171: 717587, 173: 717585, 174: 716939, 179: 773974, 181: 717513, 184: 767494, 194: 759772, 199: 774204, 203: 717595
54244-BDCRNNVariableDecoder-METR_LA-Jan-18-2024_09-16-46.log 2: 767542, 10: 765604, 11: 767471, 13: 773906, 24: 717578, 26: 717804, 27: 767572, 36: 760987, 44: 774012, 46: 767609, 49: 716956, 51: 761604, 53: 716554, 58: 773954, 61: 773939, 62: 774067, 67: 773880, 68: 764766, 69: 717497, 71: 717491, 72: 717492, 74: 765176, 78: 718064, 85: 767621, 103: 717488, 106: 718072, 108: 767454, 109: 761599, 110: 717099, 111: 773916, 112: 716968, 115: 717573, 118: 717570, 123: 767523, 126: 769867, 129: 759591, 132: 762329, 133: 716953, 135: 767509, 139: 716958, 140: 718496, 143: 718499, 147: 759602, 152: 773996, 153: 773995, 164: 769345, 171: 717587, 173: 717585, 174: 716939, 177: 767554, 179: 773974, 181: 717513, 184: 767494, 194: 759772, 199: 774204, 203: 717595, 205: 718141
23871-BDCRNNVariableDecoder-METR_LA-Jan-18-2024_06-49-25.log 2: 767542, 10: 765604, 11: 767471, 13: 773906, 16: 771667, 24: 717578, 26: 717804, 27: 767572, 36: 760987, 39: 769418, 42: 773927, 44: 774012, 46: 767609, 49: 716956, 51: 761604, 53: 716554, 55: 767470, 58: 773954, 61: 773939, 62: 774067, 65: 767751, 67: 773880, 68: 764766, 69: 717497, 70: 717490, 71: 717491, 72: 717492, 73: 717493, 74: 765176, 78: 718064, 82: 769430, 85: 767621, 95: 718379, 96: 717481, 99: 764120, 103: 717488, 105: 718076, 106: 718072, 108: 767454, 109: 761599, 110: 717099, 111: 773916, 112: 716968, 114: 717576, 115: 717573, 116: 717572, 118: 717570, 121: 769847, 122: 717608, 123: 767523, 126: 769867, 129: 759591, 132: 762329, 133: 716953, 135: 767509, 137: 769358, 139: 716958, 140: 718496, 143: 718499, 147: 759602, 152: 773996, 153: 773995, 164: 769345, 167: 717582, 171: 717587, 173: 717585, 174: 716939, 177: 767554, 178: 773975, 179: 773974, 181: 717513, 184: 767494, 190: 764858, 194: 759772, 199: 774204, 203: 717595, 205: 718141


23467-BDCRNNVariableDecoder-METR_LA-Jan-15-2024_08-28-46.log delete [-3:0] of 181
83111-BDCRNNVariableDecoder-METR_LA-Jan-15-2024_08-29-15.log delete [-6:0] of 181
91137-BDCRNNVariableDecoder-METR_LA-Jan-15-2024_08-33-31.log delete [-12:0] of 181
92360-BDCRNNVariableDecoder-METR_LA-Jan-14-2024_10-11-45.log delete [-12:0] of 181 (delete y wrongly)
87037-BDCRNNVariableDecoder-METR_LA-Jan-15-2024_08-26-05.log delete [-3:0] of 0
22891-BDCRNNVariableDecoder-METR_LA-Jan-15-2024_08-27-06.log delete [-6:0] of 0
85139-BDCRNNVariableDecoder-METR_LA-Jan-15-2024_08-27-41.log delete [-12:0] of 0
20232-BDCRNNVariableDecoder-METR_LA-Jan-14-2024_10-12-43.log delete [-12:0] of 0 (delete y wrongly)

67445-BDCRNNVariableDecoder-METR_LA-Jan-07-2024_15-12-21.log (235-239)
31478-BDCRNNVariableDecoder-METR_LA-Jan-07-2024_15-13-12.log (240-244)
29188-BDCRNNVariableDecoder-METR_LA-Jan-07-2024_15-14-48.log (245-249)
72032-BDCRNNVariableDecoder-METR_LA-Jan-07-2024_15-15-32.log (250-254)

11522-DCRNNDropout-METR_LA-Dec-02-2023_16-27-01.log (5.37)

53537-MTGNN-METR_LA-Nov-30-2023_04-39-24.log (5.13)

18652-STGCN-METR_LA-Dec-18-2023_11-48-53.log (7.65)

67517-STTN-METR_LA-Nov-30-2023_04-34-10.log (7.49)
