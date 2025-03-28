# Time-Series-Forecasting-Models
This work examines key time series forecasting models spanning classical to deep learning methods.

---

# 📘 时间序列预测模型发展完整脉络

> ⏳ 从经典统计模型到最前沿的深度学习架构的全面发展历程

---

## 📜 目录

1. [📊 传统统计模型](#-传统统计模型)
2. [🧠 基于机器学习的模型](#-基于机器学习的模型)
3. [🤖 深度学习模型](#-深度学习模型)
4. [🧮 混合与集成模型](#-混合与集成模型)
5. [🔮 基于Transformer的模型](#-基于transformer的模型)
6. [📈 最新进展与基础模型](#-最新进展与基础模型)
7. [🌐 多变量与概率预测专门模型](#-多变量与概率预测专门模型)
8. [💼 特定领域的时间序列模型](#-特定领域的时间序列模型)

---

## 📊 传统统计模型

| 模型 | 年份 | 论文 | GitHub |
|-------|------|-------|--------|
| 🔢 AR (自回归模型) | 1920s | N/A | N/A |
| 🧮 MA (移动平均模型) | 1920s | N/A | N/A |
| 🔁 ARMA (自回归移动平均模型) | 1951 | [Wold, 1951](https://projecteuclid.org/euclid.aoms/1177729432) | [Statsmodels](https://github.com/statsmodels/statsmodels) |
| 🔄 ARIMA (差分自回归移动平均模型) | 1970 | [Box-Jenkins, 1970](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+Revised+Edition-p-9781118675021) | [pmdarima](https://github.com/alkaline-ml/pmdarima) |
| 🎚️ SARIMA (季节性ARIMA) | 1976 | [Box-Jenkins Seasonal](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+Revised+Edition-p-9781118675021) | [Statsmodels](https://github.com/statsmodels/statsmodels) |
| 🌊 VARIMA (向量ARIMA) | 1980s | [Tiao & Box, 1981](https://www.jstor.org/stable/2287617) | [Statsmodels VAR](https://github.com/statsmodels/statsmodels) |
| 📏 指数平滑法 (ETS, Holt-Winters) | 1957+ | [Holt's Linear Trend Model](https://doi.org/10.2307/3001644) | [Statsmodels ETS](https://github.com/statsmodels/statsmodels) |
| 🔥 GARCH (广义自回归条件异方差) | 1986 | [Bollerslev, 1986](https://doi.org/10.1016/0304-4076(86)90063-1) | [arch](https://github.com/bashtage/arch) |
| ⛅ 状态空间模型/卡尔曼滤波 | 1960 | [Kalman, 1960](https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1960.tb03958.x) | [pykalman](https://github.com/pykalman/pykalman) |
| 📈 STL分解 (Seasonal-Trend-Loess) | 1990 | [Cleveland et al., 1990](https://www.jstor.org/stable/1403114) | [Statsmodels STL](https://github.com/statsmodels/statsmodels) |
| 🔄 ARIMAX (带外生变量的ARIMA) | 1970s | [Box-Jenkins扩展](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+Revised+Edition-p-9781118675021) | [Statsmodels SARIMAX](https://github.com/statsmodels/statsmodels) |
| 🌐 VAR (向量自回归) | 1980 | [Sims, 1980](https://www.jstor.org/stable/1912017) | [Statsmodels VAR](https://github.com/statsmodels/statsmodels) |
| 🔍 UCM (不可观测组件模型) | 1982 | [Harvey, 1989](https://doi.org/10.1017/CBO9781107049994) | [Statsmodels UnobservedComponents](https://github.com/statsmodels/statsmodels) |
| 📉 VECM (向量误差修正模型) | 1987 | [Engle & Granger, 1987](https://www.jstor.org/stable/1913236) | [Statsmodels VECM](https://github.com/statsmodels/statsmodels) |
| 🔗 TBATS (三角基础、Box-Cox变换、ARMA误差、趋势和季节性) | 2011 | [De Livera et al., 2011](https://doi.org/10.1080/01621459.2011.604693) | [forecast](https://github.com/robjhyndman/forecast) |

---

## 🧠 基于机器学习的模型

| 模型 | 年份 | 论文 | GitHub |
|-------|------|-------|--------|
| 🌳 针对时间序列的随机森林 | ~2000 | [Bontempi, 2012](https://hal.science/hal-00650910) | [sklearn](https://github.com/scikit-learn/scikit-learn) |
| 🎯 梯度提升 (XGBoost, LightGBM, CatBoost) | 2016+ | [XGBoost](https://arxiv.org/abs/1603.02754), [LightGBM](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), [CatBoost](https://arxiv.org/abs/1706.09516) | [XGBoost](https://github.com/dmlc/xgboost), [LightGBM](https://github.com/microsoft/LightGBM), [CatBoost](https://github.com/catboost/catboost) |
| 🧮 SVR (支持向量回归) | 1997 | [SVR](https://www.isical.ac.in/~ecsu/handbook/chapters/svm.pdf) | [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) |
| 🧠 时间序列的kNN | 2002 | [DASARATHY, 2002](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.107.716&rep=rep1&type=pdf) | [tslearn](https://github.com/tslearn-team/tslearn) |
| 🔮 GP (高斯过程) | 2006 | [Rasmussen & Williams, 2006](http://www.gaussianprocess.org/gpml/) | [GPy](https://github.com/SheffieldML/GPy) |
| 🌊 TBATS | 2011 | [De Livera et al., 2011](https://www.tandfonline.com/doi/abs/10.1080/01621459.2011.604693) | [tbats](https://github.com/intive-DataScience/tbats) |
| 🔄 Dynamic Time Warping | 1990s | [Berndt & Clifford, 1994](https://www.aaai.org/Papers/Workshops/1994/WS-94-03/WS94-03-031.pdf) | [dtw-python](https://github.com/pierre-rouanet/dtw) |
| 🎭 ARIMA-SVM混合 | 2003 | [Zhang, 2003](https://www.sciencedirect.com/science/article/abs/pii/S0957417403001135) | 自定义实现 |
| 🧪 LGBM-based Temporal Fusion | 2020 | [Temporal Fusion](https://arxiv.org/abs/1912.09363) | [PyTorch Forecasting](https://github.com/jdb78/pytorch-forecasting) |
| 🔍 Isolation Forest异常检测 | 2008 | [Liu et al., 2008](https://ieeexplore.ieee.org/abstract/document/4781136) | [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) |
| 🔄 ROCKET (随机卷积核变换) | 2020 | [Dempster et al., 2020](https://arxiv.org/abs/1910.13051) | [ROCKET](https://github.com/angus924/rocket) |
| 📊 Matrix Profile | 2016 | [Yeh et al., 2016](https://ieeexplore.ieee.org/document/7837898) | [STUMPY](https://github.com/TDAmeritrade/stumpy) |

---

## 🤖 深度学习模型

| 模型 | 年份 | 论文 | GitHub |
|-------|------|-------|--------|
| 🧠 LSTM (长短期记忆网络) | 1997/~2015普及 | [Hochreiter & Schmidhuber, 1997](https://www.bioinf.jku.at/publications/older/2604.pdf) | [Keras LSTM](https://keras.io/api/layers/recurrent_layers/lstm/) |
| 🔁 GRU (门控循环单元) | 2014 | [Cho et al., 2014](https://arxiv.org/abs/1406.1078) | [Keras GRU](https://keras.io/api/layers/recurrent_layers/gru/) |
| 📉 DeepAR (亚马逊) | 2017 | [DeepAR](https://arxiv.org/abs/1704.04110) | [GluonTS](https://github.com/awslabs/gluon-ts) |
| ⛓️ LSTNet | 2018 | [LSTNet](https://arxiv.org/abs/1703.07015) | [LSTNet](https://github.com/laiguokun/LSTNet) |
| 🔗 TCN (时间卷积网络) | 2018 | [TCN Paper](https://arxiv.org/abs/1803.01271) | [TCN GitHub](https://github.com/locuslab/TCN) |
| 🎯 DeepTCN | 2019 | [DeepTCN](https://arxiv.org/abs/1906.01715) | [DeepTCN](https://github.com/diyumeng1012/DeepTCN) |
| 🧱 N-BEATS | 2020 | [N-BEATS](https://arxiv.org/abs/1905.10437) | [N-BEATS GitHub](https://github.com/ElementAI/N-BEATS) |
| 📦 N-HITS | 2022 | [N-HITS](https://arxiv.org/abs/2201.12886) | [N-HiTS GitHub](https://github.com/Nixtla/neuralforecast) |
| 🌊 WaveNet | 2016 | [WaveNet](https://arxiv.org/abs/1609.03499) | [WaveNet](https://github.com/ibab/tensorflow-wavenet) |
| 🔮 DeepState | 2018 | [DeepState](https://arxiv.org/abs/1905.12374) | [GluonTS](https://github.com/awslabs/gluon-ts) |
| 🧠 TFT (Temporal Fusion Transformer) | 2020 | [TFT](https://arxiv.org/abs/1912.09363) | [TFT GitHub](https://github.com/google-research/google-research/tree/master/tft) |
| 🎭 MQRNN (多量化RNN) | 2017 | [MQRNN](https://arxiv.org/abs/1711.11053) | [GluonTS](https://github.com/awslabs/gluon-ts) |
| 🎭 MQCNN (多量化CNN) | 2018 | [DeepTCN](https://arxiv.org/abs/1711.11053) | [GluonTS](https://github.com/awslabs/gluon-ts) |
| 🔄 序列到序列RNN | 2015 | [Seq2Seq](https://arxiv.org/abs/1506.02216) | [PyTorch](https://github.com/pytorch/pytorch) |
| 🔍 Deep Factor Models | 2019 | [Deep Factor](https://arxiv.org/abs/1905.12417) | [GluonTS](https://github.com/awslabs/gluon-ts) |
| 🧠 Neural ODE | 2018 | [Neural ODE](https://arxiv.org/abs/1806.07366) | [torchdyn](https://github.com/DiffEqML/torchdyn) |
| 📡 NeuralProphet | 2021 | [NeuralProphet](https://arxiv.org/abs/2111.15397) | [NeuralProphet](https://github.com/ourownstory/neural_prophet) |
| 🌀 NHITS | 2022 | [NHITS](https://arxiv.org/abs/2201.12886) | [NHITS](https://github.com/Nixtla/neuralforecast) |

---

## 🧮 混合与集成模型

| 模型 | 年份 | 论文 | GitHub |
|-------|------|-------|--------|
| 🧪 Prophet (Facebook) | 2017 | [Prophet Paper](https://peerj.com/preprints/3190/) | [Prophet GitHub](https://github.com/facebook/prophet) |
| 🧬 混合 ARIMA + ML | ~2010+ | [Zhang, 2003](https://www.sciencedirect.com/science/article/abs/pii/S0957417403001135) | 自定义实现 |
| 🧠 用于时间序列的AutoML (AutoTS, H2O) | 2020+ | [AutoTS](https://github.com/winedarksea/AutoTS) | [AutoTS GitHub](https://github.com/winedarksea/AutoTS) |
| 📈 TBATS + Neural Networks | 2021 | [N-BEATS + TBATS](https://ieeexplore.ieee.org/document/9412562) | 自定义实现 |
| 🔮 ETS + ARIMA (Theta方法) | 2000 | [Theta Method](https://www.sciencedirect.com/science/article/abs/pii/S0169207000000662) | [forecast](https://github.com/robjhyndman/forecast) |
| 🤖 统计 + 深度学习集成 | 2020+ | [ES-RNN](https://www.sciencedirect.com/science/article/pii/S0169207019301128) | [M4-methods](https://github.com/Mcompetitions/M4-methods) |
| 🌀 DETS (动态组合专家) | 2022 | [DETS](https://arxiv.org/abs/2211.06350) | [DETS](https://github.com/DAMO-DI-ML/DETS) |
| 📊 AutoGluon-TimeSeries | 2022 | [AutoGluon-TimeSeries](https://arxiv.org/abs/2308.02022) | [AutoGluon](https://github.com/autogluon/autogluon) |
| 🧩 ESRNN (ES-RNN) | 2019 | [Smyl, 2019](https://www.sciencedirect.com/science/article/pii/S0169207019301128) | [M4 Competition Winner](https://github.com/Mcompetitions/M4-methods/tree/master/118%20-%20slaweks17) |
| 🧮 KDD'21时间序列集成方法 | 2021 | [KDD Cup 2021 Winner](https://arxiv.org/abs/2108.10196) | 自定义实现 |
| 📚 Nixtla METS | 2022 | [METS](https://arxiv.org/abs/2110.13389) | [METS](https://github.com/Nixtla/METS) |

---

## 🔮 基于Transformer的模型

| 模型 | 年份 | 论文 | GitHub |
|-------|------|-------|--------|
| 🔮 Transformer (原始) | 2017 | [Vaswani et al.](https://arxiv.org/abs/1706.03762) | [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) |
| 🌀 Informer | 2021 | [Informer](https://arxiv.org/abs/2012.07436) | [Informer GitHub](https://github.com/zhouhaoyi/Informer2020) |
| 🌊 Autoformer | 2021 | [Autoformer](https://arxiv.org/abs/2106.13008) | [Autoformer GitHub](https://github.com/thuml/Autoformer) |
| ⏱️ TimesNet | 2022 | [TimesNet](https://arxiv.org/abs/2210.02186) | [TimesNet GitHub](https://github.com/thuml/TimesNet) |
| 🧠 PatchTST | 2023 | [PatchTST](https://arxiv.org/abs/2211.14730) | [PatchTST GitHub](https://github.com/yuqinie98/PatchTST) |
| 🔁 FEDformer | 2022 | [FEDformer](https://arxiv.org/abs/2201.12740) | [FEDformer GitHub](https://github.com/MAZiqing/FEDformer) |
| 💡 Crossformer | 2023 | [Crossformer](https://arxiv.org/abs/2303.05389) | [Crossformer GitHub](https://github.com/Thinklab-SJTU/Crossformer) |
| 📡 LagLLama | 2023 | [LagLLama](https://arxiv.org/abs/2310.06625) | [Lag-Llama GitHub](https://github.com/microsoft/Lag-Llama) |
| 🧠 TiDE | 2022 | [TiDE](https://arxiv.org/abs/2304.08424) | [TiDE GitHub](https://github.com/google-research/google-research/tree/master/tide) |
| 🌐 Pyraformer | 2022 | [Pyraformer](https://arxiv.org/abs/2109.12218) | [Pyraformer GitHub](https://github.com/alipay/Pyraformer) |
| 🔄 iTransformer | 2023 | [iTransformer](https://arxiv.org/abs/2310.06625) | [iTransformer GitHub](https://github.com/thuml/Time-Series-Library) |
| 🪄 DLinear | 2023 | [DLinear](https://arxiv.org/abs/2205.13504) | [DLinear GitHub](https://github.com/cure-lab/LTSF-Linear) |
| 📊 Stationary | 2023 | [Non-stationary Transformers](https://arxiv.org/abs/2205.14415) | [Non-stationary](https://github.com/thuml/Nonstationary_Transformers) |
| 🧬 Chronos | 2023 | [Chronos](https://arxiv.org/abs/2306.12021) | [Chronos GitHub](https://github.com/thuml/Chronos) |
| 🎯 TACTiS | 2022 | [TACTiS](https://arxiv.org/abs/2202.07125) | [TACTiS GitHub](https://github.com/amazon-science/chronos-forecasting) |
| 🔍 MICN | 2023 | [MICN](https://arxiv.org/abs/2303.14186) | [MICN GitHub](https://github.com/wzhwzhwzh0921/MICN) |
| 🔮 TimeGPT | 2023 | [TimeGPT](https://arxiv.org/abs/2310.03589) | [Nixtla TimesGPT](https://github.com/Nixtla/nixtla) |
| 🚀 TSMixer | 2023 | [TSMixer](https://arxiv.org/abs/2303.06053) | [TSMixer GitHub](https://github.com/google-research/google-research/tree/master/tsmixer) |

---

## 📈 最新进展与基础模型

| 模型 | 年份 | 论文 | GitHub |
|-------|------|-------|--------|
| 🧠 TimesFM (基础模型) | 2023 | [TimesFM](https://arxiv.org/abs/2310.05918) | [TimesFM](https://github.com/JunweiLiang/TimesFM) |
| 🌐 TimeGPT-1 | 2023 | [TimeGPT-1](https://arxiv.org/abs/2310.03589) | [Nixtla TimeGPT](https://github.com/Nixtla/nixtla) |
| 🚀 GPT-4TS | 2023 | [GPT-4TS](https://arxiv.org/abs/2308.11176) | 非公开 |
| 🧮 MOMENT | 2023 | [MOMENT](https://arxiv.org/abs/2312.04557) | [MOMENT](https://github.com/mbzuai-oryx/MOMENT) |
| 🔮 NeuralForecast | 2022+ | [NeuralForecast](https://arxiv.org/abs/2104.10264) | [NeuralForecast](https://github.com/Nixtla/neuralforecast) |
| 🧬 TimesNet | 2023 | [TimesNet](https://arxiv.org/abs/2210.02186) | [TimesNet](https://github.com/thuml/TimesNet) |
| 📡 Tensorflow Probability时间序列 | 2022+ | [TFP Structural Time Series](https://www.tensorflow.org/probability/examples/Structural_Time_Series_Modeling_Case_Studies_Atmospheric_CO2_and_Electricity_Demand) | [TensorFlow Probability](https://github.com/tensorflow/probability) |
| 🌀 Chronos (Uber) | 2023 | [Chronos](https://eng.uber.com/chronos-and-grafana-cloud/) | 内部实现 |
| 💫 Merlion | 2022 | [Merlion](https://arxiv.org/abs/2109.09265) | [Merlion](https://github.com/salesforce/Merlion) |
| 🛠️ Kats (Facebook) | 2021 | [Kats](https://arxiv.org/abs/2105.06821) | [Kats](https://github.com/facebookresearch/Kats) |
| 🌊 TS2Vec | 2022 | [TS2Vec](https://arxiv.org/abs/2202.05205) | [TS2Vec](https://github.com/yuezhihan/ts2vec) |
| 🧠 Theta模型扩展 | 2023 | [MSTL](https://arxiv.org/abs/2107.13586) | [Theta Models](https://github.com/Mcompetitions/M4-methods) |

---

## 🌐 多变量与概率预测专门模型

| 模型 | 年份 | 论文 | GitHub |
|-------|------|-------|--------|
| 🧠 DeepVAR | 2019 | [DeepVAR](https://arxiv.org/abs/1910.03002) | [GluonTS](https://github.com/awslabs/gluon-ts) |
| 🔮 GPVAR | 2021 | [GPVAR](https://arxiv.org/abs/2104.04999) | [GluonTS](https://github.com/awslabs/gluon-ts) |
| 🎲 Quantile回归森林 | 2006 | [Quantile Regression Forests](https://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf) | [scikit-garden](https://github.com/scikit-garden/scikit-garden) |
| 💫 LightGBMQuantile | 2017+ | [LightGBM官方文档](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html) | [LightGBM](https://github.com/microsoft/LightGBM) |
| 🌊 NGBoost | 2020 | [NGBoost](https://arxiv.org/abs/1910.03225) | [NGBoost](https://github.com/stanfordmlgroup/ngboost) |
| 🤖 Deep Quantile Regression | 2019+ | [DeepAR变体](https://arxiv.org/abs/1704.04110) | [GluonTS](https://github.com/awslabs/gluon-ts) |
| 📊 BNP (贝叶斯非参数模型) | 2018 | [BNP-Seq](https://arxiv.org/abs/1910.03002) | [GluonTS](https://github.com/awslabs/gluon-ts) |
| 🎯 Conformal Prediction | 2021+ | [Time Series CP](https://arxiv.org/abs/2010.09107) | [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) |
| 🔄 SDE (随机微分方程) | 2022 | [Latent SDE](https://arxiv.org/abs/2001.01328) | [torchsde](https://github.com/google-research/torchsde) |
| 🧩 Flow-based Models | 2020+ | [IAF for Time Series](https://arxiv.org/abs/2007.06468) | [TorchDyn](https://github.com/diffeqml/torchdyn) |

---

## 💼 特定领域的时间序列模型

| 模型 | 年份 | 论文 | GitHub |
|-------|------|-------|--------|
| 🌡️ 气象预测ECMWF | 2020+ | [ECMWF IFS](https://www.ecmwf.int/en/research/modelling-and-prediction) | [ECMWF API](https://github.com/ecmwf/ecmwf-opendata) |
| 💰 金融市场预测 | 2020+ | [DeepLOB](https://arxiv.org/abs/1808.03711) | [DeepLOB](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books) |
| ⚡ 能源负载预测 | 2019+ | [DeepAR Energy](https://www.sciencedirect.com/science/article/pii/S0360544219311363) | [GluonTS](https://github.com/awslabs/gluon-ts) |
| 🚗 交通流量预测 | 2018+ | [STGCN](https://arxiv.org/abs/1709.04875) | [STGCN](https://github.com/VeritasYin/STGCN_IJCAI-18) |
| 🩺 医疗健康时间序列 | 2020+ | [Dynamic-Deep-Learning-for-ECG](https://www.nature.com/articles/s41591-018-0268-3) | [ECG-DL](https://github.com/awni/ecg) |
| 🔍 异常检测 | 2021+ | [TimesNet for Anomaly](https://arxiv.org/abs/2210.02186) | [TimesNet](https://github.com/thuml/TimesNet) |
| 📱 物联网预测 | 2022+ | [IoTStream](https://dl.acm.org/doi/abs/10.1145/3589335) | [IoTStream](https://github.com/imperial-qore/IotSim) |
| 🏭 工业过程预测 | 2020+ | [IndRNN](https://ieeexplore.ieee.org/document/8996530) | [IndRNN](https://github.com/Sunnydreamrain/IndRNN_Pytorch) |
