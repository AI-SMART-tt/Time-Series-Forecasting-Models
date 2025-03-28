# Time-Series-Forecasting-Models
This work examines key time series forecasting models spanning classical to deep learning methods.

---

✅ 分类：

- 🧠 按照年代和模型类型分类
- 📌 每个模型提供论文链接和 GitHub 仓库（如有）
- 🎯 使用图标美化排版
- 🧩 包含传统模型、机器学习模型、深度学习模型、Transformer模型、混合模型等

---

## 📘 Time Series Forecasting Models – Full Development Timeline

> ⏳ From classical statistical models to cutting-edge deep learning architectures.

---

### 📜 Table of Contents

1. [📊 Traditional Statistical Models](#-traditional-statistical-models)
2. [🧠 Machine Learning-based Models](#-machine-learning-based-models)
3. [🤖 Deep Learning Models](#-deep-learning-models)
4. [🧮 Hybrid & Ensemble Models](#-hybrid--ensemble-models)
5. [🔮 Transformer-based Models](#-transformer-based-models)
6. [📈 Recent Advances & Foundation Models](#-recent-advances--foundation-models)

---

## 📊 Traditional Statistical Models

| Model | Year | Paper | GitHub |
|-------|------|-------|--------|
| 🔢 AR (AutoRegression) | 1920s | N/A | N/A |
| 🧮 MA (Moving Average) | 1920s | N/A | N/A |
| 🔁 ARMA | 1951 | [Wold, 1951](https://projecteuclid.org/euclid.aoms/1177729432) | [Statsmodels](https://github.com/statsmodels/statsmodels) |
| 🔄 ARIMA | 1970 | [Box-Jenkins, 1970](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+Revised+Edition-p-9781118675021) | [pmdarima](https://github.com/alkaline-ml/pmdarima) |
| 🎚️ SARIMA | 1976 | [Box-Jenkins Seasonal](https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+Revised+Edition-p-9781118675021) | [Statsmodels](https://github.com/statsmodels/statsmodels) |
| 📏 Exponential Smoothing (ETS, Holt-Winters) | 1957+ | [Holt’s Linear Trend Model](https://doi.org/10.2307/3001644) | [Statsmodels ETS](https://github.com/statsmodels/statsmodels) |
| ⛅ State Space Models / Kalman Filters | 1960 | [Kalman, 1960](https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1960.tb03958.x) | [pykalman](https://github.com/pykalman/pykalman) |

---

## 🧠 Machine Learning-based Models

| Model | Year | Paper | GitHub |
|-------|------|-------|--------|
| 🌳 Random Forest for TS | ~2000 | [ML for TS - Bontempi, 2012](https://hal.science/hal-00650910) | [sklearn](https://github.com/scikit-learn/scikit-learn) |
| 🎯 Gradient Boosting (XGBoost, LightGBM) | 2016 | [XGBoost](https://arxiv.org/abs/1603.02754) | [XGBoost](https://github.com/dmlc/xgboost) / [LightGBM](https://github.com/microsoft/LightGBM) |
| 🧮 SVR (Support Vector Regression) | 1997 | [SVR](https://www.isical.ac.in/~ecsu/handbook/chapters/svm.pdf) | [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) |
| 🧠 kNN for Time Series | 2002 | [DASARATHY, 2002](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.107.716&rep=rep1&type=pdf) | [tslearn](https://github.com/tslearn-team/tslearn) |

---

## 🤖 Deep Learning Models

| Model | Year | Paper | GitHub |
|-------|------|-------|--------|
| 🧠 LSTM | 1997 / popularized ~2015 | [Hochreiter & Schmidhuber, 1997](https://www.bioinf.jku.at/publications/older/2604.pdf) | [Keras LSTM](https://keras.io/api/layers/recurrent_layers/lstm/) |
| 🔁 GRU | 2014 | [Cho et al., 2014](https://arxiv.org/abs/1406.1078) | [Keras GRU](https://keras.io/api/layers/recurrent_layers/gru/) |
| 📉 DeepAR (Amazon) | 2017 | [DeepAR](https://arxiv.org/abs/1704.04110) | [GluonTS](https://github.com/awslabs/gluon-ts) |
| ⛓️ LSTNet | 2018 | [LSTNet](https://arxiv.org/abs/1703.07015) | [LSTNet](https://github.com/laiguokun/LSTNet) |
| 🔗 TCN (Temporal Convolutional Network) | 2018 | [TCN Paper](https://arxiv.org/abs/1803.01271) | [TCN GitHub](https://github.com/locuslab/TCN) |
| 🧱 N-BEATS | 2020 | [N-BEATS](https://arxiv.org/abs/1905.10437) | [N-BEATS GitHub](https://github.com/ElementAI/N-BEATS) |
| 📦 N-HITS | 2022 | [N-HITS](https://arxiv.org/abs/2201.12886) | [N-HiTS GitHub](https://github.com/Nixtla/neuralforecast) |

---

## 🧮 Hybrid & Ensemble Models

| Model | Year | Paper | GitHub |
|-------|------|-------|--------|
| 🧪 Prophet (Facebook) | 2017 | [Prophet Paper](https://peerj.com/preprints/3190/) | [Prophet GitHub](https://github.com/facebook/prophet) |
| 🧬 Hybrid ARIMA + ML | ~2010+ | [Zhang, 2003](https://www.sciencedirect.com/science/article/abs/pii/S0957417403001135) | Custom Implementations |
| 🧠 AutoML for TS (AutoTS, H2O) | 2020+ | [AutoTS](https://github.com/winedarksea/AutoTS) | [AutoTS GitHub](https://github.com/winedarksea/AutoTS) |

---

## 🔮 Transformer-based Models

| Model | Year | Paper | GitHub |
|-------|------|-------|--------|
| 🔮 Transformer (original) | 2017 | [Vaswani et al.](https://arxiv.org/abs/1706.03762) | [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) |
| 🌀 Informer | 2021 | [Informer](https://arxiv.org/abs/2012.07436) | [Informer GitHub](https://github.com/zhouhaoyi/Informer2020) |
| 🌊 Autoformer | 2021 | [Autoformer](https://arxiv.org/abs/2106.13008) | [Autoformer GitHub](https://github.com/thuml/Autoformer) |
| ⏱️ TimesNet | 2022 | [TimesNet](https://arxiv.org/abs/2210.02186) | [TimesNet GitHub](https://github.com/thuml/TimesNet) |
| 🧠 PatchTST | 2023 | [PatchTST](https://arxiv.org/abs/2211.14730) | [PatchTST GitHub](https://github.com/yuqinie98/PatchTST) |
| 🔁 FEDformer | 2022 | [FEDformer](https://arxiv.org/abs/2201.12740) | [FEDformer GitHub](https://github.com/MAZiqing/FEDformer) |
| 💡 Crossformer | 2023 | [Crossformer](https://arxiv.org/abs/2303.05389) | [Crossformer GitHub](https://github.com/Thinklab-SJTU/Crossformer) |
| 📡 LagLLama | 2023 | [LagLLama](https://arxiv.org/abs/2310.06625) | [Lag-Llama GitHub](https://github.com/microsoft/Lag-Llama
