# Time-Series-Forecasting-Models
This work examines key time series forecasting models spanning classical to deep learning methods.


# ⏳ 时间序列预测模型发展图谱（1880s–2025s）

> 本图谱按 **模型范式分层**，结合时间序列建模的技术演化路径，展示了从 19 世纪末至今的主要模型、代表论文与开源实现。

---

## 🧱 1. 传统统计建模阶段（1880s – 1980s）

| 分类         | 代表模型                 | 代表论文 / 作者                     | GitHub链接 / 工具 |
|--------------|--------------------------|--------------------------------------|--------------------|
| 周期分析     | 周期图、谱分析、相关分析 | Yule (1927), Slutzky (1937)         | N/A                |
| 分解方法     | X-11、STL、Census        | Cleveland et al. (1990)             | `statsmodels`      |
| AR/MA模型    | AR, MA, ARMA             | Wold (1938), Box & Jenkins (1970)   | `statsmodels`      |
| ARIMA系列    | ARIMA, SARIMA, ARIMAX    | Box & Jenkins (1970)                | [`pmdarima`](https://github.com/alkaline-ml/pmdarima) |
| 状态空间模型 | Kalman Filter, DLM       | Kalman (1960), Durbin & Koopman     | `pydlm`, `bsts`    |
| 指数平滑     | Holt-Winters, ETS        | Holt (1957), Winters (1960)         | `statsmodels`, `prophet` |

---

## 🧠 2. 经典机器学习阶段（1990s – 2015）

| 分类         | 代表模型                 | 代表论文 / 作者                     | GitHub链接 / 工具 |
|--------------|--------------------------|--------------------------------------|--------------------|
| 支持向量机   | SVR, SVM                 | Vapnik (1995)                        | `sklearn`          |
| 相似性方法   | KNN                      | -                                    | `sklearn`          |
| 决策树       | CART, MARS               | Breiman et al. (1984), Friedman      | `sklearn`          |
| 集成方法     | Bagging, AdaBoost        | Breiman (1996), Freund & Schapire    | `sklearn`          |
| Boosting     | XGBoost, LightGBM        | Chen & Guestrin (2016), Ke et al.    | [`xgboost`](https://github.com/dmlc/xgboost), [`lightgbm`](https://github.com/microsoft/LightGBM) |
| 多项式回归   | MARS                     | Friedman (1991)                      | `py-earth`         |

---

## 🔥 3. 深度学习阶段（2015 – 2019）

| 分类         | 代表模型                 | 代表论文 / 作者                     | GitHub链接 / 工具 |
|--------------|--------------------------|--------------------------------------|--------------------|
| 循环神经网络 | RNN, LSTM, GRU           | Hochreiter & Schmidhuber (1997)     | `keras`, `pytorch` |
| 卷积网络     | TCN, 1D-CNN              | Bai et al. (2018)                   | `pytorch`, `keras` |
| 编码解码结构 | Seq2Seq, Attention RNN   | Sutskever et al. (2014)             | `OpenNMT`, `Fairseq` |
| 多步预测     | DeepAR, DeepVAR          | Salinas et al. (2019)               | [`gluon-ts`](https://github.com/awslabs/gluon-ts) |
| 可解释建模   | N-BEATS, InterpretableML | Oreshkin et al. (2020)              | [`n-beats`](https://github.com/philipperemy/n-beats) |
| 状态空间DL   | DeepState                | Rangapuram et al. (2018)            | `gluon-ts`         |

---

## ⚡ 4. Transformer & 结构创新阶段（2019 – 2022）

| 分类         | 代表模型                 | 代表论文 / 作者                     | GitHub链接 |
|--------------|--------------------------|--------------------------------------|------------|
| 注意力机制   | Transformer              | Vaswani et al. (2017)               | `huggingface` |
| 高效长序建模 | Informer                 | Zhou et al. (2021, AAAI)            | [Informer2020](https://github.com/zhouhaoyi/Informer2020) |
| 趋势建模     | Autoformer               | Wu et al. (2021, NeurIPS)           | [Autoformer](https://github.com/thuml/Autoformer) |
| 频域建模     | FEDformer                | Zhou et al. (2022)                  | [FEDformer](https://github.com/MAZiqing/FEDformer) |
| 分块建模     | PatchTST                 | Nie et al. (2023)                   | [PatchTST](https://github.com/yuqinie98/PatchTST) |
| 多尺度建模   | TimesNet                 | Wu et al. (2023, ICLR)              | [TimesNet](https://github.com/thuml/TimesNet) |
| 混合结构     | TFT                     | Lim et al. (2021, JMLR)             | [Temporal Fusion Transformer](https://github.com/jdb78/pytorch-forecasting) |
| 稀疏建模     | LogTrans, Reformer       | Li et al., Kitaev et al.            | 多实现      |

---

## 🔍 5. 自监督 & 图结构建模阶段（2021 – 2023）

| 分类         | 代表模型                 | 代表论文 / 作者                     | GitHub链接 |
|--------------|--------------------------|--------------------------------------|------------|
| 表征学习     | TS2Vec                   | Wu et al. (2021)                    | [TS2Vec](https://github.com/yuezhihan/ts2vec) |
| 对比学习     | CoST, TNC, CPC-TSC       | Franceschi et al. (2020), Wang et al.| 多仓库     |
| 图神经网络   | DCRNN, STGNN, GraphWaveNet | Li et al. (2018), Wu et al.         | 多实现     |
| 缺失值建模   | CSDI                     | Tashiro et al.                      | [CSDI](https://github.com/HySonLab/csdi-pytorch) |

---

## 🚀 6. 大模型 & 前沿模型阶段（2023 – 2025）

| 分类         | 代表模型                 | 代表论文 / 作者                     | GitHub链接 |
|--------------|--------------------------|--------------------------------------|------------|
| LLM for TS   | TimeGPT                  | Nixtla (2023)                        | [TimeGPT](https://github.com/Nixtla/TimeGPT) |
| 微调大模型   | Chronos                  | Microsoft (2023)                     | [Chronos](https://github.com/microsoft/chronos) |
| 多模态时序   | MM-TS, CM-TS             | 结合文本/图像/传感器                | N/A        |
| 自适应建模   | Nonstationary Transf.    | Wu et al. (2023)                     | [NonstationaryTransformer](https://github.com/thuml/Nonstationary_Transformers) |
| Koopman建模  | Koopformer               | Gao et al. (2023)                    | [Koopformer](https://github.com/LongxingTan/Koopformer) |
| 混合专家模型 | MoE for Time Series      | Google Brain (2023)                  | 多实现     |
| 记忆增强     | MemoryTS, RETAIN         | Choi et al. (2016)                   | 多实现     |
| 时间因子LLM  | Lag-Llama                | Meta (2024)                          | 待开源     |

---

## 🛠 7. 工业级平台工具推荐

| 工具 / 框架       | 开发者         | 链接 |
|--------------------|----------------|------|
| Facebook Prophet   | Meta           | [Prophet](https://github.com/facebook/prophet) |
| NeuralProp
