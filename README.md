# Time-Series-Forecasting-Models
This work examines key time series forecasting models spanning classical to deep learning methods.


1880s ---------- 1920s ---------- 1950s ---------- 1970s ---------- 1980s ---------- 1990s ---------- 2000s ---------- 2010s ---------- 2015+ ---------- 2020+
  |                |                |                |                |                |                |                |                |                |
早期统计方法      古典分解方法        ARIMA系列         状态空间模型        非线性模型          机器学习方法        集成方法            深度学习            注意力机制          混合与前沿模型
  |                |                |                |                |                |                |                |                |                |
周期图分析        季节性分解         Box-Jenkins      Kalman滤波        GARCH族模型        神经网络            Bagging           RNN/LSTM/GRU      Transformer       多模态融合
(Schuster,1898)   (加法/乘法)       方法(1970)       (1960)           (Bollerslev,1986) (Lapedes,1987)    (Breiman,1996)   (Hochreiter,1997) (Vaswani,2017)   (Multimodal)

周期回归          STL分解           AR/MA/ARMA       指数平滑族         TAR/SETAR模型      SVR               随机森林          CNN用于时序       Informer          神经ODE
(Fourier,1822)    (Cleveland,1990)  (Yule,1927)      (Brown,1956)      (Tong,1978)       (Drucker,1997)    (Breiman,2001)   (Binkowski,2018)  (Zhou,2021)       (Chen,2018)

古典谐波分析      X-11/X-12        ARIMA            Holt-Winters     马尔可夫切换       SVM               Boosting方法      TCN               Autoformer        Neural Process
(Stokes,1879)     (Shiskin,1967)    (Box-Jenkins)    (Winters,1960)    (Hamilton,1989)   (Vapnik,1995)     (Freund,1997)    (Bai,2018)        (Wu,2021)         (Garnelo,2018)
                                                                    
相关分析          Census方法       SARIMA           状态空间ETS        STAR模型          k近邻              AdaBoost          Seq2Seq          LogTrans         混合物专家模型
(Yule,1921)       (US Bureau)       (季节性ARIMA)    (Hyndman,2008)    (Chan,1993)       (Yakowitz,1987)   (Freund,1995)    (Sutskever,2014)  (Li,2019)         (MoE)

谱分析            X-13ARIMA-SEATS  ARIMAX/SARIMAX    UCM              平滑转换AR         决策树             Gradient          DeepAR           ETSformer        自适应学习
(Wiener,1930)     (Findley,1998)   (带外生变量)      (不可观测成分)    (Teräsvirta,1994) (Breiman,1984)    Boosting         (Salinas,2020)   (Woo,2022)        (Online Learning)
                                                                    
周期性自回归      季节性调整        多变量ARIMA       动态线性模型       双线性模型         回归树             Stacking          Wavenet          TFT              TimeGPT
(Yule-Walker)     (X-12方法)       (VARIMA)         (West,1997)      (Granger,1978)    (CART)            (Wolpert,1992)    (Oord,2016)      (Lim,2020)        (Garza,2023)
                                                                    
最小二乘估计      SEATS分解        向量自回归        贝叶斯结构        ARCH模型          MARS              XGBoost           N-BEATS           Reformer          时空融合模型
(Gauss,1795)      (Gómez,1996)     (VAR/VECM)       (Harrison,1999)   (Engle,1982)      (Friedman,1991)   (Chen,2016)      (Oreshkin,2020)   (Kitaev,2020)     (ST-Transformer)
                                                                    
谐波分析          结构化分解        ARFIMA           结构化模型         长记忆模型         投影寻踪回归       LightGBM          DeepVAR           Pyraformer       Patchtst
(谐波回归)        (基础模型)       (分数差分)        (Harvey,1989)     (Hosking,1981)    (Friedman,1981)   (Ke,2017)        (Salinas,2019)    (Liu,2021)        (Nie,2022)
                                                                    
移动平均          分解预测法        介入分析          空间状态表示       分数差分模型       广义可加模型       CatBoost          多量化RNN         Autoformer       Timeception
(滑动平均)        (综合预测)       (Box,1975)       (SSR)            (Granger,1980)    (GAM)             (Prokhorenkova)   (MQRNN)          (Wu,2021)         (Hussain,2019)
                                                                    
经验模态分解      霍尔特线性趋势    季节性调整法      指数加权移动平均   EGARCH模型         Lasso回归          GradientBoost     TFT               FEDformer        Timesnet
(EMD方法)         (Holt,1957)      (X-11ARIMA)      (EWMA)           (Nelson,1991)     (Tibshirani,1996) (GBM变种)        (Lim,2021)        (Zhou,2022)       (Wu,2023)
                                                                    
非参数方法        快速傅里叶变换    ARIMAX           状态依赖模型       随机系数模型       决策树集成         XGBoost变种       DeepState         Crossformer      DLinear/NLinear
(Kernel方法)      (FFT分析)        (回归ARIMA)      (状态空间)        (随机参数)        (Ensemble Trees)  (正则化变种)      (Rangapuram,2018) (Zhang,2022)      (Zeng,2023)
                                                                    
Slutsky效应       小波分析          VARMA            Innovations模型   门限自回归         神经网络集成       Quantile Forest   MQRNN            NonstatTransf    FiLM
(Slutsky,1927)    (Wavelet)        (向量ARMA)       (De Jong,1991)   (Threshold AR)    (NN Ensemble)     (分位数森林)      (分位数RNN)       (Liu,2022)        (Zhou,2022)
                                                                    
Wold分解          时频分析          协整分析         非高斯状态空间     非线性状态空间     神经模糊系统       梯度提升机        深度状态空间      PatchTST         Koopman预测
(Wold,1938)       (Time-Freq)      (Granger,1987)   (非高斯滤波)      (Harvey,1991)     (ANFIS)           (GBM原始)        (DSS)             (Yuqietal,2022)   (动力系统)
                                                                    
谱估计            季节性检验        VAR单位根检验    BATS/TBATS        FIGARCH           Elastic Net       级联森林          Temporal CNN      TimesNet         LLM用于时序
(频域分析)        (季节性识别)      (单位根)         (De Livera,2011)  (Baillie,1996)    (Zou,2005)        (gcForest)       (Temporal Conv)   (时频模型)        (GPT4TS)
                                                                    
自相关函数        结构突变检测      灰色预测法        因子模型          自回归条件持续时间  降维方法          超梯度提升        深度因子模型      Spacetimeformer   自回归分布式滞后
(ACF/PACF)        (Chow测试)       (Deng,1982)      (DFM)            (ACD)             (PCA/ICA)         (SuperGBM)       (DeepFactor)     (时空融合)        (ADL模型)
                                                                    
谱窗方法          指数平滑基础      Geweke分解       贝叶斯VAR         混沌理论应用       支持向量机变种     旋转森林          Prophet          Informer变种      非参数贝叶斯
(Bartlett窗)     (指数加权)        (参数分解)       (BVAR)           (Chaos Theory)    (ν-SVM)           (Rodriguez)      (Taylor,2018)    (Informer++,等)   (BNP方法)
