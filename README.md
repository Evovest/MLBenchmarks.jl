# MLBenchmarks.jl 

This repo provides Julia based benchmarks for ML algo on tabular data. 
It was developed to support both [NeuroTabModels.jl](https://github.com/Evovest/NeuroTabModels.jl) and [EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl) projects.

## Methodology

For each dataset and algo, the following methodology is followed:
- Data is split in three parts: `train`, `eval` and `test`
- A random grid of 16 hyper-parameters is generated
- For each parameter configuration, a model is trained on `train` data until the evaluation metric tracked against the `eval` stops improving (early stopping)
- The trained model is evaluated against the `test` data
- The metric presented below are the ones obtained on the `test` for the model that generated the best `eval` metric.

## Datasets

### Datasets are now sourced from OpenML, using[OpenML](https://github.com/JuliaAI/OpenML.jl):

```julia
data_map = Dict(
    :titanic => 40945,
    :higgs_11M => 45570,
    :higgs_1M => 42769,
    :boston => 531,
    :year => 44027,
    :microsoft => 45579,
    :sberbank => 46898,
    :allstate_claims => 45046,
    :creditcard => 1597
)
```

### Legacy datasets from older release

The following selection of common tabular datasets is covered:

- [Year](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd): min squared error regression
- [MSRank](https://www.microsoft.com/en-us/research/project/mslr/): ranking problem with min squared error regression 
- [YahooRank](https://webscope.sandbox.yahoo.com/): ranking problem with min squared error regression
- [Higgs](https://archive.ics.uci.edu/dataset/280/higgs): 2-level classification with logistic regression
- [Boston Housing](https://juliaml.github.io/MLDatasets.jl/stable/datasets/misc/#MLDatasets.BostonHousing): min squared error regression
- [Titanic](https://juliaml.github.io/MLDatasets.jl/stable/datasets/misc/#MLDatasets.Titanic): 2-level classification with logistic regression

## Algorithms

Comparison is performed against the following algos (implementation in link) considered as state of the art on tabular data problems tasks:

- [NeuroTabModels](https://github.com/Evovest/NeuroTabModels.jl)
- [EvoTrees](https://github.com/Evovest/EvoTrees.jl)
- [XGBoost](https://github.com/dmlc/XGBoost.jl)
- [LightGBM](https://github.com/IQVIA-ML/LightGBM.jl)
- [CatBoost](https://github.com/JuliaAI/CatBoost.jl)
- [NODE](https://github.com/manujosephv/pytorch_tabular)

### Boston

| **model\_type** | **train\_time** | **test\_mse** | **test\_gini** |
|:---------------:|:---------------:|:-------------:|:--------------:|
|    catboost     |      0.191      |     0.194     |     0.945      |
|    evotrees     |      0.123      |     0.254     |     0.935      |
|    lightgbm     |     0.0915      |     0.326     |     0.934      |
|    neurotree    |      3.87       |     0.283     |     0.926      |
|      tabm       |      7.66       |     0.223     |     0.937      |
|     xgboost     |     0.0855      |     0.265     |      0.93      |

### Titanic

| **model\_type** | **train\_time** | **test\_logloss** | **test\_gini** |
|:---------------:|:---------------:|:-----------------:|:--------------:|
|    catboost     |     0.0932      |       0.375       |     0.802      |
|    evotrees     |      0.033      |       0.362       |     0.806      |
|    lightgbm     |     0.0593      |       0.363       |     0.809      |
|    neurotree    |      2.12       |       0.375       |     0.817      |
|      tabm       |      24.1       |       0.382       |     0.768      |
|     xgboost     |      0.019      |       0.37        |     0.795      |

### Year

| **model\_type** | **train\_time** | **test\_mse** | **test\_gini** |
|:---------------:|:---------------:|:-------------:|:--------------:|
|    catboost     |      66.9       |     0.621     |     0.664      |
|    evotrees     |      78.4       |     0.613     |     0.666      |
|    lightgbm     |      106.0      |     0.607     |      0.67      |
|    neurotree    |      146.0      |     0.597     |      0.68      |
|      tabm       |      26.0       |     0.606     |     0.675      |
|     xgboost     |      42.1       |     0.614     |     0.666      |

### Microsoft

| **model\_type** | **train\_time** | **test\_mse** | **test\_gini** |
|:---------------:|:---------------:|:-------------:|:--------------:|
|    catboost     |      183.0      |     0.73      |     0.561      |
|    evotrees     |      95.0       |     0.722     |     0.567      |
|    lightgbm     |      44.8       |     0.717     |     0.571      |
|    neurotree    |     3990.0      |     0.761     |      0.53      |
|      tabm       |      165.0      |     0.771     |     0.517      |
|     xgboost     |      37.4       |     0.719     |      0.57      |

### Higgs

| **model\_type** | **train\_time** | **test\_logloss** | **test\_gini** |
|:---------------:|:---------------:|:-----------------:|:--------------:|
|    catboost     |      135.0      |       0.494       |     0.674      |
|    evotrees     |      54.2       |       0.496       |      0.67      |
|    lightgbm     |      92.8       |       0.495       |     0.673      |
|    neurotree    |      480.0      |       0.488       |     0.684      |
|      tabm       |      39.1       |       0.498       |     0.669      |
|     xgboost     |      31.2       |       0.496       |      0.67      |

### AllState claims

| **model\_type** | **train\_time** | **test\_mse** | **test\_gini** |
|:---------------:|:---------------:|:-------------:|:--------------:|
|    catboost     |      2.21       |     0.438     |     0.741      |
|    evotrees     |      3.06       |     0.44      |      0.74      |
|    lightgbm     |      1.95       |     0.437     |     0.742      |
|    neurotree    |      102.0      |     0.442     |     0.738      |
|      tabm       |      35.4       |     0.438     |     0.743      |
|     xgboost     |      1.31       |     0.439     |     0.741      |

### Creditcart

| **model\_type** | **train\_time** | **test\_logloss** | **test\_gini** |
|:---------------:|:---------------:|:-----------------:|:--------------:|
|    catboost     |      2.96       |      0.00211      |      0.97      |
|    evotrees     |       2.3       |      0.00199      |     0.982      |
|    lightgbm     |      1.57       |      0.00208      |     0.979      |
|    neurotree    |      25.4       |      0.00248      |     0.983      |
|      tabm       |      10.1       |      0.00219      |     0.975      |
|     xgboost     |      0.648      |      0.00198      |      0.98      |

## References

- [Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data](https://arxiv.org/abs/1909.06312v2).
    - [https://github.com/Qwicen/node](https://github.com/Qwicen/node)
    - [https://github.com/manujosephv/pytorch_tabular](https://github.com/manujosephv/pytorch_tabular)
- [Attention augmented differentiable forest for tabular data](https://arxiv.org/abs/2010.02921)
- [NCART: Neural Classification and Regression Tree for Tabular Data](https://arxiv.org/abs/2307.12198)
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- [Deep Neural Decision Trees](https://arxiv.org/abs/1806.06988)
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [CatBoost: gradient boosting with categorical features support](https://arxiv.org/abs/1810.11363)
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
- [Deep Neural Decision Trees](https://arxiv.org/abs/1806.06988)
- [Neural Decision Trees](https://arxiv.org/abs/1702.07360)
