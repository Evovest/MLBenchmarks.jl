# MLBenchmarks.jl 

This repo provides Julia based benchmarks for ML algo on tabular data. 
It was developed to support both [NeuroTreeModels.jl](https://github.com/Evovest/NeuroTreeModels.jl) and [EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl) projects.

## Methodology

For each dataset and algo, the following methodology is followed:
- Data is split in three parts: `train`, `eval` and `test`
- A random grid of 16 hyper-parameters is generated
- For each parameter configuration, a model is trained on `train` data until the evaluation metric tracked against the `eval` stops improving (early stopping)
- The trained model is evaluated against the `test` data
- The metric presented in below are the ones obtained on the `test` for the model that generated the best `eval` metric.

## Datasets

The following selection of common tabular datasets is covered:

- [Year](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd): min squared error regression
- [MSRank](https://www.microsoft.com/en-us/research/project/mslr/): ranking problem with min squared error regression 
- [YahooRank](https://webscope.sandbox.yahoo.com/): ranking problem with min squared error regression
- [Higgs](https://archive.ics.uci.edu/dataset/280/higgs): 2-level classification with logistic regression
- [Boston Housing](https://juliaml.github.io/MLDatasets.jl/stable/datasets/misc/#MLDatasets.BostonHousing): min squared error regression
- [Titanic](https://juliaml.github.io/MLDatasets.jl/stable/datasets/misc/#MLDatasets.Titanic): 2-level classification with logistic regression

## Algorithms

Comparison is performed against the following algos (implementation in link) considered as state of the art on tabular data problems tasks:

- [EvoTrees](https://github.com/Evovest/EvoTrees.jl)
- [XGBoost](https://github.com/dmlc/XGBoost.jl)
- [LightGBM](https://github.com/IQVIA-ML/LightGBM.jl)
- [CatBoost](https://github.com/JuliaAI/CatBoost.jl)
- [NODE](https://github.com/manujosephv/pytorch_tabular)

### Boston

| **model\_type** | **train\_time** | **test\_mse** | **test\_gini** |
|:---------------:|:---------------:|:-------------:|:--------------:|
|    catboost     |      0.193      |     0.194     |     0.945      |
|    evotrees     |      0.12       |     0.254     |     0.935      |
|    lightgbm     |      0.303      |     0.326     |     0.934      |
|   neurotrees    |      3.61       |     0.285     |     0.924      |
|      tabm       |      7.11       |     0.291     |      0.91      |
|     xgboost     |     0.0926      |     0.265     |      0.93      |

### Titanic

| **model\_type** | **train\_time** | **test\_logloss** | **test\_gini** |
|:---------------:|:---------------:|:-----------------:|:--------------:|
|    catboost     |     0.0741      |       0.375       |     0.802      |
|    evotrees     |     0.0326      |       0.362       |     0.806      |
|    lightgbm     |      0.25       |       0.363       |     0.809      |
|   neurotrees    |      3.51       |       0.377       |     0.809      |
|      tabm       |      7.71       |       0.373       |     0.796      |
|     xgboost     |     0.0222      |       0.37        |     0.795      |

### Year

| **model\_type** | **train\_time** | **test\_mse** | **test\_gini** |
|:---------------:|:---------------:|:-------------:|:--------------:|
|    catboost     |      64.8       |     0.621     |     0.664      |
|    evotrees     |      78.8       |     0.613     |     0.666      |
|    lightgbm     |      127.0      |     0.607     |      0.67      |
|   neurotrees    |      32.8       |     0.61      |     0.673      |
|      tabm       |      115.0      |     0.62      |     0.666      |
|     xgboost     |      41.9       |     0.614     |     0.666      |

### Microsoft

| **model\_type** | **train\_time** | **test\_mse** | **test\_gini** |
|:---------------:|:---------------:|:-------------:|:--------------:|
|    catboost     |      192.0      |     0.73      |     0.561      |
|    evotrees     |      99.9       |     0.722     |     0.567      |
|    lightgbm     |      50.5       |     0.717     |     0.571      |
|   neurotrees    |      88.8       |     0.778     |     0.514      |
|      tabm       |      69.9       |     0.776     |     0.512      |
|     xgboost     |      38.9       |     0.719     |      0.57      |

### Higgs

| **model\_type** | **train\_time** | **test\_logloss** | **test\_gini** |
|:---------------:|:---------------:|:-----------------:|:--------------:|
|    catboost     |      147.0      |       0.494       |     0.674      |
|    evotrees     |      57.4       |       0.496       |      0.67      |
|    lightgbm     |      118.0      |       0.495       |     0.673      |
|   neurotrees    |      58.0       |       0.502       |     0.662      |
|      tabm       |      123.0      |       0.497       |     0.671      |
|     xgboost     |      33.9       |       0.496       |      0.67      |

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
