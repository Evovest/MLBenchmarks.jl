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

| **model\_type** | **train\_time** | **test\_gini** | **test\_mse** |
|:---------------:|:---------------:|:--------------:|:-------------:|
|    catboost     |      0.521      |     0.937      |   **0.206**   |
|    evotrees     |      0.136      |     0.930      |     0.282     |
|    lightgbm     |      0.265      |     0.931      |     0.316     |
|   neurotrees    |      0.422      |   **0.942**    |     0.233     |
|     xgboost     |      0.063      |     0.927      |     0.302     |

### Titanic

| **model\_type** | **train\_time** | **test\_logloss** | **test\_gini** |
|:---------------:|:---------------:|:-----------------:|:--------------:|
|    catboost     |      0.098      |       0.376       |     0.787      |
|    evotrees     |      0.035      |     **0.367**     |     0.801      |
|    lightgbm     |      0.142      |       0.380       |     0.794      |
|   neurotrees    |      0.335      |       0.379       |     0.812      |
|     xgboost     |      0.017      |       0.372       |   **0.814**    |

### Year

| **model\_type** | **train\_time** | **test\_gini** | **test\_mse** |
|:---------------:|:---------------:|:--------------:|:-------------:|
|    catboost     |      88.9       |     0.664      |     0.621     |
|    evotrees     |      72.1       |     0.666      |     0.615     |
|    lightgbm     |      82.1       |     0.668      |     0.611     |
|   neurotrees    |     107.0       |   **0.683**    |   **0.595**   |
|     xgboost     |      59.3       |     0.668      |     0.611     |

### Microsoft

| **model\_type** | **train\_time** | **test\_gini** | **test\_mse** |
|:---------------:|:---------------:|:--------------:|:-------------:|
|    catboost     |      261.0      |     0.561      |     0.730     |
|    evotrees     |      123.0      |   **0.568**    |   **0.720**   |
|    lightgbm     |      44.9       |   **0.568**    |   **0.720**   |
|   neurotrees    |      46.3       |     0.476      |     0.820     |
|     xgboost     |      40.5       |     0.566      |     0.722     |

### Higgs

| **model\_type** | **train\_time** | **test\_logloss** | **test\_gini** |
|:---------------:|:---------------:|:-----------------:|:--------------:|
|    catboost     |      146.0      |       0.495       |     0.674      |
|    evotrees     |      60.0       |       0.496       |     0.671      |
|    lightgbm     |      110.0      |       0.494       |     0.674      |
|   neurotrees    |      202.0      |     **0.491**     |   **0.680**    |
|     xgboost     |      30.3       |       0.496       |     0.671      |

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
