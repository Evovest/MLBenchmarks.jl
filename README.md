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
- [MSRank](https://www.microsoft.com/en-us/research/project/mslr/): ranking problem using logistic regression 
- [YahooRank](https://webscope.sandbox.yahoo.com/): ranking problem using logistic regression
- [Higgs](https://archive.ics.uci.edu/dataset/280/higgs): 2-class classification using logistic regression
- [Boston](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd): min squared error regression
- [Titanic](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd): min squared error regression

## Algorithms

Comparison is performed against the following algos (implementation in link) considered as state of the art on tabular data problems tasks:

- [EvoTrees](https://github.com/Evovest/EvoTrees.jl)
- [XGBoost](https://github.com/dmlc/XGBoost.jl)
- [LightGBM](https://github.com/IQVIA-ML/LightGBM.jl)
- [CatBoost](https://github.com/JuliaAI/CatBoost.jl)
- [NODE](https://github.com/manujosephv/pytorch_tabular)

### Boston

| **model\_type** | **train\_time** | **mse** | **gini** |
|:---------------:|:---------------:|:-------:|:--------:|
| neurotrees      | 12.8            | 18.9    | **0.947**|
| evotrees        | 0.206           | 19.7    | 0.927    |
| xgboost         | 0.0648          | 19.4    | 0.935    |
| lightgbm        | 0.865           | 25.4    | 0.926    |
| catboost        | 0.0511          | **13.9**| 0.946    |

### Titanic

| **model\_type** | **train\_time** | **logloss** | **accuracy** |
|:---------------:|:---------------:|:-----------:|:------------:|
| neurotrees      | 7.58            | 0.407       | 0.828        |
| evotrees        | 0.673           | 0.382       | 0.828        |
| xgboost         | 0.0379          | **0.375**   | 0.821        |
| lightgbm        | 0.615           | 0.390       | **0.836**    |
| catboost        | 0.0326          | 0.388       | **0.836**    |

### Year

| **model\_type** | **train\_time** | **mse** | **gini** |
|:---------------:|:---------------:|:-------:|:--------:|
| neurotrees      | 280.0           | **76.4**| **0.652**|
| evotrees        | 18.6            | 80.1    | 0.627    |
| xgboost         | 17.2            | 80.2    | 0.626    |
| lightgbm        | 8.11            | 80.3    | 0.624    |
| catboost        | 80.0            | 79.2    | 0.635    |

### MSRank

| **model\_type** | **train\_time** | **mse** | **ndcg** |
|:---------------:|:---------------:|:-------:|:--------:|
| neurotrees      | 39.1            | 0.578   | 0.462    |
| evotrees        | 37.0            | 0.554   | **0.504**|
| xgboost         | 12.5            | 0.554   | 0.503    |
| lightgbm        | 37.5            |**0.553**| 0.503    |
| catboost        | 15.1            | 0.558   | 0.497    |

### Yahoo

| **model\_type** | **train\_time** | **mse** | **ndcg** |
|:---------------:|:---------------:|:-------:|:--------:|
| neurotrees      | 417.0           | 0.584   | 0.781    |
| evotrees        | 687.0           | 0.545   | 0.797    |
| xgboost         | 120.0           | 0.547   | **0.798**|
| lightgbm        | 244.0           |**0.540**| 0.796    |
| catboost        | 161.0           | 0.561   | 0.794    |

### Higgs

| **model\_type** | **train\_time** | **logloss** | **accuracy** |
|:---------------:|:---------------:|:-----------:|:------------:|
| neurotrees      | 12300.0         | **0.452**   | **0.781**    |
| evotrees        | 2620.0          | 0.464       | 0.776        |
| xgboost         | 1390.0          | 0.462       | 0.776        |
| lightgbm        | 1330.0          | 0.461       | 0.779        |
| catboost        | 7180.0          | 0.464       | 0.775        |

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
