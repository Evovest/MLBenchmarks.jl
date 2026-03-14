using MLBenchmarks
import MLBenchmarks: mse, mae, logloss, accuracy, gini, ndcg
import MLBenchmarks: run_experiment

using DataFrames
using CSV
using Statistics: mean, std
using StatsBase: sample
using OrderedCollections

data_name = :titanic
data = load_data(data_name)
hyper_size = 16
uniformize = false

################################
# NeuroTrees
################################
hyper_list = MLBenchmarks.get_hyper_neurotrees(hyper_size; loss=:logloss, metric=:logloss, nrounds=200, early_stopping_rounds=2, lr=3e-2, ntrees=[16, 32, 64], depth=[3, 4, 5], hidden_size=[8, 16, 32], init_scale=0.0)
results_df = run_experiment(:NeuroTabModels, data, hyper_list)
CSV.write(joinpath("results", string(data_name), "neurotrees.csv"), results_df)

################################
# EvoTrees
################################
hyper_list = MLBenchmarks.get_hyper_evotrees(hyper_size; loss="logloss", metric="logloss", nrounds=1000, early_stopping_rounds=10, eta=0.05, max_depth=5:2:11, rowsample=[0.4, 0.6, 0.8, 1.0], colsample=[0.4, 0.6, 0.8, 1.0], L2=[0, 1, 10])
results_df = run_experiment(:EvoTrees, data, hyper_list)
CSV.write(joinpath("results", string(data_name), "evotrees.csv"), results_df)

################################
# XGBoost
################################
hyper_list = MLBenchmarks.get_hyper_xgboost(hyper_size; objective="reg:logistic", eval_metric="logloss", num_round=1000, early_stopping_rounds=10, eta=0.05, max_depth=4:2:10, subsample=[0.4, 0.6, 0.8, 1.0], colsample_bytree=[0.4, 0.6, 0.8, 1.0], lambda=[0, 1, 10])
results_df = run_experiment(:XGBoost, data, hyper_list)
CSV.write(joinpath("results", string(data_name), "xgboost.csv"), results_df)

################################
# LightGBM
################################
hyper_list = MLBenchmarks.get_hyper_lgbm(hyper_size; objective="cross_entropy", metric=["cross_entropy"], num_iterations=1000, early_stopping_round=10, learning_rate=0.05, num_leaves=[32, 128, 512, 2048], bagging_fraction=[0.3, 0.6, 0.9], feature_fraction=[0.5, 0.9], lambda_l2=[0, 1, 10])
results_df = run_experiment(:LightGBM, data, hyper_list)
CSV.write(joinpath("results", string(data_name), "lightgbm.csv"), results_df)

################################
# CatBoost
################################
hyper_list = MLBenchmarks.get_hyper_catboost(hyper_size; objective="Logloss", eval_metric="Logloss", iterations=1000, early_stopping_rounds=10, learning_rate=0.05, max_depth=4:2:10, subsample=[0.3, 0.6, 0.9], rsm=[0.5, 0.9], reg_lambda=[0, 1, 10])
results_df = run_experiment(:CatBoost, data, hyper_list)
CSV.write(joinpath("results", string(data_name), "catboost.csv"), results_df)
