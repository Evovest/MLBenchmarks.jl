using MLBenchmarks
using MLBenchmarks: mse, mae, logloss, accuracy

using DataFrames
using CSV
using StatsBase: sample
using OrderedCollections

import NeuroTrees
import EvoTrees
import XGBoost
import LightGBM
import CatBoost

data = load_data(:titanic)
result_vars = [:model_type, :train_time, :logloss, :accuracy]
hyper_size = 16

################################
# EvoTrees
################################
dtrain = data[:dtrain]
deval = data[:deval]
dtest = data[:dtest]
feature_names = data[:feature_names]
target_name = data[:target_name]

hyper_list = MLBenchmarks.get_hyper_evotrees(loss="mlogloss", metric="mlogloss", nrounds=500, eta=0.05, max_depth=5:2:11, rowsample=[0.25, 0.5, 0.75, 1.0], colsample=[0.4, 0.6, 0.8, 1.0], L2=[0, 1, 10])
hyper_list = sample(hyper_list, hyper_size, replace=false)

results = Dict{Symbol,Any}[]
for (i, hyper) in enumerate(hyper_list)
    config = EvoTrees.EvoTreeClassifier(; hyper...)
    train_time = @elapsed m, logger = EvoTrees.fit_evotree(config, dtrain; deval, fnames=feature_names, target_name, metric=hyper[:metric], early_stopping_rounds=hyper[:early_stopping_rounds], return_logger=true)
    p_test = EvoTrees.predict(m, dtest)[:, 2]
    _logloss = logloss(p_test, data[:dtest][:, data[:target_name]])
    _accuracy = accuracy(p_test, data[:dtest][:, data[:target_name]])
    res = Dict(:model_type => "evotrees", :train_time => train_time, :best_nround => logger[:best_iter], :logloss => _logloss, :accuracy => _accuracy, hyper...)
    push!(results, res)
end
results_df = DataFrame(results)
select!(results_df, result_vars, Not(result_vars))
CSV.write(joinpath("results", "titanic", "evotrees-class.csv"), results_df)

################################
# XGBoost
################################
dtrain = XGBoost.DMatrix(data[:dtrain][:, data[:feature_names]], data[:dtrain][:, data[:target_name]])
deval = XGBoost.DMatrix(data[:deval][:, data[:feature_names]], data[:deval][:, data[:target_name]])
dtest = XGBoost.DMatrix(data[:dtest][:, data[:feature_names]])

hyper_list = MLBenchmarks.get_hyper_xgboost(objective="multi:softprob", eval_metric="mlogloss", num_class=2, num_round=500, eta=0.05, max_depth=4:2:10, subsample=[0.25, 0.5, 0.75, 1.0], colsample_bytree=[0.4, 0.6, 0.8, 1.0], lambda=[0, 1, 10])
hyper_list = sample(hyper_list, hyper_size, replace=false)

results = Dict{Symbol,Any}[]
for (i, hyper) in enumerate(hyper_list)
    train_time = @elapsed bst = XGBoost.xgboost(dtrain, watchlist=OrderedDict(["eval" => deval]); hyper...)
    p_test = XGBoost.predict(bst, dtest)[:, 2]
    _logloss = logloss(p_test, data[:dtest][:, data[:target_name]])
    _accuracy = accuracy(p_test, data[:dtest][:, data[:target_name]])
    res = Dict(:model_type => "xgboost", :train_time => train_time, :best_nround => bst.best_iteration, :logloss => _logloss, :accuracy => _accuracy, hyper...)
    push!(results, res)
end
results_df = DataFrame(results)
select!(results_df, result_vars, Not(result_vars))
CSV.write(joinpath("results", "titanic", "xgboost-class.csv"), results_df)
