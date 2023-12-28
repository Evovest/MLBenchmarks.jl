using MLBenchmarks
using MLBenchmarks: mse, mae, logloss, accuracy

using DataFrames
using CSV
using Statistics: mean, std
using StatsBase: sample
using OrderedCollections

import NeuroTrees
import EvoTrees
import XGBoost
import LightGBM
# import CatBoost

data_name = "titanic"
data = load_data(Symbol(data_name))
result_vars = [:model_type, :train_time, :best_nround, :logloss, :accuracy]
hyper_size = 16

################################
# NeuroTrees
################################
dtrain = data[:dtrain]
deval = data[:deval]
dtest = data[:dtest]
feature_names = data[:feature_names]
target_name = data[:target_name]
batchsize = min(4096, ceil(Int, 0.5 * nrow(dtrain)))

hyper_list = MLBenchmarks.get_hyper_neurotrees(; loss="logloss", metric="logloss", tree_type="stack", device="gpu", nrounds=500, lr=3e-3, num_trees=[16, 32, 64], stack_size=[1, 2, 3], boosting_size=1, depth=[3, 4, 5], hidden_size=[8, 16, 32], early_stopping_rounds=2, batchsize)
hyper_list = sample(hyper_list, hyper_size, replace=false)

results = Dict{Symbol,Any}[]
for (i, hyper) in enumerate(hyper_list)
    @info "iter $i"
    config = NeuroTrees.NeuroTreeRegressor(; hyper...)
    train_time = @elapsed m, logger = NeuroTrees.fit(config, dtrain; deval, feature_names, target_name, metric=hyper[:metric], early_stopping_rounds=hyper[:early_stopping_rounds], print_every_n=10, return_logger=true)
    dinfer = NeuroTrees.get_df_loader_infer(dtest; feature_names, batchsize=config.batchsize, device=config.device)
    p_test = NeuroTrees.infer(m, dinfer)
    _logloss = logloss(p_test, data[:dtest][:, data[:target_name]])
    _accuracy = accuracy(p_test, data[:dtest][:, data[:target_name]])
    res = Dict(:model_type => "neurotrees", :train_time => train_time, :best_nround => logger[:best_iter], :logloss => _logloss, :accuracy => _accuracy, hyper...)
    push!(results, res)
end
results_df = DataFrame(results)
select!(results_df, result_vars, Not(result_vars))
CSV.write(joinpath("results", data_name, "neurotrees.csv"), results_df)

################################
# EvoTrees
################################
dtrain = data[:dtrain]
deval = data[:deval]
dtest = data[:dtest]
feature_names = data[:feature_names]
target_name = data[:target_name]

hyper_list = MLBenchmarks.get_hyper_evotrees(loss="logloss", metric="logloss", nrounds=500, early_stopping_rounds=5, eta=0.05, max_depth=5:2:11, rowsample=[0.4, 0.6, 0.8, 1.0], colsample=[0.4, 0.6, 0.8, 1.0], L2=[0, 1, 10])
hyper_list = sample(hyper_list, hyper_size, replace=false)

results = Dict{Symbol,Any}[]
for (i, hyper) in enumerate(hyper_list)
    config = EvoTrees.EvoTreeRegressor(; hyper...)
    train_time = @elapsed m, logger = EvoTrees.fit_evotree(config, dtrain; deval, fnames=feature_names, target_name, metric=hyper[:metric], early_stopping_rounds=hyper[:early_stopping_rounds], print_every_n=10, return_logger=true)
    p_test = EvoTrees.predict(m, dtest)
    _logloss = logloss(p_test, data[:dtest][:, data[:target_name]])
    _accuracy = accuracy(p_test, data[:dtest][:, data[:target_name]])
    res = Dict(:model_type => "evotrees", :train_time => train_time, :best_nround => logger[:best_iter], :logloss => _logloss, :accuracy => _accuracy, hyper...)
    push!(results, res)
end
results_df = DataFrame(results)
select!(results_df, result_vars, Not(result_vars))
CSV.write(joinpath("results", data_name, "evotrees.csv"), results_df)

################################
# XGBoost
################################
dtrain = XGBoost.DMatrix(data[:dtrain][:, data[:feature_names]], data[:dtrain][:, data[:target_name]])
deval = XGBoost.DMatrix(data[:deval][:, data[:feature_names]], data[:deval][:, data[:target_name]])
dtest = XGBoost.DMatrix(data[:dtest][:, data[:feature_names]])

hyper_list = MLBenchmarks.get_hyper_xgboost(objective="reg:logistic", eval_metric="logloss", num_round=500, eta=0.05, max_depth=4:2:10, subsample=[0.4, 0.6, 0.8, 1.0], colsample_bytree=[0.4, 0.6, 0.8, 1.0], lambda=[0, 1, 10])
hyper_list = sample(hyper_list, hyper_size, replace=false)

results = Dict{Symbol,Any}[]
for (i, hyper) in enumerate(hyper_list)
    train_time = @elapsed bst = XGBoost.xgboost(dtrain, watchlist=OrderedDict(["eval" => deval]); hyper...)
    p_test = XGBoost.predict(bst, dtest)
    _logloss = logloss(p_test, data[:dtest][:, data[:target_name]])
    _accuracy = accuracy(p_test, data[:dtest][:, data[:target_name]])
    res = Dict(:model_type => "xgboost", :train_time => train_time, :best_nround => bst.best_iteration, :logloss => _logloss, :accuracy => _accuracy, hyper...)
    push!(results, res)
end
results_df = DataFrame(results)
select!(results_df, result_vars, Not(result_vars))
CSV.write(joinpath("results", data_name, "xgboost.csv"), results_df)

################################
# LightGBM
################################
dtrain, ytrain = Matrix(data[:dtrain][:, data[:feature_names]]), data[:dtrain][:, data[:target_name]]
deval, yeval = Matrix(data[:deval][:, data[:feature_names]]), data[:deval][:, data[:target_name]]
dtest = Matrix(data[:dtest][:, data[:feature_names]])

hyper_list = MLBenchmarks.get_hyper_lgbm(objective="binary", metric=["logloss"], num_iterations=500, early_stopping_round=5, learning_rate=0.1, num_leaves=[32, 128, 512, 2048], bagging_fraction=[0.3, 0.6, 0.9], feature_fraction=[0.5, 0.9], lambda_l2=[0, 1, 10])
hyper_list = sample(hyper_list, hyper_size, replace=false)

results = Dict{Symbol,Any}[]
for (i, hyper) in enumerate(hyper_list)
    estimator = LightGBM.LGBMClassification(; hyper...)
    train_time = @elapsed res = LightGBM.fit!(estimator, dtrain, ytrain, (deval, yeval))
    p_test = vec(LightGBM.predict(estimator, dtest))
    _logloss = logloss(p_test, data[:dtest][:, data[:target_name]])
    _accuracy = accuracy(p_test, data[:dtest][:, data[:target_name]])
    res = Dict(:model_type => "lightgbm", :train_time => train_time, :best_nround => res["best_iter"], :logloss => _logloss, :accuracy => _accuracy, hyper...)
    push!(results, res)
end
results_df = DataFrame(results)
select!(results_df, result_vars, Not(result_vars))
CSV.write(joinpath("results", data_name, "lightgbm.csv"), results_df)


################################
# CatBoost
################################
# using PythonCall
# dtrain = CatBoost.Pool(data[:dtrain][:, data[:feature_names]], label=PyList(data[:dtrain][:, data[:target_name]]))
# deval = CatBoost.Pool(data[:deval][:, data[:feature_names]], label=PyList(data[:deval][:, data[:target_name]]))
# dtest = CatBoost.Pool(data[:dtest][:, data[:feature_names]])

# hyper_list = MLBenchmarks.get_hyper_catboost(objective="Logloss", eval_metric="Logloss", iterations=500, early_stopping_rounds=5, learning_rate=0.1, max_depth=4:2:10, subsample=[0.3, 0.6, 0.9], rsm=[0.5, 0.9], reg_lambda=[0, 1, 10])
# hyper_list = sample(hyper_list, hyper_size, replace=false)

# results = Dict{Symbol,Any}[]
# for (i, hyper) in enumerate(hyper_list)
#     model = CatBoost.CatBoostClassifier(; hyper...)
#     train_time = @elapsed res = CatBoost.fit!(model, dtrain; eval_set=deval)
#     p_test = CatBoost.predict(model, dtest; prediction_type="Probability")[:, 2]
#     _logloss = logloss(p_test, data[:dtest][:, data[:target_name]])
#     _accuracy = accuracy(p_test, data[:dtest][:, data[:target_name]])
#     res = Dict(:model_type => "catboost", :train_time => train_time, :best_nround => pyconvert(Int, res.best_iteration_), :logloss => _logloss, :accuracy => _accuracy, hyper...)
#     push!(results, res)
# end
# results_df = DataFrame(results)
# select!(results_df, result_vars, Not(result_vars))
# CSV.write(joinpath("results", data_name, "catboost.csv"), results_df)
