using MLBenchmarks
import MLBenchmarks: mse, mae, logloss, accuracy, gini, ndcg
import MLBenchmarks.Datasets: aws_config

using DataFrames
using CSV
using Statistics: mean, std
using StatsBase: sample
using OrderedCollections

import NeuroTrees
import EvoTrees
import XGBoost
import LightGBM
import CatBoost

uniformize = false

data_name = uniformize ? "msrank/norm" : "msrank/raw"
data = load_data(:msrank; uniformize, aws_config)
result_vars = [:model_type, :train_time, :best_nround, :mse, :ndcg]
hyper_size = 16

#############################
# EDA
#############################
# using PlotlyLight
# using MLBenchmarks.Datasets: uniformer

# ops = uniformer(
#     data[:dtrain];
#     vars_in=data[:feature_names],
#     vars_out=data[:feature_names],
#     nbins=255,
#     min=-1,
#     max=1,
# )
# dtrain = copy(data[:dtrain])
# transform!(dtrain, ops)

# sort(unique(data[:dtrain].x12))
# sort(unique(dtrain.x12))
# PlotlyLight.Plot(; x=data[:dtrain].x12, type="histogram")
# PlotlyLight.Plot(; x=dtrain.x12, type="histogram")
# _sum0 = map(x -> mean(x .== 0), eachcol(data[:dtrain][!, data[:feature_names]]))
# sort(_sum0)
# PlotlyLight.Plot(; x=_sum0, type="histogram")


################################
# NeuroTrees
################################
dtrain = data[:dtrain]
deval = data[:deval]
dtest = data[:dtest]
feature_names = data[:feature_names]
target_name = data[:target_name]
batchsize = min(2048, nrow(dtrain))

_mean = mean(dtrain[!, target_name])
_std = std(dtrain[!, target_name])
dtrain.target_norm = (dtrain[!, target_name] .- _mean) ./ _std
deval.target_norm = (deval[!, target_name] .- _mean) ./ _std

hyper_list = MLBenchmarks.get_hyper_neurotrees(; loss="mse", metric="mse", tree_type="stack", device="gpu", nrounds=200, early_stopping_rounds=2, lr=1e-3, ntrees=[32, 64, 128, 256], stack_size=[1, 2], depth=[3, 4, 5], hidden_size=[8, 16, 32, 64], init_scale=0.0, batchsize)
hyper_list = sample(hyper_list, hyper_size, replace=false)

results = Dict{Symbol,Any}[]
for (i, hyper) in enumerate(hyper_list)
    @info "Loop $i"
    config = NeuroTrees.NeuroTreeRegressor(; hyper...)
    train_time = @elapsed m, logger = NeuroTrees.fit(config, dtrain; deval, feature_names, target_name="target_norm", metric=hyper[:metric], early_stopping_rounds=hyper[:early_stopping_rounds], print_every_n=10, return_logger=true)
    dinfer = NeuroTrees.get_df_loader_infer(dtest; feature_names, batchsize=config.batchsize, device=config.device)
    p_test = NeuroTrees.infer(m, dinfer) .* _std .+ _mean
    _mse = mse(p_test, data[:dtest][:, data[:target_name]])
    ndcg_df = DataFrame(p=p_test, y=data[:dtest][!, target_name], q=data[:dtest][!, "q"])
    ndcg_df = combine(groupby(ndcg_df, "q"), ["p", "y"] => ndcg => "ndcg")
    _ndcg = mean(ndcg_df.ndcg)
    res = Dict(:model_type => "neurotrees", :train_time => train_time, :best_nround => logger[:best_iter], :mse => _mse, :ndcg => _ndcg, hyper...)
    push!(results, res)
end
results_df = DataFrame(results)
select!(results_df, result_vars, Not(result_vars))
CSV.write(joinpath("results", data_name, "neurotrees2.csv"), results_df)

################################
# EvoTrees
################################
dtrain = data[:dtrain]
deval = data[:deval]
dtest = data[:dtest]
feature_names = data[:feature_names]
target_name = data[:target_name]

hyper_list = MLBenchmarks.get_hyper_evotrees(loss="mse", metric="mse", nrounds=5000, early_stopping_rounds=10, eta=0.05, max_depth=5:2:11, rowsample=[0.4, 0.6, 0.8, 1.0], colsample=[0.4, 0.6, 0.8, 1.0], L2=[0, 1, 10])
hyper_list = sample(hyper_list, hyper_size, replace=false)

results = Dict{Symbol,Any}[]
for (i, hyper) in enumerate(hyper_list)
    config = EvoTrees.EvoTreeRegressor(; hyper...)
    train_time = @elapsed m, logger = EvoTrees.fit_evotree(config, dtrain; deval, fnames=feature_names, target_name, metric=hyper[:metric], early_stopping_rounds=hyper[:early_stopping_rounds], print_every_n=10, return_logger=true)
    p_test = EvoTrees.predict(m, dtest)
    _mse = mse(p_test, dtest[:, target_name])
    ndcg_df = DataFrame(p=p_test, y=data[:dtest][!, target_name], q=data[:dtest][!, "q"])
    ndcg_df = combine(groupby(ndcg_df, "q"), ["p", "y"] => ndcg => "ndcg")
    _ndcg = mean(ndcg_df.ndcg)
    res = Dict(:model_type => "evotrees", :train_time => train_time, :best_nround => logger[:best_iter], :mse => _mse, :ndcg => _ndcg, hyper...)
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

hyper_list = MLBenchmarks.get_hyper_xgboost(objective="reg:squarederror", eval_metric="rmse", num_round=5000, early_stopping_rounds=10, eta=0.05, max_depth=4:2:10, subsample=[0.4, 0.6, 0.8, 1.0], colsample_bytree=[0.4, 0.6, 0.8, 1.0], lambda=[0, 1, 10])
hyper_list = sample(hyper_list, hyper_size, replace=false)

results = Dict{Symbol,Any}[]
for (i, hyper) in enumerate(hyper_list)
    train_time = @elapsed bst = XGBoost.xgboost(dtrain, watchlist=OrderedDict(["eval" => deval]); hyper...)
    p_test = XGBoost.predict(bst, dtest)
    _mse = mse(p_test, data[:dtest][:, data[:target_name]])
    ndcg_df = DataFrame(p=p_test, y=data[:dtest][!, target_name], q=data[:dtest][!, "q"])
    ndcg_df = combine(groupby(ndcg_df, "q"), ["p", "y"] => ndcg => "ndcg")
    _ndcg = mean(ndcg_df.ndcg)
    res = Dict(:model_type => "xgboost", :train_time => train_time, :best_nround => bst.best_iteration, :mse => _mse, :ndcg => _ndcg, hyper...)
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

hyper_list = MLBenchmarks.get_hyper_lgbm(objective="mse", metric=["mse"], num_iterations=5000, early_stopping_round=10, learning_rate=0.05, num_leaves=[32, 128, 512, 2048], bagging_fraction=[0.3, 0.6, 0.9], feature_fraction=[0.5, 0.9], lambda_l2=[0, 1, 10])
hyper_list = sample(hyper_list, hyper_size, replace=false)

results = Dict{Symbol,Any}[]
for (i, hyper) in enumerate(hyper_list)
    estimator = LightGBM.LGBMRegression(; hyper...)
    train_time = @elapsed res = LightGBM.fit!(estimator, dtrain, ytrain, (deval, yeval))
    p_test = vec(LightGBM.predict(estimator, dtest))
    _mse = mse(p_test, data[:dtest][:, data[:target_name]])
    ndcg_df = DataFrame(p=p_test, y=data[:dtest][!, target_name], q=data[:dtest][!, "q"])
    ndcg_df = combine(groupby(ndcg_df, "q"), ["p", "y"] => ndcg => "ndcg")
    _ndcg = mean(ndcg_df.ndcg)
    res = Dict(:model_type => "lightgbm", :train_time => train_time, :best_nround => res["best_iter"], :mse => _mse, :ndcg => _ndcg, hyper...)
    push!(results, res)
end
results_df = DataFrame(results)
select!(results_df, result_vars, Not(result_vars))
CSV.write(joinpath("results", data_name, "lightgbm.csv"), results_df)


################################
# CatBoost
################################
using PythonCall
dtrain = CatBoost.Pool(data[:dtrain][:, data[:feature_names]], label=PyList(data[:dtrain][:, data[:target_name]]))
deval = CatBoost.Pool(data[:deval][:, data[:feature_names]], label=PyList(data[:deval][:, data[:target_name]]))
dtest = CatBoost.Pool(data[:dtest][:, data[:feature_names]])

hyper_list = MLBenchmarks.get_hyper_catboost(objective="RMSE", eval_metric="RMSE", iterations=5000, early_stopping_rounds=10, learning_rate=0.1, max_depth=4:2:10, subsample=[0.3, 0.6, 0.9], rsm=[0.5, 0.9], reg_lambda=[0, 1, 10])
hyper_list = sample(hyper_list, hyper_size, replace=false)

results = Dict{Symbol,Any}[]
for (i, hyper) in enumerate(hyper_list)
    model = CatBoost.CatBoostRegressor(; hyper...)
    train_time = @elapsed res = CatBoost.fit!(model, dtrain; eval_set=deval)
    p_test = CatBoost.predict(model, dtest)
    _mse = mse(p_test, data[:dtest][:, data[:target_name]])
    ndcg_df = DataFrame(p=p_test, y=data[:dtest][!, target_name], q=data[:dtest][!, "q"])
    ndcg_df = combine(groupby(ndcg_df, "q"), ["p", "y"] => ndcg => "ndcg")
    _ndcg = mean(ndcg_df.ndcg)
    res = Dict(:model_type => "catboost", :train_time => train_time, :best_nround => pyconvert(Int, res.best_iteration_), :mse => _mse, :ndcg => _ndcg, hyper...)
    push!(results, res)
end
results_df = DataFrame(results)
select!(results_df, result_vars, Not(result_vars))
CSV.write(joinpath("results", data_name, "catboost.csv"), results_df)
