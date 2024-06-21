using MLBenchmarks
import MLBenchmarks: mse, mae, logloss, accuracy, gini, ndcg
import MLBenchmarks.Datasets: aws_config

using DataFrames
using CSV
using Statistics: mean, std
using StatsBase: sample
using OrderedCollections

import NeuroTreeModels
import EvoTrees
import XGBoost
import LightGBM
import CatBoost

uniformize = false

data_name = uniformize ? "year/norm" : "year/raw"
data = load_data(:year; uniformize, aws_config)
result_vars = [:model_type, :train_time, :best_nround, :mse, :gini]
hyper_size = 16
nthreads = Threads.nthreads()

preds = Dict{String,Vector}()
results_test = Dict{Symbol,Any}[]

################################
# NeuroTrees
################################
dtrain = data[:dtrain]
deval = data[:deval]
dtest = data[:dtest]
feature_names = data[:feature_names]
target_name = data[:target_name]
batchsize = min(2048, nrow(dtrain))
device = :gpu

_mean = mean(dtrain[!, target_name])
_std = std(dtrain[!, target_name])
dtrain.target_norm = (dtrain[!, target_name] .- _mean) ./ _std
deval.target_norm = (deval[!, target_name] .- _mean) ./ _std

hyper_list = MLBenchmarks.get_hyper_neurotrees(; loss=:mse, metric=:mse, device=:gpu, nrounds=200, early_stopping_rounds=2, lr=1e-3, ntrees=[32, 64, 128], stack_size=[1], depth=[3, 4, 5], hidden_size=[8, 16, 32], init_scale=0.0, batchsize)
hyper_list = sample(hyper_list, hyper_size, replace=false)

results = Dict{Symbol,Any}[]
models = Vector()
for (i, hyper) in enumerate(hyper_list)
    @info "Loop $i"
    config = NeuroTreeModels.NeuroTreeRegressor(; hyper...)
    train_time = @elapsed m = NeuroTreeModels.fit(config, dtrain; deval, feature_names, target_name="target_norm", metric=hyper[:metric], early_stopping_rounds=hyper[:early_stopping_rounds], print_every_n=10, device)
    push!(models, m)
    p_eval = m(deval) .* _std .+ _mean
    _mse = mse(p_eval, data[:deval][:, data[:target_name]])
    _gini = gini(p_eval, data[:deval][:, data[:target_name]])
    res = Dict(:model_type => "neurotrees", :hyper_id => i, :train_time => train_time, :best_nround => m.info[:logger][:best_iter], :mse => _mse, :gini => _gini, hyper...)
    push!(results, res)
end
results_df = DataFrame(results)
select!(results_df, result_vars, Not(result_vars))
CSV.write(joinpath("results", data_name, "neurotrees.csv"), results_df)

best_hyper = findmin(results_df.mse)[2]
m = models[best_hyper]
p_test = m(dtest) .* _std .+ _mean
_mse = mse(p_test, data[:dtest][:, data[:target_name]])
_gini = gini(p_test, data[:dtest][:, data[:target_name]])
_results_test = copy(results[best_hyper])
push!(_results_test, :mse => _mse, :gini => _gini)
push!(results_test, _results_test)
push!(preds, "neurotrees" => p_test)

################################
# EvoTrees
################################
dtrain = data[:dtrain]
deval = data[:deval]
dtest = data[:dtest]
feature_names = data[:feature_names]
target_name = data[:target_name]

hyper_list = MLBenchmarks.get_hyper_evotrees(loss="mse", metric="mse", nrounds=2000, early_stopping_rounds=10, eta=0.1, max_depth=5:2:11, rowsample=[0.4, 0.6, 0.8, 1.0], colsample=[0.4, 0.6, 0.8, 1.0], L2=[0, 1, 10])
hyper_list = sample(hyper_list, hyper_size, replace=false)

results = Dict{Symbol,Any}[]
models = Vector()
for (i, hyper) in enumerate(hyper_list)
    config = EvoTrees.EvoTreeRegressor(; hyper...)
    train_time = @elapsed m, logger = EvoTrees.fit_evotree(config, dtrain; deval, fnames=feature_names, target_name, metric=hyper[:metric], early_stopping_rounds=hyper[:early_stopping_rounds], print_every_n=10, return_logger=true)
    push!(models, m)
    p_eval = EvoTrees.predict(m, deval)
    _mse = mse(p_eval, data[:deval][:, data[:target_name]])
    _gini = gini(p_eval, data[:deval][:, data[:target_name]])
    res = Dict(:model_type => "evotrees", :train_time => train_time, :best_nround => logger[:best_iter], :mse => _mse, :gini => _gini, hyper...)
    push!(results, res)
end
results_df = DataFrame(results)
select!(results_df, result_vars, Not(result_vars))
CSV.write(joinpath("results", data_name, "evotrees.csv"), results_df)

best_hyper = findmin(results_df.mse)[2]
m = models[best_hyper]
p_test = EvoTrees.predict(m, dtest)
_mse = mse(p_test, data[:dtest][:, data[:target_name]])
_gini = gini(p_test, data[:dtest][:, data[:target_name]])
_results_test = copy(results[best_hyper])
push!(_results_test, :mse => _mse, :gini => _gini)
push!(results_test, _results_test)
push!(preds, "evotrees" => p_test)

################################
# XGBoost
################################
dtrain = XGBoost.DMatrix(data[:dtrain][:, data[:feature_names]], data[:dtrain][:, data[:target_name]])
deval = XGBoost.DMatrix(data[:deval][:, data[:feature_names]], data[:deval][:, data[:target_name]])
dtest = XGBoost.DMatrix(data[:dtest][:, data[:feature_names]])

hyper_list = MLBenchmarks.get_hyper_xgboost(objective="reg:squarederror", eval_metric="rmse", num_round=2000, early_stopping_rounds=10, eta=0.1, max_depth=4:2:10, subsample=[0.4, 0.6, 0.8, 1.0], colsample_bytree=[0.4, 0.6, 0.8, 1.0], lambda=[0, 1, 10])
hyper_list = sample(hyper_list, hyper_size, replace=false)

results = Dict{Symbol,Any}[]
models = Vector()
for (i, hyper) in enumerate(hyper_list)
    train_time = @elapsed m = XGBoost.xgboost(dtrain, watchlist=OrderedDict(["eval" => deval]); hyper...)
    push!(models, m)
    p_eval = XGBoost.predict(m, deval)
    _mse = mse(p_eval, data[:deval][:, data[:target_name]])
    _gini = gini(p_eval, data[:deval][:, data[:target_name]])
    res = Dict(:model_type => "xgboost", :train_time => train_time, :best_nround => m.best_iteration, :mse => _mse, :gini => _gini, hyper...)
    push!(results, res)
end
results_df = DataFrame(results)
select!(results_df, result_vars, Not(result_vars))
CSV.write(joinpath("results", data_name, "xgboost.csv"), results_df)

best_hyper = findmin(results_df.mse)[2]
m = models[best_hyper]
p_test = XGBoost.predict(m, dtest)
_mse = mse(p_test, data[:dtest][:, data[:target_name]])
_gini = gini(p_test, data[:dtest][:, data[:target_name]])
_results_test = copy(results[best_hyper])
push!(_results_test, :mse => _mse, :gini => _gini)
push!(results_test, _results_test)
push!(preds, "xgboost" => p_test)

################################
# LightGBM
################################
dtrain, ytrain = Matrix(data[:dtrain][:, data[:feature_names]]), data[:dtrain][:, data[:target_name]]
deval, yeval = Matrix(data[:deval][:, data[:feature_names]]), data[:deval][:, data[:target_name]]
dtest = Matrix(data[:dtest][:, data[:feature_names]])

hyper_list = MLBenchmarks.get_hyper_lgbm(objective="mse", metric=["mse"], num_iterations=2000, early_stopping_round=10, learning_rate=0.05, num_leaves=[32, 128, 512, 2048], bagging_fraction=[0.3, 0.6, 0.9], feature_fraction=[0.5, 0.9], lambda_l2=[0, 1, 10])
hyper_list = sample(hyper_list, hyper_size, replace=false)

results = Dict{Symbol,Any}[]
models = Vector()
for (i, hyper) in enumerate(hyper_list)
    m = LightGBM.LGBMRegression(; hyper...)
    train_time = @elapsed res = LightGBM.fit!(m, dtrain, ytrain, (deval, yeval))
    push!(models, m)
    p_eval = vec(LightGBM.predict(m, deval))
    _mse = mse(p_eval, data[:deval][:, data[:target_name]])
    _gini = gini(p_eval, data[:deval][:, data[:target_name]])
    res = Dict(:model_type => "lightgbm", :train_time => train_time, :best_nround => res["best_iter"], :mse => _mse, :gini => _gini, hyper...)
    push!(results, res)
end
results_df = DataFrame(results)
select!(results_df, result_vars, Not(result_vars))
CSV.write(joinpath("results", data_name, "lightgbm.csv"), results_df)

best_hyper = findmin(results_df.mse)[2]
m = models[best_hyper]
p_test = vec(LightGBM.predict(m, dtest))
_mse = mse(p_test, data[:dtest][:, data[:target_name]])
_gini = gini(p_test, data[:dtest][:, data[:target_name]])
_results_test = copy(results[best_hyper])
push!(_results_test, :mse => _mse, :gini => _gini)
push!(results_test, _results_test)
push!(preds, "lightgbm" => p_test)

################################
# CatBoost
################################
using PythonCall
dtrain = CatBoost.Pool(data[:dtrain][:, data[:feature_names]], label=PyList(data[:dtrain][:, data[:target_name]]))
deval = CatBoost.Pool(data[:deval][:, data[:feature_names]], label=PyList(data[:deval][:, data[:target_name]]))
dtest = CatBoost.Pool(data[:dtest][:, data[:feature_names]])

hyper_list = MLBenchmarks.get_hyper_catboost(objective="RMSE", eval_metric="RMSE", iterations=2000, early_stopping_rounds=10, learning_rate=0.1, max_depth=4:2:10, subsample=[0.3, 0.6, 0.9], rsm=[0.5, 0.9], reg_lambda=[0, 1, 10])
hyper_list = sample(hyper_list, hyper_size, replace=false)

# model = CatBoost.CatBoostRegressor(; hyper_list[1]...)
# res = CatBoost.fit!(model, dtrain; eval_set=deval)
# p_test = CatBoost.predict(model, dtest)

results = Dict{Symbol,Any}[]
models = Vector()
for (i, hyper) in enumerate(hyper_list)
    m = CatBoost.CatBoostRegressor(; hyper...)
    train_time = @elapsed res = CatBoost.fit!(m, dtrain; eval_set=deval)
    push!(models, m)
    p_eval = CatBoost.predict(m, deval)
    _mse = mse(p_eval, data[:deval][:, data[:target_name]])
    _gini = gini(p_eval, data[:deval][:, data[:target_name]])
    res = Dict(:model_type => "catboost", :train_time => train_time, :best_nround => pyconvert(Int, res.best_iteration_), :mse => _mse, :gini => _gini, hyper...)
    push!(results, res)
end
results_df = DataFrame(results)
select!(results_df, result_vars, Not(result_vars))
CSV.write(joinpath("results", data_name, "catboost.csv"), results_df)

best_hyper = findmin(results_df.mse)[2]
m = models[best_hyper]
p_test = CatBoost.predict(m, dtest)
_mse = mse(p_test, data[:dtest][:, data[:target_name]])
_gini = gini(p_test, data[:dtest][:, data[:target_name]])
_results_test = copy(results[best_hyper])
push!(_results_test, :mse => _mse, :gini => _gini)
push!(results_test, _results_test)
push!(preds, "catboost" => p_test)


################################
# aggregate test results
################################
df = map(results_test) do x
    DataFrame(x)[!, [:model_type, :train_time, :mse, :gini]]
end
df = vcat(df...)
CSV.write(joinpath("results", data_name, "summary.csv"), df)

################################
# correlations
################################
using Statistics: cor
using PlotlyLight
using PlotlyKaleido
PlotlyKaleido.start()

preds_df = DataFrame(preds)[!, ["neurotrees", "evotrees", "xgboost", "lightgbm", "catboost"]]
cors = cor(Matrix(preds_df))
p = plot.heatmap(; z=cors, x=names(preds_df), y=names(preds_df), colorscale="Viridis")
PlotlyKaleido.savefig((; data=p.data, p.layout, p.config), joinpath("results", data_name, "corr.png"))
PlotlyKaleido.kill_kaleido()
