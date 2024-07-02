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

using Random: seed!
seed!(123)

uniformize = false

data_name = uniformize ? "higgs/norm" : "higgs/raw"
data = load_data(:higgs; uniformize, aws_config)
result_vars = [:model_type, :train_time, :best_nround, :logloss, :accuracy]
hyper_size = 8

preds = Dict{String,Vector}()
results_test = Dict{Symbol,Any}[]

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

# sort(unique(data[:deval].Column3))
# sort(unique(dtrain.Column2))
# PlotlyLight.Plot(; x=data[:deval].Column29, type="histogram")
# PlotlyLight.Plot(; x=dtrain.Column2, type="histogram")
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
device = :gpu

hyper_list = MLBenchmarks.get_hyper_neurotrees(; loss=:logloss, metric=:logloss, nrounds=200, early_stopping_rounds=5, lr=5e-3, ntrees=[32, 64, 128, 256], stack_size=[2, 3], depth=[3], hidden_size=[16, 24, 32], init_scale=0.0, batchsize)
hyper_list = sample(hyper_list, hyper_size, replace=false)

results = Dict{Symbol,Any}[]
models = Vector()

# warmup
hyper = copy(first(hyper_list))
hyper[:nrounds] = 1
config = NeuroTreeModels.NeuroTreeRegressor(; hyper...)
NeuroTreeModels.fit(config, dtrain; deval, feature_names, target_name=target_name, metric=hyper[:metric], early_stopping_rounds=hyper[:early_stopping_rounds], print_every_n=10, device)

for (i, hyper) in enumerate(hyper_list)
    @info "Loop $i"
    config = NeuroTreeModels.NeuroTreeRegressor(; hyper...)
    train_time = @elapsed m = NeuroTreeModels.fit(config, dtrain; deval, feature_names, target_name, metric=hyper[:metric], early_stopping_rounds=hyper[:early_stopping_rounds], print_every_n=10, device)
    push!(models, m)
    p_eval = m(deval)
    _logloss = logloss(p_eval, data[:deval][:, data[:target_name]])
    _accuracy = accuracy(p_eval, data[:deval][:, data[:target_name]])
    res = Dict(:model_type => "neurotrees", :train_time => train_time, :best_nround => m.info[:logger][:best_iter], :logloss => _logloss, :accuracy => _accuracy, hyper...)
    push!(results, res)
end
results_df = DataFrame(results)
select!(results_df, result_vars, Not(result_vars))
CSV.write(joinpath("results", data_name, "neurotrees.csv"), results_df)

best_hyper = findmin(results_df.logloss)[2]
m = models[best_hyper]
p_test = m(dtest)
_logloss = logloss(p_test, data[:dtest][:, data[:target_name]])
_accuracy = accuracy(p_test, data[:dtest][:, data[:target_name]])
_results_test = copy(results[best_hyper])
push!(_results_test, :logloss => _logloss, :accuracy => _accuracy)
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

hyper_list = MLBenchmarks.get_hyper_evotrees(loss="logloss", metric="logloss", nrounds=12000, early_stopping_rounds=50, eta=0.1, max_depth=7:2:11, rowsample=[0.4, 0.6, 0.8, 1.0], colsample=[0.5, 0.7, 0.9, 1.0], L2=[0, 1, 10])
hyper_list = sample(hyper_list, hyper_size, replace=false)

results = Dict{Symbol,Any}[]
models = Vector()

# warmup
hyper = copy(first(hyper_list))
hyper[:nrounds] = 1
config = EvoTrees.EvoTreeRegressor(; hyper...)
EvoTrees.fit_evotree(config, dtrain; deval, fnames=feature_names, target_name, metric=hyper[:metric], early_stopping_rounds=hyper[:early_stopping_rounds], print_every_n=10, return_logger=true)

for (i, hyper) in enumerate(hyper_list)
    config = EvoTrees.EvoTreeRegressor(; hyper...)
    train_time = @elapsed m, logger = EvoTrees.fit_evotree(config, dtrain; deval, fnames=feature_names, target_name, metric=hyper[:metric], early_stopping_rounds=hyper[:early_stopping_rounds], print_every_n=10, return_logger=true)
    push!(models, m)
    p_eval = EvoTrees.predict(m, deval)
    _logloss = logloss(p_eval, data[:deval][:, data[:target_name]])
    _accuracy = accuracy(p_eval, data[:deval][:, data[:target_name]])
    res = Dict(:model_type => "evotrees", :train_time => train_time, :best_nround => logger[:best_iter], :logloss => _logloss, :accuracy => _accuracy, hyper...)
    push!(results, res)
end
results_df = DataFrame(results)
select!(results_df, result_vars, Not(result_vars))
CSV.write(joinpath("results", data_name, "evotrees.csv"), results_df)

best_hyper = findmin(results_df.logloss)[2]
m = models[best_hyper]
p_test = EvoTrees.predict(m, dtest)
_logloss = logloss(p_test, data[:dtest][:, data[:target_name]])
_accuracy = accuracy(p_test, data[:dtest][:, data[:target_name]])
_results_test = copy(results[best_hyper])
push!(_results_test, :logloss => _logloss, :accuracy => _accuracy)
push!(results_test, _results_test)
push!(preds, "evotrees" => p_test)

################################
# XGBoost
################################
dtrain = XGBoost.DMatrix(data[:dtrain][:, data[:feature_names]], data[:dtrain][:, data[:target_name]])
deval = XGBoost.DMatrix(data[:deval][:, data[:feature_names]], data[:deval][:, data[:target_name]])
dtest = XGBoost.DMatrix(data[:dtest][:, data[:feature_names]])

hyper_list = MLBenchmarks.get_hyper_xgboost(objective="reg:logistic", eval_metric="logloss", num_round=12000, early_stopping_rounds=50, eta=0.1, max_depth=6:2:10, subsample=[0.4, 0.6, 0.8, 1.0], colsample_bytree=[0.5, 0.7, 0.9, 1.0], lambda=[0, 1, 10])
hyper_list = sample(hyper_list, hyper_size, replace=false)

results = Dict{Symbol,Any}[]
models = Vector()
for (i, hyper) in enumerate(hyper_list)
    train_time = @elapsed m = XGBoost.xgboost(dtrain, watchlist=OrderedDict(["eval" => deval]); hyper...)
    push!(models, m)
    p_eval = XGBoost.predict(m, deval)
    _logloss = logloss(p_eval, data[:deval][:, data[:target_name]])
    _accuracy = accuracy(p_eval, data[:deval][:, data[:target_name]])
    res = Dict(:model_type => "xgboost", :train_time => train_time, :best_nround => m.best_iteration, :logloss => _logloss, :accuracy => _accuracy, hyper...)
    push!(results, res)
end
results_df = DataFrame(results)
select!(results_df, result_vars, Not(result_vars))
CSV.write(joinpath("results", data_name, "xgboost.csv"), results_df)

best_hyper = findmin(results_df.logloss)[2]
m = models[best_hyper]
p_test = XGBoost.predict(m, dtest)
_logloss = logloss(p_test, data[:dtest][:, data[:target_name]])
_accuracy = accuracy(p_test, data[:dtest][:, data[:target_name]])
_results_test = copy(results[best_hyper])
push!(_results_test, :logloss => _logloss, :accuracy => _accuracy)
push!(results_test, _results_test)
push!(preds, "xgboost" => p_test)

################################
# LightGBM
################################
dtrain, ytrain = Matrix(data[:dtrain][:, data[:feature_names]]), data[:dtrain][:, data[:target_name]]
deval, yeval = Matrix(data[:deval][:, data[:feature_names]]), data[:deval][:, data[:target_name]]
dtest = Matrix(data[:dtest][:, data[:feature_names]])

hyper_list = MLBenchmarks.get_hyper_lgbm(objective="cross_entropy", metric=["logloss"], num_iterations=12000, early_stopping_round=50, learning_rate=0.1, num_leaves=[128, 512, 2048], bagging_fraction=[0.3, 0.6, 0.9], feature_fraction=[0.5, 0.7, 0.9, 1.0], lambda_l2=[0, 1, 10])
hyper_list = sample(hyper_list, hyper_size, replace=false)

results = Dict{Symbol,Any}[]
models = Vector()
for (i, hyper) in enumerate(hyper_list)
    m = LightGBM.LGBMRegression(; hyper...)
    train_time = @elapsed res = LightGBM.fit!(m, dtrain, ytrain, (deval, yeval))
    push!(models, m)
    p_eval = vec(LightGBM.predict(m, deval))
    _logloss = logloss(p_eval, data[:deval][:, data[:target_name]])
    _accuracy = accuracy(p_eval, data[:deval][:, data[:target_name]])
    res = Dict(:model_type => "lightgbm", :train_time => train_time, :best_nround => res["best_iter"], :logloss => _logloss, :accuracy => _accuracy, hyper...)
    push!(results, res)
end
results_df = DataFrame(results)
select!(results_df, result_vars, Not(result_vars))
CSV.write(joinpath("results", data_name, "lightgbm.csv"), results_df)

best_hyper = findmin(results_df.logloss)[2]
m = models[best_hyper]
p_test = vec(LightGBM.predict(m, dtest))
_logloss = logloss(p_test, data[:dtest][:, data[:target_name]])
_accuracy = accuracy(p_test, data[:dtest][:, data[:target_name]])
_results_test = copy(results[best_hyper])
push!(_results_test, :logloss => _logloss, :accuracy => _accuracy)
push!(results_test, _results_test)
push!(preds, "lightgbm" => p_test)


################################
# CatBoost
################################
using PythonCall
dtrain = CatBoost.Pool(data[:dtrain][:, data[:feature_names]], label=PyList(data[:dtrain][:, data[:target_name]]))
deval = CatBoost.Pool(data[:deval][:, data[:feature_names]], label=PyList(data[:deval][:, data[:target_name]]))
dtest = CatBoost.Pool(data[:dtest][:, data[:feature_names]])

hyper_list = MLBenchmarks.get_hyper_catboost(objective="Logloss", eval_metric="Logloss", iterations=12000, early_stopping_rounds=50, learning_rate=0.1, max_depth=6:2:10, subsample=[0.3, 0.6, 0.9], rsm=[0.5, 0.7, 0.9, 1.0], reg_lambda=[0, 1, 10])
hyper_list = sample(hyper_list, hyper_size, replace=false)

results = Dict{Symbol,Any}[]
models = Vector()
for (i, hyper) in enumerate(hyper_list)
    m = CatBoost.CatBoostClassifier(; hyper...)
    train_time = @elapsed res = CatBoost.fit!(m, dtrain; eval_set=deval)
    push!(models, m)
    p_eval = CatBoost.predict(m, deval; prediction_type="Probability")[:, 2]
    _logloss = logloss(p_eval, data[:deval][:, data[:target_name]])
    _accuracy = accuracy(p_eval, data[:deval][:, data[:target_name]])
    res = Dict(:model_type => "catboost", :train_time => train_time, :best_nround => pyconvert(Int, res.best_iteration_), :logloss => _logloss, :accuracy => _accuracy, hyper...)
    push!(results, res)
end
results_df = DataFrame(results)
select!(results_df, result_vars, Not(result_vars))
CSV.write(joinpath("results", data_name, "catboost.csv"), results_df)

best_hyper = findmin(results_df.logloss)[2]
m = models[best_hyper]
p_test = CatBoost.predict(m, dtest; prediction_type="Probability")[:, 2]
_logloss = logloss(p_test, data[:dtest][:, data[:target_name]])
_accuracy = accuracy(p_test, data[:dtest][:, data[:target_name]])
_results_test = copy(results[best_hyper])
push!(_results_test, :logloss => _logloss, :accuracy => _accuracy)
push!(results_test, _results_test)
push!(preds, "catboost" => p_test)

################################
# aggregate test results
################################
df = map(results_test) do x
    DataFrame(x)[!, [:model_type, :train_time, :logloss, :accuracy]]
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
