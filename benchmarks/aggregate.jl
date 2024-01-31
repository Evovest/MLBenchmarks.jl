using CSV
using DataFrames
using PrettyTables
using StatsBase: tiedrank, ordinalrank

model_types = [
    "neurotrees",
    "evotrees",
    "xgboost",
    "lightgbm",
    "catboost"
]

data_names = [
    "boston/raw",
    "titanic/raw",
    "year/raw",
    "msrank/raw",
    "yahoo/raw",
    "higgs/raw"
]

data_name = data_names[6]

# metric_vars = ["mse", "gini"]
metric_vars = ["logloss", "accuracy"]
# metric_vars = ["mse", "ndcg"]
metric_var = first(metric_vars)
# var_names = ["model_type", "train_time", "best_nround", metric_vars...]
var_names = ["model_type", "train_time", metric_vars...]
var_nums = ["train_time", metric_vars...]

df = DataFrame()
for algo in model_types
    path = joinpath(@__DIR__, "../results/$data_name/$algo.csv")
    _df = CSV.read(path, DataFrame)
    select!(_df, var_names)
    # append!(df, _df; cols=:union)
    append!(df, _df; cols=:intersect)
end

transform!(groupby(df, :model_type), metric_var => ordinalrank => :rank)
subset!(groupby(df, :model_type), :rank => (x -> x .== minimum(x)))
select!(df, :model_type, var_nums .=> (x -> round.(x, sigdigits=3)) .=> var_nums)
pretty_table(df; backend=Val(:markdown), show_subheader=false, alignment = :c)
