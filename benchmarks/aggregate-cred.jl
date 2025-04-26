using CSV
using DataFrames
using PrettyTables
using StatsBase: tiedrank, ordinalrank

data_names = [
    "boston/raw",
    # "titanic/raw",
    "year/raw",
    "msrank/raw",
    "yahoo/raw/no-null",
    # "higgs/raw"
]

metric_dict = Dict(
    "boston/raw" => ["mse", "gini"],
    "titanic/raw" => ["logloss", "accuracy"],
    "year/raw" => ["mse", "gini"],
    "msrank/raw" => ["mse", "ndcg"],
    "yahoo/raw/no-null" => ["mse", "ndcg"],
    "higgs/raw" => ["logloss", "accuracy"]
)
# data_name = data_names[6]

results = DataFrame()
for data_name in data_names
    metrics = metric_dict[data_name]
    metric = first(metrics)

    path = joinpath(@__DIR__, "..", "results", "$data_name", "evotrees-mse.csv")
    df = CSV.read(path, DataFrame)
    transform!(groupby(df, :model_type), metric => ordinalrank => :rank)
    subset!(groupby(df, :model_type), :rank => (x -> x .== minimum(x)))
    select!(df, metrics .=> (x -> round.(x, sigdigits=3)) .=> metrics)
    res = stack(df; variable_name="metric", value_name="mse")

    path = joinpath(@__DIR__, "..", "results", "$data_name", "evotrees-mae.csv")
    df = CSV.read(path, DataFrame)
    transform!(groupby(df, :model_type), metric => ordinalrank => :rank)
    subset!(groupby(df, :model_type), :rank => (x -> x .== minimum(x)))
    select!(df, metrics .=> (x -> round.(x, sigdigits=3)) .=> metrics)
    df = stack(df; variable_name="metric", value_name="mae")
    leftjoin!(res, df; on="metric")

    path = joinpath(@__DIR__, "..", "results", "$data_name", "evotrees-cred_var.csv")
    df = CSV.read(path, DataFrame)
    transform!(groupby(df, :model_type), metric => ordinalrank => :rank)
    subset!(groupby(df, :model_type), :rank => (x -> x .== minimum(x)))
    select!(df, metrics .=> (x -> round.(x, sigdigits=3)) .=> metrics)
    df = stack(df; variable_name="metric", value_name="cred_var")
    leftjoin!(res, df; on="metric")

    path = joinpath(@__DIR__, "..", "results", "$data_name", "evotrees-cred_std.csv")
    df = CSV.read(path, DataFrame)
    transform!(groupby(df, :model_type), metric => ordinalrank => :rank)
    subset!(groupby(df, :model_type), :rank => (x -> x .== minimum(x)))
    select!(df, metrics .=> (x -> round.(x, sigdigits=3)) .=> metrics)
    df = stack(df; variable_name="metric", value_name="cred_std")
    leftjoin!(res, df; on="metric")

    insertcols!(res, 1, "model" => first(split(data_name, "/")))
    append!(results, res; cols=:union)
end

pretty_table(results; backend=Val(:markdown), show_subheader=false, alignment=:c)
