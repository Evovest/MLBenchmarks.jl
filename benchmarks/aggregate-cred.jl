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

    path = joinpath(@__DIR__, "..", "results", "$data_name", "evotrees-ref.csv")
    ref = CSV.read(path, DataFrame)
    transform!(groupby(ref, :model_type), metric => ordinalrank => :rank)
    subset!(groupby(ref, :model_type), :rank => (x -> x .== minimum(x)))
    select!(ref, metrics .=> (x -> round.(x, sigdigits=3)) .=> metrics)
    ref = stack(ref; variable_name="metric", value_name="ref")

    path = joinpath(@__DIR__, "..", "results", "$data_name", "evotrees-cred_var.csv")
    cred = CSV.read(path, DataFrame)
    transform!(groupby(cred, :model_type), metric => ordinalrank => :rank)
    subset!(groupby(cred, :model_type), :rank => (x -> x .== minimum(x)))
    select!(cred, metrics .=> (x -> round.(x, sigdigits=3)) .=> metrics)
    cred = stack(cred; variable_name="metric", value_name="cred_var")
    leftjoin!(ref, cred; on="metric")

    path = joinpath(@__DIR__, "..", "results", "$data_name", "evotrees-cred_std.csv")
    cred = CSV.read(path, DataFrame)
    transform!(groupby(cred, :model_type), metric => ordinalrank => :rank)
    subset!(groupby(cred, :model_type), :rank => (x -> x .== minimum(x)))
    select!(cred, metrics .=> (x -> round.(x, sigdigits=3)) .=> metrics)
    cred = stack(cred; variable_name="metric", value_name="cred_std")
    leftjoin!(ref, cred; on="metric")

    # path = joinpath(@__DIR__, "..", "results", "$data_name", "evotrees-credV1B.csv")
    # cred = CSV.read(path, DataFrame)
    # transform!(groupby(cred, :model_type), metric => ordinalrank => :rank)
    # subset!(groupby(cred, :model_type), :rank => (x -> x .== minimum(x)))
    # select!(cred, metrics .=> (x -> round.(x, sigdigits=3)) .=> metrics)
    # cred = stack(cred; variable_name="metric", value_name="credV1B")
    # leftjoin!(ref, cred; on="metric")

    # path = joinpath(@__DIR__, "..", "results", "$data_name", "evotrees-credV2B.csv")
    # cred = CSV.read(path, DataFrame)
    # transform!(groupby(cred, :model_type), metric => ordinalrank => :rank)
    # subset!(groupby(cred, :model_type), :rank => (x -> x .== minimum(x)))
    # select!(cred, metrics .=> (x -> round.(x, sigdigits=3)) .=> metrics)
    # cred = stack(cred; variable_name="metric", value_name="credV2B")
    # leftjoin!(ref, cred; on="metric")

    insertcols!(ref, 1, "model" => first(split(data_name, "/")))
    append!(results, ref; cols=:union)
end

pretty_table(results; backend=Val(:markdown), show_subheader=false, alignment=:c)
