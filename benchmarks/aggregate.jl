using CSV
using DataFrames
using PrettyTables
using StatsBase: tiedrank, ordinalrank

data_names = [
    "boston",
    "titanic",
    "year",
    "microsoft",
    "higgs_1M",
    "allstate_claims",
    "creditcard",
]

data_name = data_names[7]
dir_path = joinpath(@__DIR__, "..", "results", "$data_name")
csv_files = filter(f -> occursin(r"\.csv$", f), readdir(dir_path, join=true))
dfs = [CSV.read(file, DataFrame) for file in csv_files]
df = vcat(dfs...)

# Determine metric column
metric_candidates = [:eval_mse, :eval_logloss]
metric = first(intersect(metric_candidates, Symbol.(names(df))))

# Keep top-N
df = combine(groupby(df, :model_type)) do sdf
    first(sort(sdf, metric), 1)
end

# Round numeric columns
select!(df, :model_type, :train_time, r"test_")
var_nums = setdiff(names(df), ["model_type"])
select!(df, :model_type, var_nums .=> (x -> round.(x, sigdigits=3)) .=> var_nums)

@info data_name
pretty_table(df; backend=:markdown, column_labels=names(df), alignment=:c)
# pretty_table(df; backend=:markdown, table_format=MarkdownTableFormat(compact_table=true), column_labels=names(df), alignment=:c)
