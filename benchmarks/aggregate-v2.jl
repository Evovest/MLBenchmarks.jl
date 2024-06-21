using CSV
using DataFrames
using PrettyTables
using StatsBase: tiedrank, ordinalrank

data_names = [
    "boston/raw",
    "titanic/raw",
    "year/raw",
    "msrank/raw",
    "yahoo/raw",
    "higgs/raw"
]

data_name = data_names[2]

path = joinpath(@__DIR__, "..", "results", "$data_name", "summary.csv")
df = CSV.read(path, DataFrame)
var_nums = setdiff(names(df), ["model_type"])
select!(df, :model_type, var_nums .=> (x -> round.(x, sigdigits=3)) .=> var_nums)

@info data_name
pretty_table(df; backend=Val(:markdown), show_subheader=false, alignment = :c)
