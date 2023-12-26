using MLBenchmarks

load_data(:year)

function load_data(x::Symbol)
    data = load_data(Data{x})
    return data
end
load_data(:salut)

load_data(:higgs)

