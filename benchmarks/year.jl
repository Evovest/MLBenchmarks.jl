
abstract type Data{T} end

Data{:allo}


function load_data(::Type{Data{:salut}})
    return 2
end

load_data(Data{:salut})


# function load_data(::Allo{:salut})
#     return 1
# end

function load_data(x::Symbol)
    data = load_data(Data{x})
    return data
end
load_data(:salut)

load_data(:higgs)

