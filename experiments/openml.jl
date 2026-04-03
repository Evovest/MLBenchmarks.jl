using OpenML
using DataFrames
using .Iterators: partition
using EvoTrees
using EvoTrees: fit, predict

data_map = Dict(
    :titanic => 40945,
    :higgs_11M => 45570,
    :higgs_1M => 42769,
    :boston => 531,
    :year => 44027,
    :microsoft => 45579
)

id = data_map[:titanic]
desc = OpenML.describe_dataset(id)
df = OpenML.load(id) |> DataFrame

# load data
# build dataset: 
#   - folds: vector of eidx
#   - dtot
#   - dtest
#   - target_name
#   - feature_names
function get_data(name::Symbol; nfolds=5, test_perc=0.2)
    id = data_map[name]
    desc = OpenML.describe_dataset(id)
    df = OpenML.load(id) |> DataFrame

    idx = randperm(nrow(df))
    idx_cut = floor(Int, (1 - test_perc) * nrow(df))
    dtot = df[idx[1:idx_cut], :]
    dtest = df[view(idx, idx_cut+1:end), :]

    folds = makechunks(1:nrow(dtrain), nfolds)

    # TODO: apply table-specific recipe
    target_name = "X2urvived"
    feature_names = setdiff(names(df), [target_name])

    return (;
        folds=folds,
        dtot=dtot,
        dtest=dtest,
        target_name=target_name,
        feature_names=feature_names)
end

function fit_cv(; data, learner, hyper_list)
    for (fold, eidx) in enumerate(data[:folds])
        dtot = data[:dtot]
        mask = trues(nrow(dtot))
        view(mask, eidx) .= false
        dtrain = view(dtot, mask, :)
        deval = view(dtot, .!mask, :)
        for hyper in hyper_list
            # config = learner(; hyper...)
            config = EvoTreeRegressor()
            m = fit(config, dtrain; deval, data.target_name, data.feature_names)
            p = predict(m, deval)
            metric = mean((p .- deval[!, data.target_name]) .^ 2)
            @info "Fit iteration" fold metric
        end
    end
end

@views function makechunks(X::AbstractVector, n::Integer)
    c = fld(length(X), n)
    return [X[1+c*k:(k == n - 1 ? end : c * k + c)] for k = 0:n-1]
end

learner = EvoTreeRegressor
hyper_list = Dict{Symbol,Any}[]
hyper_list = [Dict{Symbol,Any}(:depth => 4)]
fit_cv(; data, learner, hyper_list)
