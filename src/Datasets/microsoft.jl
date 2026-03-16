function data_recipe(::Type{Dataset{:microsoft}}, df; eval_perc=0.15, test_perc=0.15, seed=123, kwargs...)
    rng = Xoshiro(seed)

    transform!(df, :relevance => (x -> parse.(Float64, string.(x))) => :relevance)
    transform!(df, :relevance => (x -> (x .- mean(x)) ./ std(x)) => :relevance)

    idx = randperm(rng, nrow(df))
    eval_cut = floor(Int, (1 - eval_perc - test_perc) * nrow(df))
    test_cut = floor(Int, (1 - test_perc) * nrow(df))

    dtrain = df[view(idx, 1:eval_cut), :]
    deval = df[view(idx, eval_cut+1:test_cut), :]
    dtest = df[view(idx, test_cut+1:end), :]

    target_name = "relevance"
    feature_names = setdiff(names(df), [target_name, "query_id"])

    return (;
        loss=:mse,
        metric=:mse,
        metrics=[:mse, :gini],
        dtrain,
        deval,
        dtest,
        target_name,
        feature_names)
end
