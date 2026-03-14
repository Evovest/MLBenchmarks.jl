function data_recipe(::Type{Dataset{:boston}}, df; eval_perc=0.15, test_perc=0.15, seed=123, kwargs...)

    idx = randperm(nrow(df))
    eval_cut = floor(Int, (1 - eval_perc - test_perc) * nrow(df))
    test_cut = floor(Int, (1 - test_perc) * nrow(df))

    dtrain = df[view(idx, 1:eval_cut), :]
    deval = df[view(idx, eval_cut+1:test_cut), :]
    dtest = df[view(idx, test_cut+1:end), :]

    target_name = "MEDV"
    feature_names = setdiff(names(df), [target_name])

    return (;
        loss=:mse,
        metric=:mse,
        dtrain,
        deval,
        dtest,
        target_name,
        feature_names)
end
