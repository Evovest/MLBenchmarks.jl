function data_recipe(::Type{Dataset{:higgs_1M}}, df; eval_perc=0.15, test_perc=0.15, seed=123, uniformize=false, kwargs...)
    rng = Xoshiro(seed)

    transform!(df, :target => (x -> parse.(Int, string.(x))) => :target)

    idx = randperm(rng, nrow(df))
    eval_cut = floor(Int, (1 - eval_perc - test_perc) * nrow(df))
    test_cut = floor(Int, (1 - test_perc) * nrow(df))

    dtrain = df[view(idx, 1:eval_cut), :]
    deval = df[view(idx, eval_cut+1:test_cut), :]
    dtest = df[view(idx, test_cut+1:end), :]

    target_name = "target"
    feature_names = setdiff(names(df), [target_name])

    if uniformize
        ops = uniformer(
            dtrain;
            vars_in=feature_names,
            vars_out=feature_names,
            nbins=128,
            min=-1,
            max=1,
        )

        transform!(dtrain, ops)
        transform!(deval, ops)
        transform!(dtest, ops)
    end

    return (;
        loss=:logloss,
        metric=:logloss,
        metrics=[:logloss, :gini],
        dtrain,
        deval,
        dtest,
        target_name,
        feature_names)
end
