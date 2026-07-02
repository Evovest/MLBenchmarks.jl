function data_recipe(
    ::Type{Dataset{:allstate_claims}},
    df;
    eval_perc=0.15,
    test_perc=0.15,
    seed=123,
    uniformize=false,
    kwargs...,
)
    rng = Xoshiro(seed)

    transform!(df, :loss => (x -> parse.(Float64, string.(x))) => :loss)
    transform!(df, :loss => (x -> (x .- mean(x)) ./ std(x)) => :loss)

    for cn in names(df)
        cn == "loss" && continue
        if startswith(cn, "cat") || startswith(cn, "cont")
            transform!(df, cn => (x -> parse.(Float64, string.(x))) => cn)
        end
    end
    select!(df, Not("cat77"))

    disallowmissing!(df)

    idx = randperm(rng, nrow(df))
    eval_cut = floor(Int, (1 - eval_perc - test_perc) * nrow(df))
    test_cut = floor(Int, (1 - test_perc) * nrow(df))

    dtrain = df[view(idx, 1:eval_cut), :]
    deval = df[view(idx, eval_cut+1:test_cut), :]
    dtest = df[view(idx, test_cut+1:end), :]

    target_name = "loss"
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
        loss=:mse,
        metric=:mse,
        metrics=[:mse, :gini],
        dtrain,
        deval,
        dtest,
        target_name,
        feature_names)
end
