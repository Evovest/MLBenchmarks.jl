function _sberbank_prepare_features!(df, target::AbstractString)
    for cn in names(df)
        cn == target && continue
        col = df[!, cn]
        if nonmissingtype(eltype(col)) <: Real
            m = median(skipmissing(col))
            transform!(df, cn => (x -> Float64.(coalesce.(x, m))) => cn)
        else
            transform!(df, cn => (x -> coalesce.(string.(x), "⟨missing⟩")) => cn)
            transform!(df, cn => categorical => cn)
            transform!(df, cn => ByRow(levelcode) => cn)
        end
    end
    disallowmissing!(df)
    return nothing
end

function data_recipe(
    ::Type{Dataset{:sberbank}},
    df;
    eval_perc=0.15,
    test_perc=0.15,
    seed=123,
    uniformize=false,
    kwargs...,
)
    rng = Xoshiro(seed)

    target_name = "price_doc"
    transform!(df, :price_doc => (x -> parse.(Float64, string.(x))) => :price_doc)
    transform!(df, :price_doc => (x -> (x .- mean(x)) ./ std(x)) => :price_doc)

    _sberbank_prepare_features!(df, target_name)

    idx = randperm(rng, nrow(df))
    eval_cut = floor(Int, (1 - eval_perc - test_perc) * nrow(df))
    test_cut = floor(Int, (1 - test_perc) * nrow(df))

    dtrain = df[view(idx, 1:eval_cut), :]
    deval = df[view(idx, eval_cut+1:test_cut), :]
    dtest = df[view(idx, test_cut+1:end), :]

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
