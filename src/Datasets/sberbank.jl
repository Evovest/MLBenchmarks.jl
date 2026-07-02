function _iso_date_to_unix(s)
    s = string(s)
    y = parse(Int, s[1:4])
    m = parse(Int, s[6:7])
    d = parse(Int, s[9:10])
    if m < 3
        y -= 1
        m += 12
    end
    era = y ÷ 400
    yoe = y - 400 * era
    doy = (153 * (m - 3) + 2) ÷ 5 + d - 1
    doe = yoe * 365 + yoe ÷ 4 - yoe ÷ 100 + doy
    days = era * 146097 + doe - 719468
    return Float32(days * 86_400)
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
    target_name = "price_doc"

    transform!(df, :timestamp => (x -> _iso_date_to_unix.(x)) => :t)
    transform!(df, :price_doc => (x -> parse.(Float64, string.(x))) => :price_doc)
    transform!(df, :price_doc => (x -> (x .- mean(x)) ./ std(x)) => :price_doc)
    select!(df, Not([:timestamp, :id]))

    for cn in names(df)
        cn in (target_name, "t") && continue
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

    sort!(df, :t)
    n = nrow(df)
    eval_cut = floor(Int, (1 - eval_perc - test_perc) * n)
    test_cut = floor(Int, (1 - test_perc) * n)

    dtrain = df[1:eval_cut, :]
    deval = df[eval_cut+1:test_cut, :]
    dtest = df[test_cut+1:end, :]

    t_mean = mean(dtrain.t)
    t_std_scale = std(dtrain.t)
    for part in (dtrain, deval, dtest)
        transform!(part, :t => (x -> Float32.((x .- t_mean) ./ t_std_scale)) => :t_std)
    end

    feature_names = setdiff(names(dtrain), [target_name, "t"])

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
        feature_names,
    )
end
