function load_data(::Type{Dataset{:boston}}; uniformize=false, seed=123, kwargs...)

    seed!(seed)

    df = MLDatasets.BostonHousing().dataframe

    # The full data can now be split according to train and eval indices. 
    # Target and feature names are also set.
    train_ratio = 0.7
    eval_ratio = 0.15
    idx = randperm(nrow(df))
    train_idx = idx[1:Int(round(train_ratio * nrow(df)))]
    eval_idx = idx[Int(round(train_ratio * nrow(df)))+1:Int(round((train_ratio + eval_ratio) * nrow(df)))]
    test_idx = idx[Int(round((train_ratio + eval_ratio) * nrow(df)))+1:end]

    dtrain = df[train_idx, :]
    deval = df[eval_idx, :]
    dtest = df[test_idx, :]

    target_name = "MEDV"
    feature_names = setdiff(names(df), [target_name])

    if uniformize

        transform!(dtrain, feature_names .=> ByRow(Float64) .=> feature_names)
        transform!(deval, feature_names .=> ByRow(Float64) .=> feature_names)
        transform!(dtest, feature_names .=> ByRow(Float64) .=> feature_names)

        ops = uniformer(
            dtrain;
            vars_in=feature_names,
            vars_out=feature_names,
            nbins=255,
            min=-1,
            max=1,
        )

        transform!(dtrain, ops)
        transform!(deval, ops)
        transform!(dtest, ops)
    end

    data = Dict(
        :dtrain => dtrain,
        :deval => deval,
        :dtest => dtest,
        :feature_names => feature_names,
        :target_name => target_name
    )

    return data
end