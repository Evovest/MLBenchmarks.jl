function load_data(::Type{Dataset{:yahoo}}; uniformize=false, incl_null_flag=true, aws_config=AWSConfig(), kwargs...)

    train_raw = read_libsvm_aws("share/data/yahoo-ltrc/set1.train.txt"; has_query=true, aws_config)
    eval_raw = read_libsvm_aws("share/data/yahoo-ltrc/set1.valid.txt"; has_query=true, aws_config)
    test_raw = read_libsvm_aws("share/data/yahoo-ltrc/set1.test.txt"; has_query=true, aws_config)

    colsums_train = map(sum, eachcol(train_raw[:x]))
    colsums_eval = map(sum, eachcol(eval_raw[:x]))
    colsums_test = map(sum, eachcol(test_raw[:x]))

    sum(colsums_train .== 0)
    sum(colsums_test .== 0)
    @assert all((colsums_train .== 0) .== (colsums_test .== 0))
    drop_cols = colsums_train .== 0

    x_train = train_raw[:x][:, .!drop_cols]
    x_eval = eval_raw[:x][:, .!drop_cols]
    x_test = test_raw[:x][:, .!drop_cols]

    #####################################
    # create DataFrames
    #####################################
    dtrain = DataFrame(x_train, :auto)
    dtrain.q .= train_raw[:q]
    dtrain.y .= train_raw[:y]
    dtrain.y_scale .= train_raw[:y] ./ 4

    deval = DataFrame(x_eval, :auto)
    deval.q .= eval_raw[:q]
    deval.y .= eval_raw[:y]
    deval.y_scale .= eval_raw[:y] ./ 4

    dtest = DataFrame(x_test, :auto)
    dtest.q .= test_raw[:q]
    dtest.y .= test_raw[:y]
    dtest.y_scale .= test_raw[:y] ./ 4

    _feature_names = setdiff(names(dtrain), ["q", "y", "y_scale"])

    if incl_null_flag
        _feature_names_miss = _feature_names .* "_miss"

        transform!(dtrain, _feature_names .=> (x -> ifelse.(x .== 0, 0, 1)) .=> _feature_names_miss)
        transform!(dtrain, _feature_names .=> (x -> ifelse.(x .== 0, 0.5, x)) .=> _feature_names)

        transform!(deval, _feature_names .=> (x -> ifelse.(x .== 0, 0, 1)) .=> _feature_names_miss)
        transform!(deval, _feature_names .=> (x -> ifelse.(x .== 0, 0.5, x)) .=> _feature_names)

        transform!(dtest, _feature_names .=> (x -> ifelse.(x .== 0, 0, 1)) .=> _feature_names_miss)
        transform!(dtest, _feature_names .=> (x -> ifelse.(x .== 0, 0.5, x)) .=> _feature_names)

    end

    feature_names = setdiff(names(dtrain), ["q", "y", "y_scale"])
    target_name = "y"

    if uniformize
        ops = uniformer(
            dtrain;
            vars_in=_feature_names,
            vars_out=_feature_names,
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