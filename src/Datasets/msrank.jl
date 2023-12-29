function load_data(::Type{Dataset{:msrank}}; uniformize=false, aws_config=AWSConfig(), kwargs...)

    train_raw = read_libsvm_aws("share/data/msrank/train.txt"; has_query=true, aws_config)
    eval_raw = read_libsvm_aws("share/data/msrank/vali.txt"; has_query=true, aws_config)
    test_raw = read_libsvm_aws("share/data/msrank/test.txt"; has_query=true, aws_config)

    dtrain = DataFrame(train_raw[:x], :auto)
    dtrain.q = train_raw[:q]
    dtrain.y = train_raw[:y]
    dtrain.y_scale = dtrain.y ./ 4

    deval = DataFrame(eval_raw[:x], :auto)
    deval.q = eval_raw[:q]
    deval.y = eval_raw[:y]
    deval.y_scale = deval.y ./ 4

    dtest = DataFrame(test_raw[:x], :auto)
    dtest.q = test_raw[:q]
    dtest.y = test_raw[:y]
    dtest.y_scale = dtest.y ./ 4

    feature_names = setdiff(names(dtrain), ["q", "y", "y_scale"])
    target_name = "y"

    if uniformize
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