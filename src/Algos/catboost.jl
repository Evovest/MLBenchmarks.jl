function get_hyper_catboost(;
    objective="RMSE",
    eval_metric="RMSE",
    iterations=500,
    early_stopping_rounds=5,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.5,
    rsm=0.5,
    reg_lambda=1,
)

    # tunable = [:eta, :max_depth, :subsample, :colsample_bytree, :lambda, :max_bin]
    hyper_list = Dict{Symbol,Any}[]

    for _learning_rate in learning_rate, _max_depth in max_depth, _subsample in subsample, _rsm in rsm, _reg_lambda in reg_lambda

        hyper = Dict(
            :objective => objective,
            :eval_metric => eval_metric,
            :iterations => iterations,
            :early_stopping_rounds => early_stopping_rounds,
            :learning_rate => _learning_rate,
            :max_depth => _max_depth,
            :subsample => _subsample,
            :rsm => _rsm,
            :reg_lambda => _reg_lambda,
        )

        push!(hyper_list, hyper)
    end

    return hyper_list

end