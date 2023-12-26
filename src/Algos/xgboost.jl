function get_hyper_xgboost(;
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist", # hist/gpu_hist
    num_round=500,
    early_stopping_rounds=5,
    eta=0.3,
    max_depth=6,
    subsample=0.5,
    colsample_bytree=0.5,
    max_bin=64,
    lambda=1,
)

    # tunable = [:eta, :max_depth, :subsample, :colsample_bytree, :lambda, :max_bin]
    hyper_list = Dict{Symbol,Any}[]

    for _eta in eta, _max_depth in max_depth, _subsample in subsample, _colsample_bytree in colsample_bytree, _lambda in lambda, _max_bin in max_bin

        hyper = Dict(
            :objective => objective,
            :eval_metric => eval_metric,
            :tree_method => tree_method, # hist/gpu_hist
            :num_round => num_round,
            :early_stopping_rounds => early_stopping_rounds,
            :eta => _eta,
            :max_depth => _max_depth,
            :subsample => _subsample,
            :colsample_bytree => _colsample_bytree,
            :lambda => _lambda,
            :max_bin => _max_bin
        )

        push!(hyper_list, hyper)
    end

    return hyper_list

end