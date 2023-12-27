function get_hyper_lgbm(;
    objective="regression",
    metric="rmse",
    num_iterations=100,
    min_data_in_leaf=20,
    num_class=1,
    early_stopping_round=5,
    learning_rate=0.3,
    num_leaves=128,
    max_depth=-1,
    bagging_fraction=0.9,
    feature_fraction=0.8,
    max_bin=64,
    lambda_l2=1,
)

    # tunable = [:eta, :max_depth, :subsample, :colsample_bytree, :lambda, :max_bin]
    hyper_list = Dict{Symbol,Any}[]

    for _learning_rate in learning_rate, _num_leaves in num_leaves, _max_depth in max_depth, _bagging_fraction in bagging_fraction, _feature_fraction in feature_fraction, _lambda_l2 in lambda_l2, _max_bin in max_bin, _min_data_in_leaf in min_data_in_leaf

        hyper = Dict(
            :objective => objective,
            :metric => metric,
            :num_iterations => num_iterations,
            :early_stopping_round => early_stopping_round,
            :learning_rate => _learning_rate,
            :num_leaves => _num_leaves,
            :max_depth => _max_depth,
            :bagging_fraction => _bagging_fraction,
            :feature_fraction => _feature_fraction,
            :lambda_l2 => _lambda_l2,
            :max_bin => _max_bin,
            :min_data_in_leaf => _min_data_in_leaf,
            :num_class => num_class,
        )

        num_class == 0 ? delete!(hyper, :num_class) : nothing
        push!(hyper_list, hyper)
    end

    return hyper_list

end