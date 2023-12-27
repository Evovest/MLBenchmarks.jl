function get_hyper_neurotrees(;
    loss="mse",
    metric="mse",
    device="gpu",
    tree_type="base",
    early_stopping_rounds=5,
    nrounds=500,
    lr=1e-3,
    wd=0.0,
    num_trees=32,
    depth=5,
    batchsize=4096,
    stack_size=0.5,
    hidden_size=8,
    boosting_size=1,
)

    # tunable = [:eta, :max_depth, :subsample, :colsample_bytree, :lambda, :max_bin]
    hyper_list = Dict{Symbol,Any}[]

    for _lr in lr, _wd in wd, _num_trees in num_trees, _depth in depth, _stack_size in stack_size, _hidden_size in hidden_size, _boosting_size in boosting_size

        hyper = Dict(
            :loss => loss,
            :metric => metric,
            :device => device,
            :early_stopping_rounds => early_stopping_rounds,
            :tree_type => tree_type,
            :nrounds => nrounds,
            :lr => _lr,
            :wd => _wd,
            :num_trees => _num_trees,
            :depth => _depth,
            :stack_size => _stack_size,
            :hidden_size => _hidden_size,
            :boosting_size => _boosting_size,
            :batchsize => batchsize
        )

        push!(hyper_list, hyper)
    end

    return hyper_list

end