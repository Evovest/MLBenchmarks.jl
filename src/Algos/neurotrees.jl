function get_hyper_neurotrees(;
    loss="mse",
    metric="mse",
    device="gpu",
    tree_type="base",
    early_stopping_rounds=5,
    nrounds=200,
    lr=1e-3,
    wd=0.0,
    ntrees=64,
    depth=4,
    batchsize=2048,
    stack_size=1,
    hidden_size=1,
)

    # tunable = [:eta, :max_depth, :subsample, :colsample_bytree, :lambda, :max_bin]
    hyper_list = Dict{Symbol,Any}[]

    for _lr in lr, _wd in wd, _ntrees in ntrees, _depth in depth, _stack_size in stack_size, _hidden_size in hidden_size

        hyper = Dict(
            :loss => loss,
            :metric => metric,
            :device => device,
            :early_stopping_rounds => early_stopping_rounds,
            :tree_type => tree_type,
            :nrounds => nrounds,
            :lr => _lr,
            :wd => _wd,
            :ntrees => _ntrees,
            :depth => _depth,
            :stack_size => _stack_size,
            :hidden_size => _hidden_size,
            :batchsize => batchsize
        )

        push!(hyper_list, hyper)
    end

    return hyper_list

end