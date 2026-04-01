function get_hyper_neurotrees(
    hyper_size;
    loss="mse",
    metric=loss,
    device="gpu",
    early_stopping_rounds=5,
    nrounds=200,
    lr=1e-3,
    wd=0.0,
    k=1,
    ntrees=64,
    depth=4,
    actA="identity",
    tree_type="binary",
    embedding_type="batchnorm",
    d_embedding=16,
    bins=16,
    stack_size=1,
    hidden_size=1,
    scaler=true,
    init_scale=0.1,
    batchsize=1024,
)

    # tunable = [:eta, :max_depth, :subsample, :colsample_bytree, :lambda, :max_bin]
    hyper_list = Dict{Symbol,Any}[]

    for _lr in lr, _wd in wd, _k in k, _ntrees in ntrees, _depth in depth, _stack_size in stack_size, _hidden_size in hidden_size, _init_scale in init_scale,
        _d_embedding in d_embedding, _bins in bins

        hyper = Dict(
            :arch_name => "NeuroTreeConfig",
            :arch_config => Dict(
                :k => _k,
                :tree_type => tree_type,
                :ntrees => _ntrees,
                :depth => _depth,
                :actA => actA,
                :scaler => scaler,
                :init_scale => _init_scale,
                :hidden_size => _hidden_size,
                :stack_size => _stack_size
            ),
            :embedding_config => Dict(
                :embedding_type => embedding_type,
                :d_embedding => _d_embedding,
                :bins => _bins,
                :frequencies => 16,
                :activation => nothing,
            ),
            :loss => loss,
            :metric => metric,
            :device => device,
            :early_stopping_rounds => early_stopping_rounds,
            :nrounds => nrounds,
            :lr => _lr,
            :wd => _wd,
            :batchsize => batchsize
        )

        push!(hyper_list, hyper)
    end
    rng = Xoshiro(123)
    hyper_list = sample(rng, hyper_list, hyper_size, replace=false)
    return hyper_list
end
