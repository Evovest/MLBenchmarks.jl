function get_hyper_tabm(
    hyper_size;
    loss="mse",
    metric=loss,
    device="gpu",
    early_stopping_rounds=5,
    nrounds=200,
    lr=1e-3,
    wd=0.0,
    arch_type=:tabm,
    k=16,
    d_block=128,
    n_blocks=3,
    dropout=0.1,
    embedding_type="piecewise",
    d_embedding=16,
    nbins=16,
    batchsize=256,
)

    # tunable = [:eta, :max_depth, :subsample, :colsample_bytree, :lambda, :max_bin]
    hyper_list = Dict{Symbol,Any}[]

    for _lr in lr, _wd in wd, _k in k, _d_block in d_block, _n_blocks in n_blocks, _dropout in dropout,
        _d_embedding in d_embedding, _nbins in nbins

        hyper = Dict(
            :arch_name => "TabMConfig",
            :arch_config => Dict(
                :arch_type => arch_type,
                :k => _k,
                :d_block => _d_block,
                :n_blocks => _n_blocks,
                :dropout => _dropout,
            ),
            :embedding_config => Dict(
                :embedding_type => embedding_type,
                :d_embedding => _d_embedding,
                :nbins => _nbins,
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
