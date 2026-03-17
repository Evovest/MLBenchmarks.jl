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
    n_bins=16,
    use_embeddings=true,
    embedding_type=:piecewise,
    d_embedding=8,
    scaling_init=:normal,
    batchsize=256,
    stack_size=1,
    hidden_size=1,
    scaler=false
)

    # tunable = [:eta, :max_depth, :subsample, :colsample_bytree, :lambda, :max_bin]
    hyper_list = Dict{Symbol,Any}[]

    for _lr in lr, _wd in wd, _k in k, _d_block in d_block, _n_blocks in n_blocks, _dropout in dropout, _n_bins in n_bins, _embedding_type in embedding_type, _d_embedding in d_embedding

        hyper = Dict(
            :arch_name => "TabMConfig",
            :arch_config => Dict(
                :arch_type => arch_type,
                :k => _k,
                :d_block => _d_block,
                :n_blocks => _n_blocks,
                :dropout => _dropout,
                :n_bins => _n_bins,
                :use_embeddings => true,
                :embedding_type => _embedding_type, # periodic, piecewise, linear
                :d_embedding => _d_embedding,
                :scaling_init => scaling_init
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
