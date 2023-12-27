function get_hyper_evotrees(;
    loss="mse",
    metric="mse",
    tree_type="binary",
    nrounds=500,
    early_stopping_rounds=5,
    eta=0.1,
    L2=1,
    lambda=0,
    gamma=0,
    min_weight=1,
    max_depth=6,
    rowsample=0.5,
    colsample=0.5,
    nbins=64,
)

    # tunable = [:eta, :max_depth, :subsample, :colsample_bytree, :lambda, :max_bin]
    hyper_list = Dict{Symbol,Any}[]

    for _eta in eta, _max_depth in max_depth, _rowsample in rowsample, _colsample in colsample, _L2 in L2, _lambda in lambda, _gamma in gamma, _nbins in nbins, _min_weight in min_weight

        hyper = Dict(
            :loss => loss,
            :metric => metric,
            :tree_type => tree_type, # hist/gpu_hist
            :nrounds => nrounds,
            :early_stopping_rounds => early_stopping_rounds,
            :eta => _eta,
            :L2 => _L2,
            :lambda => _lambda,
            :gamma => _gamma,
            :min_weight => _min_weight,
            :max_depth => _max_depth,
            :rowsample => _rowsample,
            :colsample => _colsample,
            :nbins => _nbins
        )

        push!(hyper_list, hyper)
    end

    return hyper_list

end