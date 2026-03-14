using EvoTrees

function get_hyper_evotrees(
    hyper_size;
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
    nbins=128,
    seed=123
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
    rng = Xoshiro(123)
    hyper_list = sample(rng, hyper_list, hyper_size, replace=false)
    return hyper_list
end

function run_experiment(
    ::Val{:EvoTrees},
    data,
    hyper_list;
    metrics=[:logloss, :accuracy],
    print_every_n=10
)
    dtrain = data[:dtrain]
    deval = data[:deval]
    dtest = data[:dtest]
    feature_names = data[:feature_names]
    target_name = data[:target_name]

    results = Dict{Symbol,Any}[]

    # warmup
    hyper = copy(first(hyper_list))
    hyper[:nrounds] = 1
    config = EvoTrees.EvoTreeRegressor(; hyper...)
    EvoTrees.fit(
        config,
        dtrain;
        deval,
        feature_names,
        target_name,
    )

    for (i, hyper) in enumerate(hyper_list)
        @info "run_experiment(EvoTrees) loop $i"
        config = EvoTrees.EvoTreeRegressor(; hyper...)
        train_time = @elapsed m = EvoTrees.fit(
            config,
            dtrain;
            deval,
            feature_names,
            target_name,
            print_every_n=print_every_n,
        )

        p_eval = EvoTrees.predict(m, deval)
        p_test = EvoTrees.predict(m, dtest)

        res = Dict{Symbol,Any}(
            :model_type => "evotrees",
            :hyper_id => i,
            :train_time => train_time,
            :best_nround => m.info[:logger][:best_iter],
        )

        for metric in metrics
            fun = metric_dict[metric]
            res[Symbol("eval_", metric)] = fun(p_eval, deval[!, target_name])
            res[Symbol("test_", metric)] = fun(p_test, dtest[!, target_name])
        end

        push!(results, res)
    end

    return DataFrame(results)
end