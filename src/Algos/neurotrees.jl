using NeuroTabModels

function get_hyper_neurotrees(
    hyper_size;
    loss="mse",
    metric=loss,
    device="gpu",
    early_stopping_rounds=5,
    nrounds=200,
    lr=1e-3,
    wd=0.0,
    ntrees=64,
    depth=4,
    actA=["identity"],
    tree_type=["binary"],
    init_scale=1,
    batchsize=2048,
    stack_size=1,
    hidden_size=1,
    scaler=false
)

    # tunable = [:eta, :max_depth, :subsample, :colsample_bytree, :lambda, :max_bin]
    hyper_list = Dict{Symbol,Any}[]

    for _lr in lr, _wd in wd, _ntrees in ntrees, _depth in depth, _stack_size in stack_size, _hidden_size in hidden_size, _actA in actA, _init_scale in init_scale

        hyper = Dict(
            :arch_name => "NeuroTreeConfig",
            :arch_config => Dict(
                :depth => _depth,
                :ntrees => _ntrees,
                :hidden_size => _hidden_size,
                :actA => _actA,
                :init_scale => _init_scale,
                :stack_size => _stack_size
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

function run_experiment(
    ::Val{:NeuroTabModels},
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
    config = NeuroTabModels.NeuroTabRegressor(; hyper...)
    NeuroTabModels.fit(config, dtrain; deval, feature_names, target_name)

    for (i, hyper) in enumerate(hyper_list)
        @info "run_experiment(NeuroTabModels) loop $i"
        config = NeuroTabModels.NeuroTabRegressor(; hyper...)
        train_time = @elapsed m = NeuroTabModels.fit(
            config,
            dtrain;
            deval,
            feature_names,
            target_name,
            print_every_n=print_every_n
        )
        p_eval = m(deval)
        p_test = m(dtest)

        res = OrderedDict{Symbol,Any}(
            :model_type => "neurotrees",
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
