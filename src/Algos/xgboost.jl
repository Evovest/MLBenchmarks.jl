using XGBoost
using OrderedCollections: OrderedDict

function get_hyper_xgboost(
    hyper_size;
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist", # hist/gpu_hist
    num_round=500,
    early_stopping_rounds=5,
    eta=0.3,
    max_depth=6,
    subsample=0.5,
    colsample_bytree=0.5,
    max_bin=128,
    lambda=1,
    num_class=0,
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
            :max_bin => _max_bin,
            :num_class => num_class,
        )

        push!(hyper_list, hyper)
    end
    rng = Xoshiro(123)
    hyper_list = sample(rng, hyper_list, hyper_size, replace=false)
    return hyper_list
end

function run_experiment(
    ::Val{:XGBoost},
    data,
    hyper_list;
    metrics=[:logloss, :accuracy],
    print_every_n=10
)
    dtrain = XGBoost.DMatrix(data[:dtrain][:, data[:feature_names]], data[:dtrain][:, data[:target_name]])
    deval = XGBoost.DMatrix(data[:deval][:, data[:feature_names]], data[:deval][:, data[:target_name]])
    dtest = XGBoost.DMatrix(data[:dtest][:, data[:feature_names]])
    target_name = data[:target_name]

    results = Dict{Symbol,Any}[]

    # warmup
    hyper = copy(first(hyper_list))
    hyper[:num_round] = 1
    XGBoost.xgboost(dtrain, watchlist=OrderedDict(["eval" => deval]); hyper...)

    for (i, hyper) in enumerate(hyper_list)
        @info "run_experiment(XGBoost) loop $i"
        train_time = @elapsed m = XGBoost.xgboost(dtrain, watchlist=OrderedDict(["eval" => deval]); hyper...)

        p_eval = XGBoost.predict(m, deval)
        p_test = XGBoost.predict(m, dtest)

        res = Dict{Symbol,Any}(
            :model_type => "xgboost",
            :hyper_id => i,
            :train_time => train_time,
            :best_nround => m.best_iteration,
        )

        for metric in metrics
            fun = metric_dict[metric]
            res[Symbol("eval_", metric)] = fun(p_eval, data[:deval][!, target_name])
            res[Symbol("test_", metric)] = fun(p_test, data[:dtest][!, target_name])
        end

        push!(results, res)
    end

    return DataFrame(results)
end