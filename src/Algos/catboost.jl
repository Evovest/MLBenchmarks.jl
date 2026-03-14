using CatBoost
using PythonCall: PyList, pyconvert

function get_hyper_catboost(
    hyper_size;
    objective="RMSE",
    eval_metric="RMSE",
    iterations=500,
    early_stopping_rounds=5,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.5,
    rsm=0.5,
    reg_lambda=1,
)

    # tunable = [:eta, :max_depth, :subsample, :colsample_bytree, :lambda, :max_bin]
    hyper_list = Dict{Symbol,Any}[]

    for _learning_rate in learning_rate, _max_depth in max_depth, _subsample in subsample, _rsm in rsm, _reg_lambda in reg_lambda

        hyper = Dict(
            :objective => objective,
            :eval_metric => eval_metric,
            :iterations => iterations,
            :early_stopping_rounds => early_stopping_rounds,
            :learning_rate => _learning_rate,
            :max_depth => _max_depth,
            :subsample => _subsample,
            :rsm => _rsm,
            :reg_lambda => _reg_lambda,
        )

        push!(hyper_list, hyper)
    end
    rng = Xoshiro(123)
    hyper_list = sample(rng, hyper_list, hyper_size, replace=false)
    return hyper_list
end

function run_experiment(
    ::Val{:CatBoost},
    data,
    hyper_list;
    metrics=[:logloss, :accuracy],
    print_every_n=10
)
    dtrain = CatBoost.Pool(data[:dtrain][:, data[:feature_names]], label=PyList(data[:dtrain][:, data[:target_name]]))
    deval = CatBoost.Pool(data[:deval][:, data[:feature_names]], label=PyList(data[:deval][:, data[:target_name]]))
    dtest = CatBoost.Pool(data[:dtest][:, data[:feature_names]])
    target_name = data[:target_name]

    results = Dict{Symbol,Any}[]

    # warmup
    hyper = copy(first(hyper_list))
    hyper[:iterations] = 1
    m = CatBoost.CatBoostClassifier(; hyper...)
    CatBoost.fit!(m, dtrain; eval_set=deval)

    for (i, hyper) in enumerate(hyper_list)
        @info "run_experiment(CatBoost) loop $i"
        m = CatBoost.CatBoostClassifier(; hyper...)
        train_time = @elapsed fit_result = CatBoost.fit!(m, dtrain; eval_set=deval)

        p_eval = CatBoost.predict(m, deval; prediction_type="Probability")[:, 2]
        p_test = CatBoost.predict(m, dtest; prediction_type="Probability")[:, 2]

        res = Dict{Symbol,Any}(
            :model_type => "catboost",
            :hyper_id => i,
            :train_time => train_time,
            :best_nround => pyconvert(Int, fit_result.best_iteration_),
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