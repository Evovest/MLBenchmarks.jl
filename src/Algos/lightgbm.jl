using LightGBM

const _LGBM_OBJECTIVE_MAP = Dict{Symbol,String}(
    :mse => "regression",
    :mae => "regression_l1",
    :logloss => "cross_entropy",
)

const _LGBM_METRIC_MAP = Dict{Symbol,Vector{String}}(
    :mse => ["mse"],
    :mae => ["mae"],
    :logloss => ["cross_entropy"],
)

function get_hyper_lgbm(
    hyper_size;
    loss=:mse,
    metric=loss,
    num_iterations=100,
    min_data_in_leaf=20,
    num_class=0,
    early_stopping_round=5,
    learning_rate=0.3,
    num_leaves=128,
    max_depth=-1,
    bagging_fraction=0.9,
    feature_fraction=0.8,
    max_bin=128,
    lambda_l2=1,
)

    # tunable = [:eta, :max_depth, :subsample, :colsample_bytree, :lambda, :max_bin]
    hyper_list = Dict{Symbol,Any}[]
    objective_name = _LGBM_OBJECTIVE_MAP[loss]
    metric_name = _LGBM_METRIC_MAP[metric]

    task = objective_name ∈ ["binary"] ? :classification : :regression

    for _learning_rate in learning_rate, _num_leaves in num_leaves, _max_depth in max_depth, _bagging_fraction in bagging_fraction, _feature_fraction in feature_fraction, _lambda_l2 in lambda_l2, _max_bin in max_bin, _min_data_in_leaf in min_data_in_leaf

        hyper = Dict(
            :objective => objective_name,
            :metric => metric_name,
            :num_iterations => num_iterations,
            :early_stopping_round => early_stopping_round,
            :learning_rate => _learning_rate,
            :num_leaves => _num_leaves,
            :max_depth => _max_depth,
            :bagging_fraction => _bagging_fraction,
            :feature_fraction => _feature_fraction,
            :lambda_l2 => _lambda_l2,
            :max_bin => _max_bin,
            :min_data_in_leaf => _min_data_in_leaf,
            :num_class => num_class,
            :task => task,
        )

        num_class == 0 ? delete!(hyper, :num_class) : nothing
        push!(hyper_list, hyper)
    end
    rng = Xoshiro(123)
    hyper_list = sample(rng, hyper_list, hyper_size, replace=false)
    return hyper_list
end

function run_experiment(
    ::Val{:LightGBM},
    data,
    hyper_list;
    metrics=[:logloss, :accuracy],
    print_every_n=10
)
    dtrain = Matrix(data[:dtrain][:, data[:feature_names]])
    ytrain = data[:dtrain][:, data[:target_name]]
    deval = Matrix(data[:deval][:, data[:feature_names]])
    yeval = data[:deval][:, data[:target_name]]
    dtest = Matrix(data[:dtest][:, data[:feature_names]])
    target_name = data[:target_name]

    results = OrderedDict{Symbol,Any}[]

    # warmup
    hyper = copy(first(hyper_list))
    hyper[:num_iterations] = 1
    task = pop!(hyper, :task)
    learner = task == :classification ? LightGBM.LGBMClassification : LightGBM.LGBMRegression
    m = learner(; hyper...)
    LightGBM.fit!(m, dtrain, ytrain, (deval, yeval))

    for (i, hyper) in enumerate(hyper_list)
        @info "run_experiment(LightGBM) loop $i"
        model_hyper = copy(hyper)
        pop!(model_hyper, :task)
        m = learner(; model_hyper...)
        train_time = @elapsed fit_result = LightGBM.fit!(m, dtrain, ytrain, (deval, yeval))

        p_eval = vec(LightGBM.predict(m, deval))
        p_test = vec(LightGBM.predict(m, dtest))

        res = OrderedDict{Symbol,Any}(
            :model_type => "lightgbm",
            :hyper_id => i,
            :train_time => train_time,
            :best_nround => fit_result["best_iter"],
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