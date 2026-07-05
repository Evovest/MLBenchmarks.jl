using NeuroTabModels

include("neurotrees.jl")
include("tabM.jl")

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

    results = OrderedDict{Symbol,Any}[]

    # warmup
    hyper = copy(first(hyper_list))
    hyper[:nrounds] = 1
    hyper[:backend] = "reactant"
    config = NeuroTabModels.NeuroTabRegressor(; hyper...)
    NeuroTabModels.fit(config, dtrain; deval, feature_names, target_name)

    for (i, hyper) in enumerate(hyper_list)
        @info "run_experiment(NeuroTabModels) loop $i"
        hyper[:backend] = "reactant"
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

        model_type = replace(last(split(string(typeof(config.arch)), ".")), r"config$"i => "") |> lowercase
        res = OrderedDict{Symbol,Any}(
            :model_type => model_type,
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
