using MLBenchmarks
import MLBenchmarks: mse, gini

using Zygote
using NeuroTabModels
using DataFrames
using CSV
using Statistics: mean, std

data_name = :sberbank
data = load_data(data_name; uniformize=true)
results_path = joinpath(@__DIR__, "..", "results", string(data_name))
mkpath(results_path)

################################
# TabM temporal embedding (Sberbank)
#   none            : no time column
#   num             : t_std as ordinary feature
#   temporal        : Fourier + trend
#   temporal_notrend: Fourier only (ablation)
#   temporal_only   : temporal branch only, no num embedding
################################

const SEEDS = 1:5

arch    = NeuroTabModels.TabMConfig(; arch_type=:tabm, k=16, d_block=64, n_blocks=2, dropout=0.1)
emb_num = NeuroTabModels.PiecewiseLinearEmbeddings(; d_embedding=16, bins=32)

feature_names_notime   = setdiff(names(data.dtrain), [data.target_name, "t", "t_std"])
feature_names_num      = vcat(feature_names_notime, ["t_std"])
feature_names_temporal = vcat(feature_names_notime, ["t"])
t_idx = lastindex(feature_names_temporal)
@assert feature_names_temporal[t_idx] == "t"
@assert maximum(data.dtrain.t) <= minimum(data.deval.t) <= minimum(data.dtest.t)

_temporal(trend) = NeuroTabModels.TemporalEmbeddings(;
    index=t_idx,
    order=Int[4, 1, 7, 0],
    periods=Float32[31_557_600, 2_629_800, 604_800, 86_400],
    trend,
    d_embedding=16,
)

function _fit_eval(feature_names, embedding_config, seed)
    learner = NeuroTabModels.NeuroTabRegressor(arch;
        embedding_config, loss=:mse, metric=:mse,
        nrounds=100, early_stopping_rounds=20,
        lr=1e-2, batchsize=1024, device=:cpu, backend=:zygote, seed)
    m = NeuroTabModels.fit(learner, data.dtrain;
        deval=data.deval, feature_names, target_name=data.target_name, print_every_n=10)
    p_eval = m(data.deval)
    p_test = m(data.dtest)
    return (
        eval_mse    = mse(p_eval, data.deval[!, data.target_name]),
        test_mse    = mse(p_test, data.dtest[!, data.target_name]),
        test_gini   = gini(p_test, data.dtest[!, data.target_name]),
        best_nround = get(m.info[:logger], :best_iter, m.info[:nrounds]),
    )
end

function run_variant(variant, feature_names, embedding_config)
    runs = [_fit_eval(feature_names, embedding_config, s) for s in SEEDS]
    ms(f) = (mean(f.(runs)), std(f.(runs)))
    em_m, em_s = ms(r -> r.eval_mse)
    tm_m, tm_s = ms(r -> r.test_mse)
    tg_m, tg_s = ms(r -> r.test_gini)
    bn_m, _    = ms(r -> r.best_nround)
    return (;
        variant,
        n_seeds          = length(SEEDS),
        eval_mse_mean    = em_m, eval_mse_std  = em_s,
        test_mse_mean    = tm_m, test_mse_std  = tm_s,
        test_gini_mean   = tg_m, test_gini_std = tg_s,
        best_nround_mean = bn_m,
    )
end

results_df = DataFrame([
    run_variant("none", feature_names_notime,
        NeuroTabModels.EmbeddingLayer(; num=emb_num)),
    run_variant("num", feature_names_num,
        NeuroTabModels.EmbeddingLayer(; num=emb_num)),
    run_variant("temporal", feature_names_temporal,
        NeuroTabModels.EmbeddingLayer(; num=emb_num, temp=_temporal(true))),
    run_variant("temporal_notrend", feature_names_temporal,
        NeuroTabModels.EmbeddingLayer(; num=emb_num, temp=_temporal(false))),
    run_variant("temporal_only", feature_names_temporal,
        NeuroTabModels.EmbeddingLayer(; temp=_temporal(true))),
])

CSV.write(joinpath(results_path, "tabm_temporal.csv"), results_df)
