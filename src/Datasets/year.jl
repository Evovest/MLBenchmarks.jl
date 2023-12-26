function load_data(::Type{Dataset{:year}}; kwargs...)

    path = "share/data/year/year.csv"
    raw = S3.get_object("jeremiedb", path, Dict("response-content-type" => "application/octet-stream"); aws_config)
    df = DataFrame(CSV.File(raw, header=false))
    df_tot = copy(df)

    path = "share/data/year/year-train-idx.txt"
    raw = S3.get_object("jeremiedb", path, Dict("response-content-type" => "application/octet-stream"); aws_config)
    train_idx = DataFrame(CSV.File(raw, header=false))[:, 1] .+ 1

    path = "share/data/year/year-eval-idx.txt"
    raw = S3.get_object("jeremiedb", path, Dict("response-content-type" => "application/octet-stream"); aws_config)
    eval_idx = DataFrame(CSV.File(raw, header=false))[:, 1] .+ 1

    transform!(df_tot, "Column1" => identity => "y_raw")
    transform!(df_tot, "y_raw" => (x -> (x .- mean(x)) ./ std(x)) => "y_norm")
    select!(df_tot, Not("Column1"))
    feature_names = setdiff(names(df_tot), ["y_raw", "y_norm", "w"])
    df_tot.w .= 1.0
    target_name = "y_norm"

    transform!(df_tot, feature_names .=> percent_rank .=> feature_names)

    dtrain = df_tot[train_idx, :]
    deval = df_tot[eval_idx, :]
    dtest = df_tot[(end-51630+1):end, :]

    data = Dict(
        :dtrain => dtrain,
        :deval => deval,
        :dtest => dtest,
        :feature_names => feature_names,
        :target_name => target_name
    )

    return data
end