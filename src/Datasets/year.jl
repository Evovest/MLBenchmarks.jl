function load_data(::Type{Dataset{:year}}; uniformize=false, aws_config=AWSConfig(), kwargs...)

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

    transform!(df_tot, "Column1" => identity => "y")
    transform!(df_tot, "y" => (x -> (x .- mean(x)) ./ std(x)) => "y_norm")
    select!(df_tot, Not("Column1"))
    feature_names = setdiff(names(df_tot), ["y", "y_norm"])
    df_tot.w .= 1.0
    target_name = "y"

    dtrain = df_tot[train_idx, :]
    deval = df_tot[eval_idx, :]
    dtest = df_tot[(end-51630+1):end, :]

    if uniformize
        ops = uniformer(
            dtrain;
            vars_in=feature_names,
            vars_out=feature_names,
            nbins=255,
            min=-1,
            max=1,
        )

        transform!(dtrain, ops)
        transform!(deval, ops)
        transform!(dtest, ops)
    end

    data = Dict(
        :dtrain => dtrain,
        :deval => deval,
        :dtest => dtest,
        :feature_names => feature_names,
        :target_name => target_name
    )

    return data
end