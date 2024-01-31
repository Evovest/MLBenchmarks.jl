function load_data(::Type{Dataset{:higgs}}; uniformize=false, aws_config=AWSConfig(), kwargs...)

    path = "share/data/higgs/HIGGS.arrow"
    df_tot = read_arrow_aws(path; bucket="jeremiedb", aws_config)

    rename!(df_tot, "Column1" => "y")
    feature_names = setdiff(names(df_tot), ["y"])
    target_name = "y"

    df_tot.y = Int.(df_tot.y)
    dtrain = df_tot[1:end-1_000_000, :];
    deval = df_tot[end-1_000_000+1:end-500_000, :];
    dtest = df_tot[end-500_000+1:end, :];
    
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

