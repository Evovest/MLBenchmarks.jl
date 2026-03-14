function data_recipe(::Type{Dataset{:titanic}}, df; eval_perc=0.15, test_perc=0.15, seed=123, kwargs...)
    rng = Xoshiro(seed)

    # treat string feature and missing values
    transform!(df, :survived => (x -> levelcode.(x) .- 1) => :survived)

    # convert string feature to Categorical
    transform!(df, :sex => categorical => :sex)
    transform!(df, :sex => ByRow(levelcode) => :sex)

    # treat string feature and missing values
    transform!(df, :fare => (x -> coalesce.(x, median(skipmissing(x)))) => :fare)

    transform!(df, :age => ByRow(ismissing) => :age_ismissing)
    transform!(df, :age => (x -> coalesce.(x, median(skipmissing(x)))) => :age)

    # remove unneeded variables
    df = df[:, Not(["name", "embarked", "ticket", "cabin", "boat", "home.dest", "body"])]
    disallowmissing!(df)

    idx = randperm(rng, nrow(df))
    eval_cut = floor(Int, (1 - eval_perc - test_perc) * nrow(df))
    test_cut = floor(Int, (1 - test_perc) * nrow(df))

    dtrain = df[view(idx, 1:eval_cut), :]
    deval = df[view(idx, eval_cut+1:test_cut), :]
    dtest = df[view(idx, test_cut+1:end), :]

    target_name = "survived"
    feature_names = setdiff(names(df), [target_name])

    return (;
        loss=:logloss,
        metric=:logloss,
        metrics=[:logloss, :accuracy],
        dtrain,
        deval,
        dtest,
        target_name,
        feature_names)
end
