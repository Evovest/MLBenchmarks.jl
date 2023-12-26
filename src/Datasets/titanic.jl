function load_data(::Type{Dataset{:titanic}}; seed=123, kwargs...)

    seed!(seed)

    df = MLDatasets.Titanic().dataframe

    # convert string feature to Categorical
    transform!(df, :Sex => categorical => :Sex)
    transform!(df, :Sex => ByRow(levelcode) => :Sex)

    # treat string feature and missing values
    transform!(df, :Age => ByRow(ismissing) => :Age_ismissing)
    transform!(df, :Age => (x -> coalesce.(x, median(skipmissing(x)))) => :Age)

    # remove unneeded variables
    df = df[:, Not([:PassengerId, :Name, :Embarked, :Cabin, :Ticket])]

    # The full data can now be split according to train and eval indices. 
    # Target and feature names are also set.
    train_ratio = 0.7
    eval_ratio = 0.15
    idx = randperm(nrow(df))
    train_idx = idx[1:Int(round(train_ratio * nrow(df)))]
    eval_idx = idx[Int(round(train_ratio * nrow(df)))+1:Int(round((train_ratio + eval_ratio) * nrow(df)))]
    test_idx = idx[Int(round((train_ratio + eval_ratio) * nrow(df)))+1:end]

    dtrain = df[train_idx, :]
    deval = df[eval_idx, :]
    dtest = df[test_idx, :]

    @info nrow(df)
    @info nrow(dtrain)
    @info nrow(deval)
    @info nrow(dtest)

    target_name = "Survived"
    feature_names = setdiff(names(df), [target_name])

    data = Dict(
        :dtrain => dtrain,
        :deval => deval,
        :dtest => dtest,
        :feature_names => feature_names,
        :target_name => target_name
    )

    return data
end