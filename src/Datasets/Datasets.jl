module Datasets

using Arrow
using DataFrames
using CSV
import CategoricalArrays: categorical, levelcode, levels
using Statistics: mean, std
using StatsBase: median, tiedrank, quantile

import ReadLIBSVM: read_libsvm
import Random: seed!, randperm
import MLDatasets
import AWS: AWSCredentials, AWSConfig, @service
@service S3

export load_data

const aws_creds = AWSCredentials(ENV["AWS_ACCESS_KEY_ID_JDB"], ENV["AWS_SECRET_ACCESS_KEY_JDB"])
const aws_config = AWSConfig(; creds=aws_creds, region="ca-central-1")

const dataset_list = [:year, :higgs, :yahoo, :msrank, :titanic, :boston]

include("utils.jl")

include("titanic.jl")
include("boston.jl")
include("year.jl")
include("yahoo.jl")
include("msrank.jl")
include("higgs.jl")

end
