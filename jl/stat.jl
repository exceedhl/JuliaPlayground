using Distributions
using StatsBase
using CSV
using DataFrames
using HypothesisTests
using StatPlots
using GLM
pyplot()

age = rand(10:80, 100)
