using DataFrames
using Queryverse
using Plots
using Printf
using StatsBase

df = load("ex1data1.csv") |> DataFrame

using RDatasets, LIBSVM, SparseArrays

# Load Fisher's classic iris data
iris = dataset("datasets", "iris")

# LIBSVM handles multi-class data automatically using a one-against-one strategy
labels = convert(Array, iris[:, 5])

# First dimension of input data is features; second is instances
instances = convert(Array, iris[:, 1:4])'

# Train SVM on half of the data using default parameters. See documentation
# of svmtrain for options
model = svmtrain(instances[:, 1:2:end], labels[1:2:end]);

# Test model on the other half of the data.
(predicted_labels, decision_values) = svmpredict(model, instances[:, 2:2:end]);

# Compute accuracy
@printf("Accuracy: %.2f%%\n", mean((predicted_labels .== labels[2:2:end]))*100)
print(decision_values)
