using KSVM
using RDatasets
iris = dataset("datasets", "iris")
X = convert(Array, iris[1:100, 1:2])'  # The observations have to be in the columns
y = iris[1:100, :Species]
train = bitrand(size(X,2))             # Split training and testset
model = svm(X[:, train], y[train])     # Fit the linear SVM
acc = accuracy(model, X[:, ~train], y[~train])
