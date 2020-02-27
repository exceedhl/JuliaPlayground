using MLJ, MLJModels, Statistics, PrettyPrinting, RDatasets
X, y = @load_iris;

@load DecisionTreeClassifier
tree_model = DecisionTreeClassifier()
r = range(tree_model, :max_depth, lower=1, upper=5)


tm = TunedModel(model=tree_model, ranges=[r, ], measure=cross_entropy)

tree = machine(tm, X, y)

train, test = partition(eachindex(y), 0.7, shuffle=true)

fit!(tree, rows=train)

fitted_params(tree) |> pprint

r = report(tree)
