from ModelResults import ModelResults

results = [[100, 200, 300, 400, 500], [100, 200, 300]]
real = [[100, 300, 600], [200]]

results2 = [[200, 300, 400, 500, 600], [200, 300, 400]]
real = [[100, 300, 600], [200]]

modelResults = ModelResults(results, real, "test")
model2Results = ModelResults(results2, real, "test2")
modelResults.compare_precision_recall_curve(model2Results, 0)
