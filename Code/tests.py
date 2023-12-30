from ModelResults import ModelResults

results = [[100, 200, 300, 400, 600], [100, 200, 300]]
real = [{100: 4.0, 300: 3.0, 600: 1.0}, {200: 1.0}]

results2 = [[200, 300, 400, 500, 600], [200, 300, 400]]
real2 = [{100: 4.0, 300: 3.0, 600: 1.0}, {200: 1.0}]

modelResults = ModelResults(results, real, "test")
model2Results = ModelResults(results2, real2, "test2")
print(f"Model 1 Real Results {real}")
print(f"Model 2 Real Results {real2}")
print(f"Model 1 Results {results}")
print(f"Model 2 Results {results2}")
print(f"Model 1 Query 1 Precision {modelResults.getPrecision(0)}")
print(f"Model 2 Query 1 Precision {model2Results.getPrecision(0)}")
print(f"Model 1 Query 1 Recall {modelResults.getRecall(0)}")
print(f"Model 2 Query 1 Recall {model2Results.getRecall(0)}")
print(f"Model 1 Query 1 DCG {modelResults.getDCG(0)}")
print(f"Model 2 Query 1 DCG {model2Results.getDCG(0)}")
print(f"Model 1 Query 2 Precision {modelResults.getPrecision(1)}")
print(f"Model 2 Query 2 Precision {model2Results.getPrecision(1)}")
print(f"Model 1 Query 2 Recall {modelResults.getRecall(1)}")
print(f"Model 2 Query 2 Recall {model2Results.getRecall(1)}")
print(f"Model 1 Query 2 DCG {modelResults.getDCG(1)}")
print(f"Model 2 Query 2 DCG {model2Results.getDCG(1)}")
print(f"Model 1 Mean Precision {modelResults.getMeanPrecision()}")
print(f"Model 2 Mean Precision {model2Results.getMeanPrecision()}")
print(f"Model 1 Mean Recall {modelResults.getMeanRecall()}")
print(f"Model 2 Mean Recall {model2Results.getMeanRecall()}")
print(f"Model 1 Mean DCG {modelResults.getMeanDCG()}")
print(f"Model 2 Mean DCG {model2Results.getMeanDCG()}")
modelResults.compare_precision_recall_curve(model2Results)
modelResults.compare_dcg_curve(model2Results)
# modelResults.precision_recall_curve()
# modelResults.dcg_curve()
