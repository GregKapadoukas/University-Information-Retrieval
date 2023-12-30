import math

import matplotlib.pyplot as plt
from numpy import arange


class ModelResults:
    def __init__(self, results, correct, model):
        self.results = results
        self.correct = correct
        self.precision = self.__precisionArray()
        self.recall = self.__recallArray()
        self.dcg = self.__dcgArray()
        self.model = model

    def __precisionArray(self):
        precision = []
        # For each query
        for result, correct_answer in zip(self.results, self.correct):
            # For each document in response
            query_precision = []
            for i in range(len(result)):
                # For all previous documents in response, to get incremental precision
                number_correct_documents_found = 0
                for j in range(i + 1):
                    if result[j] in correct_answer.keys():
                        number_correct_documents_found += 1
                query_precision.append(
                    number_correct_documents_found / (len(query_precision) + 1)
                )
            precision.append(query_precision)
        return precision

    def __recallArray(self):
        recall = []
        # For each query
        for result, correct_answer in zip(self.results, self.correct):
            # For each document in response
            query_recall = []
            for i in range(len(result)):
                # For all previous documents in response, to get incremental recall
                number_correct_documents_found = 0
                for j in range(i + 1):
                    if result[j] in correct_answer.keys():
                        number_correct_documents_found += 1
                query_recall.append(
                    number_correct_documents_found / len(correct_answer)
                )
            recall.append(query_recall)
        return recall

    def __dcgArray(self):
        dcg = []
        # For each query
        for result, correct_answer in zip(self.results, self.correct):
            # For each document in response
            query_dcg = []
            for i in range(len(result)):
                # For all previous documents in response, to get incremental dcg
                if i == 0:
                    if result[i] in correct_answer.keys():
                        query_dcg.append(correct_answer[result[i]])
                    else:
                        query_dcg.append(0)
                else:
                    if result[i] in correct_answer.keys():
                        query_dcg.append(
                            query_dcg[-1] / math.log2(i + 1) + correct_answer[result[i]]
                        )
                    else:
                        query_dcg.append(query_dcg[-1] / math.log2(i + 1))
            dcg.append(query_dcg)
        return dcg

    def getResults(self, query_num):
        return self.results[query_num]

    def getCorrect(self, query_num):
        return self.correct[query_num]

    def getPrecision(self, query_num):
        return self.precision[query_num]

    def getRecall(self, query_num):
        return self.recall[query_num]

    def getDCG(self, query_num):
        return self.dcg[query_num]

    def getMeanPrecision(self):
        meanPrecision = [0.0] * len(self.precision[0])
        for query in self.precision:
            for i in range(len(query)):
                meanPrecision[i] += query[i]
        for i in range(len(meanPrecision)):
            meanPrecision[i] /= len(self.precision)
        return meanPrecision

    def getMeanRecall(self):
        meanRecall = [0.0] * len(self.recall[0])
        for query in self.recall:
            for i in range(len(query)):
                meanRecall[i] += query[i]
        for i in range(len(meanRecall)):
            meanRecall[i] /= len(self.recall)
        return meanRecall

    def getMeanDCG(self):
        meanDCG = [0.0] * len(self.dcg[0])
        for query in self.dcg:
            for i in range(len(query)):
                meanDCG[i] += query[i]
        for i in range(len(meanDCG)):
            meanDCG[i] /= len(self.dcg)
        return meanDCG

    def precision_recall_curve(self, query_num):
        plt.plot(self.getRecall(query_num), self.getPrecision(query_num))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(
            f"Precision Recall Curve for model {self.model} and query {query_num}"
        )
        plt.show()

    def dcg_curve(self, query_num):
        plt.plot(self.getDCG(query_num))
        plt.xlabel("Document Number")
        plt.ylabel("DCG")
        plt.title(f"DCG Curve for model {self.model} and query {query_num}")
        plt.show()

    def mean_precision_recall_curve(self):
        plt.plot(self.getMeanRecall(), self.getMeanPrecision())
        plt.xlabel("Mean Recall")
        plt.ylabel("Mean Precision")
        plt.title(f"Precision Recall Curve for model {self.model}")
        plt.show()

    def mean_dcg_curve(self):
        plt.plot(self.getMeanDCG())
        plt.xlabel("Document Number")
        plt.ylabel("DCG")
        plt.title(f"DCG Curve for model {self.model}")
        plt.show()

    def compare_precision_recall_curve(self, other_model_results, query_num):
        plt.plot(
            self.getRecall(query_num), self.getPrecision(query_num), label=self.model
        )
        plt.plot(
            other_model_results.getRecall(query_num),
            other_model_results.getPrecision(query_num),
            label=other_model_results.model,
        )
        plt.xlabel("Mean Recall")
        plt.ylabel("Mean Precision")
        plt.title(
            f"Precision Recall Curve Comparisson for Model {self.model} and {other_model_results.model} for query {query_num}"
        )
        plt.legend()
        plt.show()

    def compare_dcg_curve(self, other_model_results, query_num):
        plt.plot(self.getDCG(query_num), label=self.model)
        plt.plot(other_model_results.getDCG(query_num), label=other_model_results.model)
        plt.xlabel("Document Number")
        plt.ylabel("DCG")
        plt.title(
            f"DCG Curve Comparisson for Model {self.model} and {other_model_results.model} for query {query_num}"
        )
        plt.legend()
        plt.show()

    def compare_mean_precision_recall_curve(self, other_model_results):
        plt.plot(self.getMeanRecall(), self.getMeanPrecision(), label=self.model)
        plt.plot(
            other_model_results.getMeanRecall(),
            other_model_results.getMeanPrecision(),
            label=other_model_results.model,
        )
        plt.xlabel("Mean Recall")
        plt.ylabel("Mean Precision")
        plt.title(
            f"Precision Recall Curve Comparisson for Model {self.model} and {other_model_results.model}"
        )
        plt.legend()
        plt.show()

    def compare_mean_dcg_curve(self, other_model_results):
        plt.plot(self.getMeanDCG(), label=self.model)
        plt.plot(other_model_results.getMeanDCG(), label=other_model_results.model)
        plt.xlabel("Document Number")
        plt.ylabel("DCG")
        plt.title(
            f"DCG Curve Comparisson for Model {self.model} and {other_model_results.model}"
        )
        plt.legend()
        plt.show()
