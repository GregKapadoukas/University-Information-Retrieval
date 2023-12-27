import matplotlib.pyplot as plt


class ModelResults:
    def __init__(self, results, correct, model):
        self.results = results
        self.correct = correct
        self.precision = self.__precisionArray(correct)
        self.recall = self.__recallArray(correct)
        self.model = model

    def __precisionArray(self, correct_answers):
        precision = []
        # For each query
        for result, correct_answer in zip(self.results, correct_answers):
            # For each document in response
            query_precision = []
            for i in range(len(result)):
                # For all previous documents in response, to get incremental precision
                number_incorrect_documents_found = 0
                for j in range(i + 1):
                    if result[j] not in correct_answer:
                        number_incorrect_documents_found += 1
                query_precision.append(
                    1 - (number_incorrect_documents_found / len(result))
                )
            precision.append(query_precision)
        return precision
        pass

    def __recallArray(self, correct_answers):
        recall = []
        # For each query
        for result, correct_answer in zip(self.results, correct_answers):
            # For each document in response
            query_recall = []
            for i in range(len(result)):
                # For all previous documents in response, to get incremental recall
                number_correct_documents_found = 0
                for j in range(i + 1):
                    if result[j] in correct_answer:
                        number_correct_documents_found += 1
                query_recall.append(
                    number_correct_documents_found / len(correct_answer)
                )
            recall.append(query_recall)
        return recall

    def getResults(self, query_num):
        return self.results[query_num]

    def getCorrect(self, query_num):
        return self.correct[query_num]

    def getPrecision(self, query_num):
        return self.precision[query_num]

    def getRecall(self, query_num):
        return self.recall[query_num]

    def precision_recall_curve(self, query_num):
        plt.plot(self.getRecall(query_num), self.getPrecision(query_num))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(
            f"Precision Recall Curve for model {self.model} for query {query_num+1}"
        )
        plt.show()

    def compare_precision_recall_curve(self, other_model_results, query_num):
        plt.plot(self.getRecall(query_num), self.getPrecision(query_num))
        plt.xlabel("Recall")
        plt.plot(
            other_model_results.getRecall(query_num),
            other_model_results.getPrecision(query_num),
        )
        plt.ylabel("Precision")
        plt.title(
            f"Precision Recall Curve Comparisson for Model {self.model} and {other_model_results.model} for query {query_num+1}"
        )
        plt.show()

    def DCG(self, real):
        pass
