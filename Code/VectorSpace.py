import math

from InvertedIndex import InvertedIndex


class QueryPreprocessInfo:
    def __init__(
        self,
        stopwords,
        removed_frequent_words,
        replaced_words,
        stemmer,
    ):
        self.stopwords = stopwords
        self.removed_frequent_words = removed_frequent_words
        self.replaced_words = replaced_words
        self.stemmer = stemmer


class VectorSpace:
    def __init__(
        self,
        documents,
        queries,
        invertedIndex: InvertedIndex,
        document_tf_weighting,
        document_idf_weighting,
        document_normalize_function,
        query_tf_weighting,
        query_idf_weighting,
        query_normalize_function,
        query_stopwords,
        query_removed_frequent_words,
        query_replaced_words,
        query_stemmer,
    ):
        self.invertedIndex = invertedIndex
        self.documents = {}
        self.queries = {}
        self.document_tf_weighting = document_tf_weighting.__get__(self, VectorSpace)
        self.document_idf_weighting = document_idf_weighting.__get__(self, VectorSpace)
        self.document_normalize_function = document_normalize_function.__get__(
            self, VectorSpace
        )
        self.query_tf_weighting = query_tf_weighting.__get__(self, VectorSpace)
        self.query_idf_weighting = query_idf_weighting.__get__(self, VectorSpace)
        self.query_normalize_function = query_normalize_function.__get__(
            self, VectorSpace
        )
        self.query_preprocess_info = QueryPreprocessInfo(
            query_stopwords,
            query_removed_frequent_words,
            query_replaced_words,
            query_stemmer,
        )
        self.addDocuments(documents)
        self.addQueries(queries)

    def addDocuments(self, documents):
        for doc in documents:
            if (
                self.document_tf_weighting == self.tf_doublehalfnormalization
                or self.document_tf_weighting == self.tf_doubleknormalization
            ):
                max_tf = max(
                    self.tf_simplefrequency(token, doc, "document", 0)
                    for token, _ in self.invertedIndex.getTokens()
                )
            else:
                max_tf = 0
            w = [
                self.compute_weights(token, doc, "document", max_tf)
                for token, _ in self.invertedIndex.getTokens()
            ]
            self.documents[doc] = w
        self.documents = self.document_normalize_function(self.documents)

    def addQueries(self, queries):
        query_id = 0
        for query in queries:
            query = self.preprocessQuery(query)
            if (
                self.query_tf_weighting == self.tf_doublehalfnormalization
                or self.query_tf_weighting == self.tf_doubleknormalization
            ):
                max_tf = max(
                    self.tf_simplefrequency(token, query, "query", 0)
                    for token, _ in self.invertedIndex.getTokens()
                )
            else:
                max_tf = 0
            w = [
                self.compute_weights(token, query, "query", max_tf)
                for token, _ in self.invertedIndex.getTokens()
            ]
            self.queries[str(query_id)] = w
            query_id += 1
        self.queries = self.query_normalize_function(self.queries)

    def compute_weights(self, token, doc, add_type, max_tf):
        if add_type == "document":
            tf = self.document_tf_weighting(token, doc, add_type, max_tf)
            idf = self.document_idf_weighting(token)
            return tf * idf
        elif add_type == "query":
            tf = self.query_tf_weighting(token, doc, add_type, max_tf)
            idf = self.query_idf_weighting(token)
            return tf * idf
        else:
            return -1

    def getFrequency(self, token, doc, add_type):
        if add_type == "document":
            return self.invertedIndex.getTF(token, doc)
        elif add_type == "query":
            return self.computeQueryFrequency(token, doc)
        return 0

    def computeQueryFrequency(self, token, doc):
        count = 0
        for word in doc:
            if word == token:
                count += 1
        return count

    def tf_binary(self, token, doc, add_type, max_tf):
        if self.getFrequency(token, doc, add_type) == 0:
            return 0
        else:
            return 1

    def tf_simplefrequency(self, token, doc, add_type, max_tf):
        return self.getFrequency(token, doc, add_type)

    def tf_logfrequency(self, token, doc, add_type, max_tf):
        return 1 + math.log(self.getFrequency(token, doc, add_type))

    def tf_doublehalfnormalization(self, token, doc, add_type, max_tf):
        return 0.5 + 0.5 * (self.getFrequency(token, doc, add_type) / max_tf)

    def tf_doubleknormalization(self, token, doc, K, add_type, max_tf):
        return K + (1 - K) * (self.getFrequency(token, doc, add_type) / max_tf)

    def idf_one(self, token):
        return 1

    def idf_logsimple(self, token):
        return math.log(
            self.invertedIndex.getTotalNumberOfDocuments()
            / self.invertedIndex.getIDF(token)
        )

    def idf_lognormalized(self, token):
        return math.log(
            1
            + self.invertedIndex.getTotalNumberOfDocuments()
            / self.invertedIndex.getIDF(token)
        )

    def idf_lognormmax(self, token):
        return math.log(
            1 + self.invertedIndex.getMaxIDF() / self.invertedIndex.getIDF(token)
        )

    def no_normalization(self, weights):
        return weights

    def cosine_normalization(self, weights):
        for entry, values in weights.items():
            scale_value = math.sqrt(sum(value**2 for value in values))
            if scale_value != 0:
                weights[entry] = [value / scale_value for value in values]
        return weights

    def getDocumentVector(self, document):
        return self.documents[document]

    def getQueryVector(self, query):
        return self.queries[query]

    def preprocessQuery(self, query):
        query = query.lower()
        query = query.split()
        result_query = []
        for i in range(len(query)):
            # Word is stopword
            if query[i] in self.query_preprocess_info.stopwords:
                continue
            # Stem word
            stemmed_word = self.query_preprocess_info.stemmer.stem(query[i].lower())
            if stemmed_word in self.query_preprocess_info.replaced_words.keys():
                # Check if word is infrequent replaced
                replaced_word = self.query_preprocess_info.replaced_words[query[i]]
                # Check if replaced word is one of frequent removed
                if (
                    replaced_word
                    not in self.query_preprocess_info.removed_frequent_words
                ):
                    result_query.append(replaced_word)
            # Check if stemmed word is one of removed frequent words
            if stemmed_word not in self.query_preprocess_info.removed_frequent_words:
                result_query.append(stemmed_word)
        query = result_query
        return query

    def cosineSimilarity(self, document, query):
        if document in self.documents.keys() and query in self.queries.keys():
            a = sum(
                wij * wiq
                for wij, wiq in zip(self.documents[document], self.queries[query])
            )
            b = math.sqrt(sum(wij**2 for wij in self.documents[document]))
            c = math.sqrt(sum(wiq**2 for wiq in self.queries[query]))
            return a / (b * c)
        else:
            print(
                f"Document {document} in self.documents.keys() = {document in self.documents.keys()} and Query {query} in self.queries.keys() = {query in self.queries.keys()}"
            )
            return -1

    def lookup(self, query, num_responses):
        document_similarities = {
            document: self.cosineSimilarity(document, query)
            for document in self.documents.keys()
        }
        sorted_similarities = sorted(
            document_similarities.items(), key=lambda item: item[1], reverse=True
        )
        sorted_similarities = sorted_similarities[:num_responses]
        sorted_similarities = [i[0].lstrip("0") for i in sorted_similarities]
        return sorted_similarities

    def __str__(self):
        output = ""
        output += "Documents:\n"
        for document, weights in self.documents.items():
            output += f"{document}:\n"
            for w in weights:
                output += f"{w},"
                output = output[:-1] + "\n"
        output += "Queries:\n"
        for query, weights in self.queries.items():
            output += f"{query}:\n"
            for w in weights:
                output += f"{w},"
                output = output[:-1] + "\n"
        return output
