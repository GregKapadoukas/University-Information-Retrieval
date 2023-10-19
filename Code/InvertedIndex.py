class InvertedIndex:

    def __init__(self):
        self.__content = {}
    
    def insert(self, word, doc):
        if word in self.__content:
            self.__addOccurance(self.__content[word], doc)
        else:
            self.__content[word] = [1, {doc:  1}]
    
    def countOccurances(self, word):
        count = 0
        if (word in self.__content.keys()):
            for doc, n in self.__content[word][1].items():
                count += n    
            return count
        else:
            return "Word not found"
            
    def delete(self, word):
        if word in self.__content:
            del self.__content[word]
            return f"Word {word} deleted successfully"
        else:
            return "Word not found"
    
    def rename(self, original_word, new_word):
        if original_word == new_word:
            return f"Original and new words are the same"
        elif original_word in self.__content:
            new_inserts = []
            for doc, frequency in self.__content[original_word][1].items():
                for i in range(0,frequency):
                    new_inserts.append(doc)
            for doc in new_inserts:
                self.insert(new_word, doc)
            self.delete(original_word)
            return f"Word {original_word} was renamed to {new_word} successfully"
        else:
            return f"Word {original_word} not found"
        
    def getWords(self):
        return self.__content.items()
            
    def __addOccurance(self, inverted_list, doc):
        if doc in inverted_list[1]:
            inverted_list[1][doc] += 1
        else:
            inverted_list[1][doc] = 1
            inverted_list[0] += 1
    
    def __str__(self):
        result = ""
        for word, data in self.__content.items():
            result += f"Word: {word} -> n: {data[0]}\n"
            for doc, n in data[1].items():
                result += f"['{str(doc)}', {str(n)}],"
            result = result[:-1] + '\n\n'
        return result
