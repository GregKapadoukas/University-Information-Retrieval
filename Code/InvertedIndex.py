class InvertedIndex:

    def __init__(self):
        self.__content = {}
    
    def insert(self, token, doc):
        if token in self.__content:
            self.__addOccurance(self.__content[token], doc)
        else:
            self.__content[token] = [1, {doc:  1}]
    
    def countOccurances(self, token):
        count = 0
        if (token in self.__content.keys()):
            for doc, n in self.__content[token][1].items():
                count += n    
            return count
        else:
            return "Token not found"
            
    def delete(self, token):
        if token in self.__content:
            del self.__content[token]
            return f"Token {token} deleted successfully"
        else:
            return "Token not found"
    
    def rename(self, original_token, new_token):
        if original_token == new_token:
            return f"Original and new tokens are the same"
        elif original_token in self.__content:
            new_inserts = []
            for doc, frequency in self.__content[original_token][1].items():
                for i in range(0,frequency):
                    new_inserts.append(doc)
            for doc in new_inserts:
                self.insert(new_token, doc)
            self.delete(original_token)
            return f"Token {original_token} was renamed to {new_token} successfully"
        else:
            return f"Token {original_token} not found"
        
    def getTokens(self):
        return self.__content.items()
            
    def __addOccurance(self, inverted_list, doc):
        if doc in inverted_list[1]:
            inverted_list[1][doc] += 1
        else:
            inverted_list[1][doc] = 1
            inverted_list[0] += 1
    
    def __str__(self):
        result = ""
        for token, data in self.__content.items():
            result += f"Token: {token} -> n: {data[0]}\n"
            for doc, n in data[1].items():
                result += f"['{str(doc)}', {str(n)}],"
            result = result[:-1] + '\n\n'
        return result
