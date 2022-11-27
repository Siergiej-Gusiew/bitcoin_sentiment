class LogReg:
    """
    Base model to predict Bitcoin sentiment usnig if-idf and logistic regression
    """

    def __init__(self):
        pass

    def predict(self, sentence: str) -> int:
        """
        return label of sentence
        """
        if len(sentence) > 10:
            return "Long"
        else:
            return "Short"


# print(LogReg().predict(sentence='12vgv'))
