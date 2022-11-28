import joblib

# import of local modules
from .utils import word_to_tf_idf_vec


class LogReg:
    """
    Base model to predict Bitcoin sentiment usnig if-idf and logistic regression
    """

    def __init__(
        self, config: str, serialized_model: str = None, vectorizer: str = None
    ):
        """
        Initialize model configuration
        @config: str = config directory
        @serialized_model: str = model .pkl directory
        @vectorizer: str = vectorizer .pkl directory
        """
        self.config = config

        self.model = None
        if serialized_model:
            self.model = joblib.load(serialized_model)

        self.tokenizer = None
        if vectorizer:
            self.vectorizer = vectorizer

    def predict(self, sentence: str) -> str:
        """
        Return news sentiment
        @param sentence: str = unique password
        @return: str = predicted category
        """
        vector = word_to_tf_idf_vec(self.vectorizer, sentence)
        y_pred = self.model.predict(vector)[0]

        # categorize
        keys = [0, 1, 2]
        vals = ["neutral", "positive", "negative"]
        pred_dict = dict(zip(keys, vals))

        return pred_dict[y_pred]

    def _load_data():
        pass

    def _process_data():
        pass

    def _fit_vectorizer():
        pass

    def save_vectorizer():
        pass

    def _split_data():
        pass

    def fit_model():
        pass

    def save_model():
        pass


# print(LogReg().predict(sentence='12vgv'))
