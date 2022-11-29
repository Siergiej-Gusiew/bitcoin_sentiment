import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from loguru import logger

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

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(
            self.config["data_input"]["all_data"],
            encoding="ISO-8859-1",
            names=["Sentiment", "News Headline"],
        )
        logger.info("load data .. done")
        return df

    def _decode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        keys = ["neutral", "positive", "negative"]
        vals = [0, 1, 2]
        map_dict = dict(zip(keys, vals))

        df.Sentiment = df.Sentiment.map(map_dict)
        logger.info("decode target .. done")
        return df

    def _split_data(self, df: pd.DataFrame) -> tuple:
        """
        @return: tuple(pd.DataFrame)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            df["News Headline"],
            df.Sentiment,
            test_size=self.config["model_input"]["test_size"],
            random_state=self.config["model_input"]["rnd_seed"],
            shuffle=True,
            stratify=df.Sentiment,
        )
        logger.info("split data .. done")
        return (X_train, X_test, y_train, y_test)

    def _tf_idf(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        """
        Fit vectorizer and vectorize X
        @param X_train: pd.DataFrame
        @param X_test: pd.DataFrame
        @return: tuple(*, *), where * is scipy.sparse._csr.csr_matrix
        """
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=self.config["vectorizer_input"]["ngram_range"],
            lowercase=True,
            max_features=self.config["vectorizer_input"]["max_features"],
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        self.vectorizer = vectorizer

        logger.info("vectorization data .. done")
        return (X_train_vec, X_test_vec)

    # def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
    #     return df

    # def _fit_vectorizer():
    #     # Look at _tf_idf function - don't duplicate
    #     pass

    def fit_model(self) -> None:
        """
        @param df_path: str = example, 'data/raw/all-data.csv'
        @return: None
        """
        df = self._load_data()
        df = self._decode_target(df)
        X_train, X_test, y_train, y_test = self._split_data(df)
        X_train, X_test = self._tf_idf(X_train, X_test)

        logreg = LogisticRegression(
            C=self.config["model_input"]["regularization_strength"],
            solver="lbfgs",
            multi_class="multinomial",
            class_weight=self.config["model_input"]["class_weight"],
            random_state=self.config["model_input"]["rnd_seed"],
            n_jobs=-1,
        )
        logger.info("train model .. start")

        logreg.fit(X_train, y_train)
        self.model = logreg

        logger.info("train model .. done")
        return None

    def save_vectorizer(self, vectorizer_path: str) -> None:
        """
        Save vectorizer in vectorizer_path
        @param vectorizer_path: str = example, 'model/vectorizer.pkl'
        @return: None
        """
        joblib.dump(self.vectorizer, vectorizer_path)
        logger.info("save vectorizer .pkl .. done")
        return None

    def save_model(self, model_path: str) -> None:
        """
        Save model in model_path
        @param model_name: str = example, 'model/base_model.pkl'
        @return: None
        """
        joblib.dump(self.model, model_path)
        logger.info("save model .pkl .. done")
        return None
