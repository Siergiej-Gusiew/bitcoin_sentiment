import joblib


def word_to_tf_idf_vec(vectorizer: str, news_headline: str):
    """
    Return sequence of tokens from password
    @param vectorizer: str = directory of vectorizer .pkl
    @param news_headline: str = text for vectorization
    @return: scipy.sparse._csr.csr_matrix equal to List[List[int]]
    """
    with open(vectorizer, "rb") as handle:
        vectorizer = joblib.load(handle)
    vector = vectorizer.transform([news_headline])
    return vector
