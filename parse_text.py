import pickle
import numpy as np
from preprocess import text_preprocess
from cls_rule_based import ner_sent
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from pyvi import ViTokenizer

# vectorize text
with open("vectorize_word.pkl", "rb") as f:
    vectorize_sent = pickle.load(f)

# model
with open("save_model\LogisticRegression.pkl", "rb") as f:
    Classifier = pickle.load(f)

test_text_3 = "Quán ăn ngon. Quán ăn bình thường. Quán ăn dở"
test_text_2 = "Quán ăn ngon"
test_text_1 = "cửa hàng dơ"
test_text_0 = "Quán ăn bình thường"


def text_to_sents(text):
    return [sent for sent in text.replace("\n", ".").split(".")]


def process_sent(sent):
    return ViTokenizer.tokenize(text_preprocess(sent))


def sent_to_vector(sent):
    sent = text_preprocess(sent)
    return vectorize_sent.transform([sent]).toarray()


def cls_sent(sent):
    return Classifier.predict_proba(sent_to_vector(sent))


def parse_text_api(text):
    sents = text_to_sents(text)
    sents = [process_sent(sent) for sent in sents]
    result = {}
    for sent in sents:
        ner_result = ner_sent(sent)
        cls_result = cls_sent(sent)
        result[sent] = {
            "ner_result": ner_result,
            "cls_result": np.squeeze(cls_result).tolist()
        }
    return result
