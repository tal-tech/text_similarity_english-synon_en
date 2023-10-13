from unique_token import Constants
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


# nltk.download('wordnet')
# init lemmatizer
# lemmatizer = WordNetLemmatizer()
# porter_stemmer=PorterStemmer()

def scrub_words(text):
    """Basic cleaning of texts."""

    # remove html markup
    text = re.sub("(<.*?>)", "", text)

    # remove non-ascii and digits
    text = re.sub("(\\W|\\d)", " ", text)

    # remove whitespace
    text = text.strip()
    return text


# version 2.0 add stemming stopword, remove number, convert to lower case, lemmatization
def seg_sentence(sentence):
    return list(s for s in sentence.split())

class sentence():
    def __init__(self, max_sent_len=30, lemma=False, stem=False, lower=False):
        self.max_sentence_len = max_sent_len
        self.lemma, self.stem, self.lower = lemma, stem, lower

    def tokenized(self, text):
        if self.lower:
            raw_sentence = scrub_words(text.lower())
        else:
            raw_sentence = scrub_words(text)

        raw_words = raw_sentence.split()
        '''
        if self.lemma:
            raw_words = [lemmatizer.lemmatize(word=word, pos='v') for word in raw_words]
        else:
            pass

        if self.stem:
            raw_words = [porter_stemmer.stem(word=word) for word in raw_words]
        else:
            pass
        '''
        return raw_words

    def tokenize_sentence(self, text):
        # signal is the signal for pad or not
        signal = 0
        words_1 = self.tokenized(text)
        if len(words_1) > self.max_sentence_len:
            words_1 = words_1[:self.max_sentence_len]
            # print('trimming sentence 1')
        else:
            words_1 += [Constants.PAD_WORD] * (self.max_sentence_len - len(words_1))
            signal = 1

        if words_1:
            return words_1, signal
        else:
            return None
