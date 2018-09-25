import os
import re
import string

try:
    from pattern3.text import en
except:
    from pattern.text import en
from polyglot.text import Text
from polyglot import load
from nltk.stem.wordnet import WordNetLemmatizer

# from config import config_processing as params


class Processing(object):

    def __init__(self, **config):
        self.delete_punctuation_marks = config['delete_punctuation_marks']
        self.delete_numeral = config['delete_numeral']
        self.delete_single_words = config['delete_single_words']
        self.initial_form = config['initial_form']
        self.stop_words = config['stop_words']
        # self.set_words = set()

    def __call__(self, document):
        if self.delete_punctuation_marks:
            c = re.compile('[{}]'.format(re.escape(string.punctuation)))
            document = c.sub('', document)

        if self.delete_numeral:
            c = re.compile('[{}]'.format(re.escape(string.digits)))
            document = c.sub('', document)

        document = document.lower()
        texts = document.split()

        if self.stop_words is not None:
            texts = [word for word in texts if word not in self.stop_words]

        words = []
        load.polyglot_path = os.path.join(os.getcwd(), 'polyglot_data')
        os.path.sep = '/'
        n = ['NOUN', 'PRON', 'PROPN']
        a = ['ADJ', 'ADP']
        v = ['VERB', 'ADV', 'AUX']
        #     initial_texts = []
        #     types = []
        #     for token in texts:
        #         pos_tag = Text(token, 'en').pos_tags[0][1]
        #         if pos_tag in n:
        #             initial_texts.append(en.singularize(token))
        #         elif pos_tag in a:
        #             initial_texts.append(WordNetLemmatizer().lemmatize(token, 'a'))
        #         elif pos_tag in v:
        #             initial_texts.append(en.lemma(token))
        #         else:
        #             initial_texts.append(token)
        #         types.append(pos_tag)
        #
        #     texts = initial_texts
        #
        # # if self.delete_single_words:
        # #     # self.set_words = set(texts).difference(self.set_words)
        # #     texts = list(set(texts).difference(self.set_words))
        # #     self.set_words.update(set(texts))
        #
        # return texts, types

        for token in texts:
            pos_tag = Text(token, 'en').pos_tags[0][1]
            if pos_tag in n:
                word = en.singularize(token)
            elif pos_tag in a:
                word = WordNetLemmatizer().lemmatize(token, 'a')
            elif pos_tag in v:
                word = en.lemma(token)
            else:
                word = token
            # print(token + " -> " + word)
            # if word == "ymy":  # TODO remove and fix 'ymy' bug
            #     word = 'your'
            word = (word, pos_tag)
            words.append(word)

        unique_words = list(set(words))
        occurrences = sorted([(x[0], x[1], words.count(x)) for x in unique_words], key=lambda y: y[2], reverse=True)
        if self.initial_form:
            return [w[0] for w in words], occurrences
        else:
            return texts, occurrences



