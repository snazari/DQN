import datetime
import os
import pickle
import uuid
from random import randint

import numpy as np
from gensim import models
from sklearn.preprocessing import OneHotEncoder

import dictionary_db
from processing import Processing
from utils import make_dirs
from random import randint

import sqlite3
# import datetime

# import tfidfmodel_db
import dictionary_db
from dqnn import ModelDQNN

import uuid


# import psutil # for DEBUG TODO remove
# def printmem(): # for DEBUG TODO remove
#     process = psutil.Process(os.getpid())
#     print(process.memory_info().rss)


class Model(object):

    def __init__(self, config=None, db_file=os.path.join("data", "db", "reddit.sqlite"), model_file=None):
        self.model = None
        self.model_id = None
        if config:
            self.config = config
        else:
            if model_file:
                with open(model_file, 'rb') as f:
                    p = pickle.load(f)
                self.config = p[0]
                self.model = p[1]
            else:
                print("Neither config nor model_file specified")
                return

        self.model_name = self.config['config_general']['model_type']
        self.occurrences_threshold = self.config['config_general']['occurrences_threshold']

        self._dictionary = dictionary_db.Dictionary(db_file, self.config['config_processing'])   # new db-based code
        self._dictionary.init_word_id_index(self.occurrences_threshold)
        if model_file:
            self.model_id = str(uuid.uuid4())
            found = None
            if os.path.isfile(db_file):
                found = self._dictionary.find_model(self.model_name, self.occurrences_threshold,
                                                   **self.config['config_model'])
            else:
                print("DB file is not found, please provide DB related to the model")
                return
            if not found:
                self.save_model(self.model_id)
                self._dictionary.add_model(self.model_name, self.model_id, self.occurrences_threshold,
                                           **self.config['config_model'])
                print("Added model with id " + self.model_id)
            else:
                print(
                    "Model with similar config found in DB, so it will not be added to DB again")

    @property
    def weekdays_onehot(self):
        enc = OneHotEncoder()
        enc.fit([[1], [2], [3], [4], [5], [6], [7]])
        return enc

    @property
    def time_in_day(self):
        return {'hours': 24,
                'minutes': 24 * 60,
                'seconds': 24 * 60 * 60
                }

    def init_onehot(self, config=None, model_file=None):
        self.onehot_model_id = None
        if model_file:
            with open(model_file, 'rb') as f:
                p = pickle.load(f)
            model_id = p[0]
            self.onehot_model_id = model_id
        else:
            if config is None:
                config = self.config['config_onehot']
            model_id = str(uuid.uuid4())
            self.create_usarname_tag_onehot_tables(model_id, usernames=config['usernames'], tags=config['tags'])
            self.save_onehot(model_id, config)
            self.onehot_model_id = model_id

    def add_csv_data(self, csv_file_path):
        print("Start rebuilding dictionary db")
        self._dictionary.add_csv_data(csv_file_path)
        print(" - done")
        self._dictionary.init_word_id_index(self.occurrences_threshold)
        # printmem() # for DEBUG, TODO remove

    def TfIdf(self, **config):
        normalize = config['normalize']
        # self.model = models.TfidfModel(self.corpus, normalize=normalize)
        # self.model = tfidfmodel_db.TfidfModel("data/db/reddit.sqlite", d=self._dictionary, normalize=normalize)
        self.model = models.TfidfModel(dictionary=self._dictionary, normalize=normalize)

    def LSI(self, **config):
        num_topics = config['num_topics']
        power_iters = config['power_iters']
        extra_samples = config['extra_samples']
        tfidf = models.TfidfModel(self._dictionary.corpus)
        corpus_tfidf = tfidf[self._dictionary.corpus]
        self.model = models.LsiModel(corpus_tfidf, id2word=self._dictionary, num_topics=num_topics,
                                     power_iters=power_iters, extra_samples=extra_samples)
        # printmem()  # for DEBUG, TODO remove

    def RP(self, **config):
        num_topics = config['num_topics']
        self.model = models.RpModel(self._dictionary.corpus, num_topics=num_topics)

    def LDA(self, **config):
        num_topics = config['num_topics']
        distributed = config['distributed']
        alpha = config['alpha']
        eta = config['eta']
        self.model = models.LdaModel(
            self._dictionary.corpus, num_topics=num_topics, distributed=distributed, alpha=alpha, eta=eta)

    def HDP(self, **config):
        gamma = config['gamma']
        kappa = config['kappa']
        tau = config['tau']
        K = config['K']
        T = config['T']
        eta = config['eta']
        self.model = models.HdpModel(
            self._dictionary.corpus, id2word=self._dictionary, gamma=gamma, kappa=kappa, tau=tau, K=K,
                                     T=T, eta=eta)

    def train_model(self):
        if self.model:
            print("Model passed to ctor, no need in training")
            return

        # db search for saved model
        model_id = self._dictionary.find_model(self.model_name, self.occurrences_threshold, **self.config['config_model'])
        if model_id:
            # model found, loading
            self.load_model(model_id)
            print("Found appropriate trained model with id " + model_id)
        else:
            # model not found, creating new one
            print("No appropriate trained model found, creating new one...")
            model_id = str(uuid.uuid4())
            train_fun = getattr(self, self.model_name)
            train_fun(**self.config['config_model'])  # word_id index table for self.model_id has to be created here
            self.save_model(model_id, None)
            self._dictionary.add_model(self.model_name, model_id, self.occurrences_threshold, **self.config['config_model'])
            print("Created model with id " + model_id)

        # printmem()  # for DEBUG, TODO remove

    def save_model(self, model_id, config):
        model_path = make_dirs('model', self.model_name)
        # self.model.save(os.path.join(model_path, model_id + '_model.pickle'))
        with open(os.path.join(model_path, model_id + '_model.pickle'), 'wb') as f:
            pickle.dump([self.config, self.model], f, protocol=pickle.HIGHEST_PROTOCOL)
        self.model_id = model_id  # in order to locate index table
        # printmem()  # for DEBUG, TODO remove

    def load_model(self, model_id):
        model_path = make_dirs('model', self.model_name)
        # if self.model_name == 'TfIdf':
        #     p = models.TfidfModel.load(os.path.join(model_path, model_id + '_model.pickle'))
        # elif self.model_name == 'LSI':
        #     p = models.LsiModel.load(os.path.join(model_path, model_id + '_model.pickle'))
        # elif self.model_name == 'RP':
        #     p = models.RpModel.load(os.path.join(model_path, model_id + '_model.pickle'))
        # elif self.model_name == 'LDA':
        #     p = models.LdaModel.load(os.path.join(model_path, model_id + '_model.pickle'))
        # elif self.model_name == 'HDP':
        #     p = models.HdpModel.load(os.path.join(model_path, model_id + '_model.pickle'))
        with open(os.path.join(model_path, model_id + '_model.pickle'), 'rb') as f:
            p = pickle.load(f)
        # self.config = p[0]
        self.model = p[1]
        self.model_id = model_id  # in order to locate index table
        # printmem()  # for DEBUG, TODO remove

    def save_onehot(self, model_id, config):
        model_path = make_dirs('onehot', self.model_name)
        with open(os.path.join(model_path, model_id + '_model.pickle'), 'wb') as f:
            pickle.dump([model_id, config], f, protocol=pickle.HIGHEST_PROTOCOL)

    def transforming(self, data):
        if isinstance(data, str):
            processor = Processing(**self.config['config_processing'])
            _, data = processor(data)
        if isinstance(data, dictionary_db.Corpus_db):
            corpus = data
        else:
            if isinstance(data[0], list):
                corpus = [
                    self._dictionary.processed_doc2bow(token, self.config['config_general']['occurrences_threshold'])
                    for token in data]
            else:
                corpus = self._dictionary.processed_doc2bow(data,
                                                            self.config['config_general']['occurrences_threshold'])
        if self.model_name in ['TfIdf', 'LDA', 'HDP']:
            model = self.model[corpus]
            if not isinstance(model, list):
                vectors = [vector for vector in model]
                return vectors
            else:
                return model
        elif self.model_name in ['LSI', 'RP']:
            model = models.TfidfModel(self._dictionary.corpus, normalize=True)
            corpus_tfidf = model[corpus]
            if not isinstance(corpus_tfidf, list):
                corpus = [token for token in corpus_tfidf]
            else:
                corpus = corpus_tfidf
            predict = self.model[corpus]
            if not isinstance(predict, list):
                vectors = [vector for vector in predict]
                return vectors
            else:
                return predict
        # printmem()  # for DEBUG, TODO remove

    def get_results(self):  # TODO

        # predict = []
        doc_id = -1
        self._dictionary.create_result_table(self.config['config_model']['num_topics'], self.model_id)
        if self.model_name in ['LSI', 'RP']:
            model_tfidf = models.TfidfModel(self._dictionary.corpus, normalize=True)
            for doc in self._dictionary.corpus:
                doc_id += 1
                corpus_tfidf = model_tfidf[doc]
                vector = self.model[corpus_tfidf]
                self._dictionary.save_result(vector, self.config['config_model']['num_topics'], self.model_id, doc_id)
                # predict.append(vector)
        else:
            for doc in self._dictionary.corpus:
                doc_id += 1
                vector = self.model[doc]
                self._dictionary.save_result(vector, self.config['config_model']['num_topics'], self.model_id, doc_id)
                # predict.append(vector)
        # print(predict)

    def create_usarname_tag_onehot_tables(self, model_id, usernames, tags):
        self._dictionary.create_usarname_tag_onehot_tables(model_id, usernames, tags)

    def compute_tag_features(self, tag_id, steps_time, to_date=None):
        if to_date:
            current_time = to_date
        else:
            current_time = datetime.datetime.now()
        counts = []
        averages_topic = []
        averages_user = []
        for step_time in steps_time:
            from_date = current_time - step_time
            count = self._dictionary.get_count_comments_by_tag(tag_id, from_date, to_date)
            counts.append(count)
            average_topic = self._dictionary.get_average_topic_by_tag(self.model_id, tag_id, from_date, to_date)
            averages_topic += average_topic.tolist()
            average_user = self._dictionary.get_average_users_onehot_by_tag(self.onehot_model_id, tag_id, from_date, to_date)
            averages_user += average_user.tolist()

        last_comment = self._dictionary.get_last_coment_by_tag(tag_id, to_date)
        diff_time = current_time.timestamp() - last_comment
        tag_features = averages_topic + averages_user + counts + [diff_time]
        return tag_features

    def compute_user_features(self, user_id, steps_time, to_date=None):
        if to_date:
            current_time = to_date
        else:
            current_time = datetime.datetime.now()
        averages_tag = []
        averages_topic = []
        for step_time in steps_time:
            from_date = current_time - step_time
            average_tag = self._dictionary.get_average_tag_onehot_by_user(self.onehot_model_id, user_id, from_date, to_date)
            averages_tag += average_tag.tolist()
            average_topic = self._dictionary.get_average_topic_by_user(self.model_id, user_id, from_date, to_date)
            averages_topic += average_topic.tolist()
        user_features = averages_tag + averages_topic
        return user_features

    def compute_user_tag_features(self, user_id, tag_id, to_date=None):
        if to_date:
            current_time = to_date
        else:
            current_time = datetime.datetime.now()
        last_time = []

        last_time_tag = self._dictionary.get_last_coment_by_user(user_id, tag_id, to_date)
        diff_time = current_time.timestamp() - last_time_tag
        last_time.append(diff_time)
        average_tag = self._dictionary.get_average_tag_of_user(self.model_id, user_id, tag_id, to_date)
        user_tag_features = last_time + average_tag
        return user_tag_features

    def compute_context_features(self, steps_time, to_date=None):
        if to_date:
            current_time = to_date
        else:
            current_time = datetime.datetime.now()
        averages_topic = []
        averages_user = []
        time = {}
        for step_time in steps_time:
            from_date = current_time - step_time
            average_topic = self._dictionary.get_average_topic(self.model_id, from_date, to_date)
            averages_topic += average_topic.tolist()
            average_user = self._dictionary.get_average_tag_onehot(self.onehot_model_id, from_date, to_date)
            averages_user += average_user.tolist()
        proportion = (current_time.hour * 60 + current_time.minute) \
                     / self.time_in_day['minutes']
        time['sin'] = np.sin(2 * np.pi * proportion)
        time['cos'] = np.cos(2 * np.pi * proportion)
        week = self.weekdays_onehot.transform([[current_time.weekday()]]).toarray()
        context_features = [week, time['sin'], time['cos']] + averages_topic \
                           + averages_user
        return context_features

    def get_features(self, user_id, tag_id, to_date=None):
        steps_time = self.config['config_dqnn']['steps_time']
        if to_date:
            current_time = to_date
        else:
            current_time = datetime.datetime.now()
        tag_features = self.compute_tag_features(tag_id, steps_time, current_time)
        user_features = self.compute_user_features(user_id, steps_time, current_time)
        user_tag_features = self.compute_user_tag_features(user_id, tag_id, current_time)
        context_features = self.compute_context_features(steps_time, current_time)
        return user_features + tag_features + user_tag_features + context_features

    def get_part_of_data(self, percent=0, parts=0):
        """
        Function for split data from database
        :param percent: 0 < x < 1
        :param parts: number of equal parts
        :return:
            IF percent - one array of train_data and
            IF parts - 2 arrays of arrays with equals splitting parts
        """
        assert (not percent or not parts), "Please use only one method to split data"
        train_data = self._dictionary.get_ordered_table_by_name('reddit_comments', 'time')

        # return two sets
        if 0 < percent < 1:
            test_data = []
            part = int(len(train_data) * percent)
            rand_index = randint(0, len(train_data) - part)
            for _ in range(part):
                test_data.append(train_data.pop(rand_index))
            return test_data, train_data

        # return 'parts' arrays
        if parts > 0:
            part = int(len(train_data) / parts)
            train_arr = []
            test_arr = []
            additional_elems = len(train_data) - part * parts
            additional_i = 0

            for i in range(parts):
                tmp_train_data = train_data.copy()
                tmp_test_data = []
                if additional_elems > 0:
                    part_mod = part + 1
                    additional_i += 1
                    additional_elems -= 1
                else:
                    part_mod = part

                for _ in range(part_mod):
                    tmp_test_data.append(tmp_train_data.pop(i*part + additional_i))
                test_arr.append(tmp_test_data)
                train_arr.append(tmp_train_data)
            return test_arr, train_arr

    def get_partition_of_data(self, number_of_parts):
        array_of_parts = []
        test_data = None
        train_data = None

        array_of_parts.append(test_data, train_data)
        return array_of_parts


class ModelingDQNN:
    def __init__(self, config):
        self.model = Model(config)
        self.model_dqnn = ModelDQNN()
        self.model_dqnn.init_dqnn(None, (1))

    def train(self):
        pass

    def add_csv_data(self, file_path):
        self.model.add_csv_data(file_path)

