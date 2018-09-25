import sqlite3
import datetime
from math import floor
from random import randint


import dictionary_db
from preparer import Preparer
from modeling import Model
from dqnn import ModelDQNN



def get_cv_data(db_file, percent=0., parts_count=0):
    """
    Function for getting date range from database
    :param percent: 0 < x < 1 - percent of test data count
    :param parts: number of equal parts
    :return:
        IF percent - start date and end date of posts in test set
        IF parts - len(parts_count) arrays with dates on test set
    """

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('SELECT count(comment_id) FROM reddit_comments')
    data_size = cursor.fetchall()[0][0]
    if percent > 0:
        test_size = floor(data_size * percent) - 1
        start_index = randint(0, data_size - test_size)
        train_dates, test_dates = get_dates(cursor, start_index, test_size)
        return train_dates, test_dates
    else:
        train_dates = []
        test_dates = []
        test_size = floor(data_size / parts_count) - 1
        remainder = data_size - test_size * parts_count
        for k in range(parts_count - 1):
            start_index = k * test_size
            train_d, test_d = get_dates(cursor, start_index, test_size)
            train_dates.append(train_d)
            test_dates.append(test_d)

        # Fill final interval including remainder
        start_index = (parts_count - 1) * test_size
        train_d, test_d = get_dates(cursor, start_index, test_size + remainder)
        train_dates.append(train_d)
        test_dates.append(test_d)

        return train_dates, test_dates


def get_dates(cursor, start_index, test_size):
    # Select min date in obtained range
    cursor.execute('''
    SELECT 
        time 
    FROM 
        reddit_comments 
    ORDER BY 
        time 
    ASC LIMIT 1 OFFSET {0}'''.format(start_index))
    start_date = datetime.datetime.strptime(cursor.fetchall()[0][0], "%Y-%m-%d %H:%M:%S")

    # Select max date in obtained range
    cursor.execute('''
    SELECT 
        time 
    FROM 
        reddit_comments 
    ORDER BY 
        time 
    ASC LIMIT 1 OFFSET {0}'''.format(start_index + test_size - 1))
    end_date = datetime.datetime.strptime(cursor.fetchall()[0][0], "%Y-%m-%d %H:%M:%S")

    # Select min date from all data
    cursor.execute('''
    SELECT 
        time 
    FROM 
        reddit_comments 
    ORDER BY 
        time 
    ASC LIMIT 1''')
    min_date = datetime.datetime.strptime(cursor.fetchall()[0][0], "%Y-%m-%d %H:%M:%S")

    # Select max date from all data
    cursor.execute('''
    SELECT 
        time 
    FROM 
        reddit_comments 
    ORDER BY 
        time 
    DESC LIMIT 1''')
    max_date = datetime.datetime.strptime(cursor.fetchall()[0][0], "%Y-%m-%d %H:%M:%S")

    test_dates = [start_date, end_date]
    train_dates = [[min_date, start_date], [end_date, max_date]]
    return train_dates, test_dates


class CrossValidationDQNN:
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def train(self, model_dqnn, intervals):
        finish = False
        prepare = Preparer(intervals, **self.config)
        prepare.generate_category_features()
        self.all_categories = [categorie[0] for categorie in prepare.list_categories]
        while not finish:
            # get events
            rows, flag = prepare.get_encode_data_to_db()
            batch_states = []
            batch_newstates = []
            batch_actions = []
            for row in rows:
                # get info about event
                id_event = row['id']
                event = self.model._dictionary.get_coment(id_event)
                time_event = event['time']
                user_id = event['username_id']

                time_delta = datetime.timedelta(hours=1)
                # init features
                time_state = time_event - datetime.timedelta(seconds=1)
                time_next_state = time_event + time_delta

                categories = self.model.get_read_categories(user_id, time_state, time_next_state, self.all_categories)
                for tag_id in categories:
                    action = categories[tag_id]
                    state = self.model.get_features(user_id, tag_id, time_state)
                    if state is None:
                        continue
                    next_state = self.model.get_features(user_id, tag_id, time_next_state)
                    batch_states.append(state)
                    batch_newstates.append(next_state)
                    batch_actions.append(action)

            if len(batch_states) > 0:
                model_dqnn.train(batch_states, batch_newstates, batch_actions)
            if not flag:
                finish = prepare.next_iteration()

    def predict(self, model_dqnn, intervals):
        finish = False
        estimator = Estimator()
        prepare = Preparer(intervals, **self.config)
        while not finish:
            # get events
            rows, flag = prepare.get_data_from_db()
            for row in rows:
                # get info about event
                time_event = None
                user_id = None
                time_delta = None
                time_state = time_event - datetime.timedelta(seconds=1)
                time_next_state = time_event + time_delta
                categories = self.model.get_read_categories(user_id, time_state, time_next_state, self.all_categories)
                result = {}
                for tag_id in categories:
                    action = categories[tag_id]
                    state = self.model.get_features(user_id, tag_id, time_state)
                    predict = model_dqnn.predict(state)

    def run(self):
        self.init_interval()
        self.all_categories = self.model.get_id_categories()
        size_state = self.model.get_size_future()
        for train_interval, test_interval in zip(self.train_interaval, self.test_interval):
            # Todo: init log dir
            log_dir = 'data/log'
            dqnn = ModelDQNN(log_dir)
            dqnn.init_dqnn((size_state, 1), 1)
            # dqnn.init_dqnn((size_state, 1, 1), 1)
            # dqnn.init_dqnn((84, 84, 1), 1)
            self.train(dqnn, train_interval)
            estimate = self.predict(dqnn, test_interval)
            dqnn.close()


    def init_interval(self):
        db =  self.model._dictionary.db_file
        train_dates, test_dates = get_cv_data(db, parts_count=3)
        self.train_interaval = train_dates
        self.test_interval = test_dates

    def prepare(self):
        pass





    # Add data
    # get_cv_data
    # Preparep
    # Train dqnn
    # predict dqnn



if __name__ == '__main__':
    dictionary = dictionary_db.Dictionary('db/db.sqlite', '')
    dictionary.add_csv_data('data/csv/test.csv')
    db = dictionary.db_file
    train_dates, test_dates = get_cv_data(db, parts_count=3)
    pass