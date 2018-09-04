import atexit
import sqlite3
from datetime import datetime, timedelta

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from config import config


class Preparer:
    def __init__(self, time_intervals, **config):
        self.list_categories = config['config_dqnn']['tags']
        self.config = config
        self.time_intervals = time_intervals
        self.batch_size = config['batch_size']
        self.silent_mode = config['silent_mode']

        self.connection = sqlite3.connect(config['db_filename'])
        self.cursor = self.connection.cursor()
        self.enc = OneHotEncoder()

        self.data_to_dump = {}
        self.table_for_dqnn = None
        atexit.register(self.drop_created_table)

        self.flag = True
        self.enc_db_was_created = False
        self.iteration = 1
        # Current rows in each interval
        self.curr_row = []
        # Current interval
        self.curr_int = 0
        self.row_counts = []
        # Get row count in each interval
        for start, end in time_intervals:
            self.cursor.execute('''
            SELECT 
                count(*) 
            FROM 
                reddit_comments 
            WHERE 
                time >= "{0}" and time <= "{1}"'''.format(
                start, end
            ))
            self.row_counts.append(self.cursor.fetchall()[0][0])
            self.curr_row.append(0)

    def drop_created_table(self):
        print('''This method called because your instance going to delete
        In your config delete table after= {}'''.format(
            self.config['delete_table']
        ))
        if self.config['delete_table']:
            print('Droping table {}'.format(self.table_for_dqnn))
            self.cursor.execute("DROP TABLE IF EXISTS {}".format(
                self.table_for_dqnn
            ))
        self.connection.close()

    def get_encode_data_to_db(self):
        """
        Main method for put data in db with encode data
        :return:
            IF db with encrypted data wis created previously return empty array
            ELSE return rows and active flag status
        """
        if self.enc_db_was_created:
            return self.get_data_from_db()

        if self.flag:
            # Select all rows in given interval
            self.cursor.execute('''
            SELECT 
                comment_id 
            FROM 
                reddit_comments 
            WHERE 
                time >= "{0}" and time <= "{1}" LIMIT {2} OFFSET {3}'''.format(
                self.time_intervals[self.curr_int][0],
                self.time_intervals[self.curr_int][1],
                self.batch_size,
                self.curr_row[self.curr_int]
            ))
            rows = self.cursor.fetchall()
            enc_arr = []
            self.curr_row[self.curr_int] += len(rows)
            for row in rows:
                self.prepare_data_to_dump(row[0])
                enc_arr.append(self.data_to_dump)
                self.dump_to_db()

            # If the number of the current row is equal to the count of rows in
            # the interval, then go to the next interval
            if self.curr_row[self.curr_int] >= self.row_counts[self.curr_int]:
                self.curr_int += 1
                # If there no more rows
                if sum(self.curr_row) >= sum(self.row_counts):
                    self.flag = False
                    self.enc_db_was_created = True
            return enc_arr, self.flag
        else:
            return [], self.flag

    def get_data_from_db(self):
        self.cursor.execute('''
        SELECT 
            comment_id
        FROM 
            reddit_comments 
        WHERE 
            time >= "{0}" and time <= "{1}" LIMIT {2} OFFSET {3}'''.format(
            self.time_intervals[self.curr_int][0],
            self.time_intervals[self.curr_int][1],
            self.batch_size,
            self.curr_row[self.curr_int]
        ))
        ids = self.cursor.fetchall()
        rows = self.get_dqnn_rows(ids)
        self.curr_row[self.curr_int] += len(rows)
        if self.curr_row[self.curr_int] >= self.row_counts[self.curr_int]:
            self.curr_int += 1
            # If there no more rows
            if sum(self.curr_row) >= sum(self.row_counts):
                self.flag = False
            return rows, self.flag
        else:
            self.flag = False
            return [], self.flag

    def get_user_categories_enc(self, comment_id):
        """
        Function for mapping category by user comment
        :param comment_id: for initialise username_id
        :param period:
        :return:
            IF category not in encoded - return zero array
            ELSE mapped category by user when is active
        """
        try:
            self.enc.n_values_
        except AttributeError:
            print('First of all generate category enc before call this method')

        self.cursor.execute('''
        SELECT 
            username_id, time
        FROM
            reddit_comments
        WHERE
            comment_id = "{}"'''.format(
            comment_id
        ))
        user_id, start = self.cursor.fetchall()[0]

        time_start = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        time_end = time_start + timedelta(**config['config_dqnn']['period'])
        print("Extract uid={}, time between {} and {}".format(
            user_id, time_start, time_end
        ))
        self.cursor.execute('''
        SELECT 
            tag_id 
        FROM 
            reddit_comments 
        WHERE 
            time >= "{}" and time <= "{}" and username_id = "{}"
        GROUP BY
            tag_id
        '''.format(
            time_start, time_end, user_id
        ))
        categories = self.cursor.fetchall()

        # output data
        print("Found comments in categories_id {}".format(categories))
        categories = [d for d in categories if d[0] in self.enc.active_features_]
        if not categories:
            return np.zeros(len(self.enc.active_features_))
        categories = self.enc.transform(categories).toarray()
        return np.any(categories, axis=0).astype(float)

    def generate_category_features(self):
        self.enc.fit(self.list_categories)
        return self.enc.n_values_

    def prepare_data_to_dump(self, comment_id):
        if not self.silent_mode:
            assert not self.data_to_dump, 'Please check previous data. ' \
                                          'Successfully completed?'
        self.data_to_dump['id'] = comment_id
        self.data_to_dump['categories_enc'] = self.get_user_categories_enc(comment_id)

    def dump_to_db(self):
        if not self.table_for_dqnn:
            self.table_for_dqnn = 'dqnn_data_{}'.format(self.config['name'])
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS {tn} (
                comment_id INTEGER PRIMARY KEY,
                categories_enc BLOB
            )'''.format(tn=self.table_for_dqnn))
            self.connection.commit()

        self.cursor.execute('''
        INSERT OR REPLACE INTO {} (comment_id, categories_enc)
            VALUES (?, ?)
        '''.format(self.table_for_dqnn),
                            (
                                self.data_to_dump['id'],
                                self.data_to_dump['categories_enc'].tostring()
                            )
        )
        self.connection.commit()

        if not self.silent_mode:
            print('Succesfuly dump row with id={}'.format(
                self.data_to_dump['id']
            ))
        self.data_to_dump = {}

    def next_iteration(self):
        self.iteration += 1
        self.curr_int = 0
        self.curr_row = [0] * len(self.curr_row)
        self.flag = True
        return self.flag

    def generate_predictable_categories(self, top=None, custom=None):
        """
        Calculate category list all or top. Default - read category from config
        :param top: how much rows returned in class field
        :param custom: if need to create custom list. send directly into class field
        :return:
        """
        if custom:
            self.list_categories = custom

        self.cursor.execute('''
        SELECT 
            tag_id
        FROM
            reddit_comments
        GROUP BY tag_id
        {}'''.format('ORDER BY count(*) DESC LIMIT {}'.format(top) if top else ''))
        self.list_categories = self.cursor.fetchall()

        if not self.silent_mode:
            print('List predictable categories = {}'.format(self.list_categories))

    def get_dqnn_one_row(self, comment_id):
        self.cursor.execute('''
        SELECT
            *
        FROM
            {}
        WHERE
            comment_id={}
        '''.format(self.table_for_dqnn, comment_id))
        row = self.cursor.fetchall()[0]
        decoded_row = [row[0]]
        for cell in row[1:]:
            decoded_row.append(np.fromstring(cell, dtype=float))
        return decoded_row

    def get_dqnn_rows(self, array_ids):
        res = []
        for id in array_ids:
            if not isinstance(id, int):
                try:
                    id = id[0]
                except TypeError:
                    print('Something goes wrong in array_ids')
            res.append(self.get_dqnn_one_row(id))
        return res


if __name__ == '__main__':
    prep = Preparer([['2017-01-01 05:25:49', '2017-01-02 13:37:51'],
                     ['2017-01-02 23:47:23', '2017-01-08 11:11:02']], **config)

    prep.generate_predictable_categories(5)
    prep.generate_category_features()
    flag = True
    while flag:
        data, flag = prep.put_encode_data_to_db()
        print(len(data), flag)

    # prep.next_iteration()
    pass
