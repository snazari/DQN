
import sys
from collections import defaultdict

import numpy

# from six import PY3, iteritems, iterkeys, itervalues, string_types
if sys.version_info[0] >= 3:
    unicode = str

import sqlite3
import os
import datetime
import csv

import processing
# from config import config_processing

from sklearn.preprocessing import OneHotEncoder


class Dictionary:

    def __init__(self, db_file, config_processing):
        self.config_processing = config_processing
        self.db_file = db_file
        self.occ_threshold = 0
        self.corpus = Corpus_db(self)

    @property
    def dfs(self):
        dfs = {}
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()

        if self.occ_threshold == 0:
            c.execute("SELECT * FROM global_dict")
            c2 = conn.cursor()
            for row in c:
                c2.execute("SELECT count(*) FROM occurrences WHERE word_id=" + str(row[0]))
                try:
                    dfs[row[0] - 1] = c2.fetchone()[0]
                except:
                    pass
        else:
            c.execute("SELECT global_dict.word_id, word_id_index_" + str(self.occ_threshold)
                      + ".word_id_out FROM global_dict INNER JOIN word_id_index_" + str(self.occ_threshold)
                      + " ON global_dict.word_id = word_id_index_" + str(self.occ_threshold) + ".word_id")
            c2 = conn.cursor()
            for row in c:
                c2.execute("SELECT count(*) FROM occurrences WHERE word_id=" + str(row[0]))
                try:
                    dfs[row[1] - 1] = c2.fetchone()[0]  # row[1] is word_id_out
                except:
                    pass

        conn.close()  # TODO try double join instead of loop
        return dfs

    @property
    def num_docs(self):
        docno = -1

        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()

        if self.occ_threshold == 0:
            c.execute("SELECT count(*) FROM reddit_comments")
        else:
            c.execute("SELECT count(*) FROM ("
                      + "SELECT DISTINCT occurrences.comment_id "
                      + "FROM occurrences INNER JOIN word_id_index_" + str(self.occ_threshold)
                      + " ON occurrences.word_id = word_id_index_" + str(self.occ_threshold) + ".word_id)")
            # count of documents that contain at least one word listed in word_id_index
        try:
            docno = c.fetchone()[0]
        except:
            pass


        conn.close()
        return docno + 1

    @property
    def num_nnz(self):
        numnnz = 0

        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()

        if self.occ_threshold == 0:
            c.execute("SELECT count(*) FROM occurrences")
        else:
            c.execute("SELECT count(*) FROM occurrences INNER JOIN word_id_index_" + str(self.occ_threshold)
                      + " ON occurrences.word_id = word_id_index_" + str(self.occ_threshold) + ".word_id")
        try:
            numnnz = c.fetchone()[0]
        except:
            pass

        conn.close()
        return numnnz

    # def doc2bow(self, document):  # left for compatibility, not in use, NO occ_threshold support!
    #     if isinstance(document, string_types):
    #         raise TypeError("doc2bow expects an array of unicode tokens on input, not a single string")
    #
    #     # Construct (word, frequency) mapping.
    #     counter = defaultdict(int)
    #     for w in document:
    #         counter[w if isinstance(w, unicode) else unicode(w, 'utf-8')] += 1
    #
    #     conn = sqlite3.connect(self.db_file)
    #     c = conn.cursor()
    #
    #     # for w in counter.keys():
    #     #     c.execute("SELECT * FROM global_dict WHERE word="+ w)
    #
    #     c.execute("SELECT * FROM global_dict")  #  TODO optimize: loop counter first and make separate queries for id (row[0]) of each term
    #
    #     result = []
    #     for row in c:
    #         if row[1] in counter:
    #             result.append((row[0]-1, counter[row[1]]))
    #     conn.close()
    #     return result

    def processed_doc2bow(self, processed_doc, threshold=0):
        # Construct (word, frequency) mapping.
        counter = defaultdict(int)
        for w in processed_doc:
            word = (w[0] if isinstance(w[0], unicode) else unicode(w[0], 'utf-8'),
                    w[1] if isinstance(w[1], unicode) else unicode(w[1], 'utf-8'))
            counter[word] += 1

        result = []

        if self.occ_threshold == 0:
            query_str = "SELECT word_id, global_occurrences FROM global_dict WHERE word='"
        else:
            query_str = "SELECT word_id_index_" + str(self.occ_threshold) \
                        + ".word_id_out, global_dict.global_occurrences FROM global_dict INNER JOIN word_id_index_" \
                        + str(self.occ_threshold) + " ON global_dict.word_id = word_id_index_" + str(self.occ_threshold) \
                        + ".word_id WHERE word='"
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        for w in counter.keys():
            c.execute(query_str + w[0] +"' AND type='" + w[1] + "'")
            word_id = None
            occ = 0
            for row in c:
                word_id = row[0]-1
                occ = row[1]
            if word_id and occ >= threshold:
                result.append((word_id, counter[w]))
        conn.close()
        return result

    def add_csv_data(self, csv_file_path):
        self._init_db()
        if os.path.isfile(csv_file_path):
            self.add_single_csv(csv_file_path)
        else:
            # isdir
            filenames = os.listdir(csv_file_path)
            for f in filenames:
                if f.endswith(".csv"):  # assume all .csv files in the folder have appropriate format
                    self.add_single_csv(os.path.join(csv_file_path, f))

    def _init_db(self):
        table_name = 'reddit_comments'
        if os.path.exists(self.db_file):
            os.remove(self.db_file)  # drop the entire db

        if not os.path.isfile(self.db_file):
            directory = '/'.join(os.path.split(self.db_file)[0:-1])
            if not os.path.exists(directory):
                os.makedirs(directory)

            conn = sqlite3.connect(self.db_file)
            c = conn.cursor()

            c.execute("PRAGMA foreign_keys = ON")

            # c.execute(
                # 'CREATE TABLE {tn} (comment_id INTEGER PRIMARY KEY, time TIMESTAMP(14), username TEXT, comment TEXT, tag TEXT)'  # , PRIMARY KEY (id)
                # .format(tn=table_name))

            c.execute('CREATE TABLE {tn} (comment_id INTEGER PRIMARY KEY, time TIMESTAMP(14), comment TEXT, username_id INTEGER,  tag_id INTEGER, FOREIGN KEY (tag_id) REFERENCES tags (tag_id), FOREIGN KEY (username_id) REFERENCES usernames (username_id))'
                .format(tn=table_name))

            c.execute(
                'CREATE TABLE global_dict (word_id INTEGER PRIMARY KEY, word TEXT, type TEXT, global_occurrences INTEGER, UNIQUE (word, type))'   # UNIQUE (word, type)  # TODO set both
                .format(tn=table_name))

            c.execute(
                'CREATE TABLE occurrences (word_id INTEGER, comment_id INTEGER, occurrences INTEGER, FOREIGN KEY(word_id) REFERENCES global_dict(word_id), FOREIGN KEY(comment_id) REFERENCES {tn}(comment_id), UNIQUE (word_id, comment_id))'
                    .format(tn=table_name))

            c.execute("CREATE TABLE tags (tag_id INTEGER PRIMARY KEY, tag TEXT, UNIQUE (tag))")

            c.execute("CREATE TABLE usernames (username_id INTEGER PRIMARY KEY, username TEXT, UNIQUE (username))")

            c.execute(
                'CREATE TABLE saved_models_tfidf (model_id TEXT, occurrences_threshold INTEGER, num_topics INTEGER, normalize INTEGER)')

            c.execute(
                'CREATE TABLE saved_models_lsi (model_id TEXT, occurrences_threshold INTEGER, num_topics INTEGER, power_iters INTEGER, extra_samples INTEGER)')

            c.execute(
                'CREATE TABLE saved_models_rp (model_id TEXT, occurrences_threshold INTEGER, num_topics INTEGER)')

            c.execute(
                'CREATE TABLE saved_models_lda (model_id TEXT, occurrences_threshold INTEGER, num_topics INTEGER, distributed INTEGER, alpha TEXT, eta REAL)')

            c.execute(
                'CREATE TABLE saved_models_hdp (model_id TEXT, occurrences_threshold INTEGER, num_topics INTEGER, gamma INTEGER, kappa REAL, tau REAL, k REAL, t REAL, eta REAL)')

            conn.commit()
            conn.close()


    def add_single_csv(self, csv_file_path):

        table_name = 'reddit_comments'

        #processor = processing.Processing(**self.config_processing)
        processor = processing.Processing(delete_punctuation_marks=True, delete_numeral=True, delete_single_words=True, initial_form=True, stop_words=None)

        conn = sqlite3.connect(self.db_file)

        c = conn.cursor()

        c.execute("PRAGMA foreign_keys = ON")

        csvfile = open(csv_file_path)
        readCSV = csv.reader(csvfile, delimiter=',')

        # to_db = [(
        #     datetime.datetime.fromtimestamp(int(row[0])).strftime('%Y-%m-%d %H:%M:%S'),
        #     row[1].replace("'", "''"),
        #     row[2].replace("'", "''"),
        #     row[3].replace("'", "''")
        # ) for row in readCSV]
        # c.executemany("INSERT INTO " + table_name + " (time, username, comment, tag) VALUES (?, ?, ?, ?)", to_db)

        for row in readCSV:
            time_ = datetime.datetime.fromtimestamp(int(row[0])).strftime('%Y-%m-%d %H:%M:%S')
            username_ = row[1].replace("'", "''")
            comment_ = row[2].replace("'", "''")
            tag_ = row[3].replace("'", "''")  # assume there are 4 fields in every line

            comment_id = None
            tag_id = None
            username_id = None

            try:
                c.execute("INSERT INTO tags (tag) VALUES ('" + tag_ + "')")
                tag_id = c.lastrowid
            except sqlite3.IntegrityError as err:
                c.execute("SELECT tag_id FROM tags WHERE tag='" + tag_ + "'")
                found = [r for r in c]
                if len(found) > 0:
                    tag_id = found[0][0]
                else:
                    tag_id = None

            try:
                c.execute("INSERT INTO usernames (username) VALUES ('" + username_ + "')")
                username_id = c.lastrowid
            except sqlite3.IntegrityError as err:
                c.execute("SELECT username_id FROM usernames WHERE username='" + username_ + "'")
                found = [r for r in c]
                if len(found) > 0:
                    username_id = found[0][0]
                else:
                    username_id = None

            try:
                c.execute("INSERT INTO " + table_name + " (time, comment, username_id, tag_id) VALUES ('" + time_ +
                          "', '" + comment_ + "', '" + str(username_id) + "', '" + str(tag_id) + "')")
                comment_id = c.lastrowid
            except sqlite3.IntegrityError as err:
                print("Error adding comment issued at " + time_ + ": " + str(err))
                comment_id = None
            # to process text and insert result
            #document, words = processor(comment_)

            # print(words)
            # for w in words:
            #     # c.execute("IF EXISTS (SELECT * FROM global_dict WHERE word='" + w[0] + "' AND type='" + w[1] + "') " +
            #     #           "UPDATE global_dict SET global_occuerrences=global_occuerrences+" + str(w[2]) +
            #     #           " WHERE word='" + w[0] + "' AND type='" + w[1] + "' " +
            #     #           "ELSE INSERT INTO global_dict (word, type, global_occuerrences) VALUES ('" + w[0] + "', '" + w[1] + "', " + str(w[2]) + ")")
            #     # # added to global dictionatyor updated number of occurrences
            #     try:
            #         c.execute("INSERT INTO global_dict (word, type, global_occurrences) VALUES ('" + w[0] + "', '" +
            #                   w[1] + "', " + str(w[2]) + ")")
            #         # print("- inserted " + w[0])
            #
            #     except sqlite3.IntegrityError as err1:
            #         # UNIQUE constraint prevents from adding, trying updating
            #         try:
            #             c.execute("UPDATE global_dict SET global_occurrences=global_occurrences+" + str(w[2]) +
            #                       " WHERE word='" + w[0] + "' AND type='" + w[1] + "' ")
            #             # print("- updated " + w[0])
            #         except sqlite3.IntegrityError as err2:
            #             print("!! failed both to insert and update word.\n   - error message on INSERT: " + str(err1)
            #                   + "\n   - error message on UPDATE: " + str(err2))
            #     c.execute("SELECT * FROM global_dict WHERE word='" + w[0] + "' AND type='" + w[1] + "'")
            #     word_id = None
            #     try:
            #         word_id = c.fetchone()[0]
            #     except:
            #         print("!! failed to select the word " + w[0] + ", " + w[1] + " in 'global_dict' table")
            #     if word_id:
            #         try:
            #             c.execute(
            #                 "INSERT INTO occurrences (word_id, comment_id, occurrences) VALUES ('" + str(
            #                     word_id) + "', '" +
            #                 str(comment_id) + "', " + str(w[2]) + ")")
            #         except sqlite3.IntegrityError as err:
            #             print("!! failed to insert record into 'occurrences' table.\n   - error message: " + str(err))

        conn.commit()
        conn.close()


    def keys(self):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()

        if self.occ_threshold == 0:
            c.execute("SELECT word_id FROM global_dict")
        else:
            c.execute("SELECT word_id_index_" + str(self.occ_threshold)
                      + ".word_id_out FROM global_dict INNER JOIN word_id_index_" + str(self.occ_threshold)
                      + " ON global_dict.word_id = word_id_index_" + str(self.occ_threshold) + ".word_id")

        result = [row[0]-1 for row in c]
        conn.close()

        return result

    def __getitem__(self, item):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()

        if self.occ_threshold == 0:
            c.execute("SELECT word FROM global_dict WHERE word_id=" + str(item + 1))
        else:
            c.execute("SELECT global_dict.word FROM global_dict "
                      + "INNER JOIN word_id_index_" + str(self.occ_threshold)
                      + " ON global_dict.word_id = word_id_index_" + str(self.occ_threshold) + ".word_id "
                      + "WHERE word_id_index_" + str(self.occ_threshold) + ".word_id_out=" + str(item+1))

        result = [row[0] for row in c]
        conn.close()

        if result:
            return result
        else:
            raise IndexError("index not found in db")

    def __len__(self):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()

        if self.occ_threshold == 0:
            c.execute("SELECT count(*) FROM global_dict")
        else:
            c.execute("SELECT count(*) FROM word_id_index_" + str(self.occ_threshold))

        result = [row[0] for row in c]
        conn.close()

        if result:
            return result[0]
        else:
            return 0

    def init_word_id_index(self, occurrences_threshold):

        self.occ_threshold = occurrences_threshold

        if not os.path.isfile(self.db_file):
            return  # has to be called later

        if self.occ_threshold == 0:
            return  # no need in index

        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        # check table, create if not exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='word_id_index_" + str(self.occ_threshold) + "'")
        result = [row[0] for row in c]
        if len(result) == 0:

            print("Start building index for occurrence threshold " + str(occurrences_threshold))
            c.execute(
                'CREATE TABLE word_id_index_' + str(self.occ_threshold)
                + ' (word_id_out INTEGER PRIMARY KEY, word_id INTEGER, FOREIGN KEY(word_id) REFERENCES global_dict(word_id))')
            c.execute(
                "INSERT INTO word_id_index_" + str(self.occ_threshold)
                + " (word_id) SELECT global_dict.word_id FROM global_dict WHERE global_dict.global_occurrences >= "
                + str(occurrences_threshold))

            conn.commit()
            print(" - done")
        else:
            print("Found index table for occurrence threshold " + str(occurrences_threshold))

        conn.close()

    def save(self, fname_or_handle):
        print("'save' function is not implemented in SQLite-based Dictionary class")
        pass  # no need to save data, that are originally stored in db

    def find_model(self, model_type, occurrences_threshold, **kwargs):
        r = None
        params = ""
        for p in kwargs:
            if str(kwargs[p]) == "True":
                kwargs[p] = 1
            if str(kwargs[p]) == "False":
                kwargs[p] = 0
            params = params + p + "='" + str(kwargs[p]) + "' AND "
        if len(params) > 5:
            params = params[0:-5]

            conn = sqlite3.connect(self.db_file)
            c = conn.cursor()

            c.execute("SELECT model_id FROM saved_models_" + model_type.lower() + " WHERE occurrences_threshold='"
                      + str(occurrences_threshold) + "' AND " + params)
            result = [row[0] for row in c]

            conn.close()

            if len(result) > 0:
                r = result[0]

        else:
            print("No parameters found in config")

        return r

    def add_model(self, model_type, model_id, occurrences_threshold, **kwargs):
        params = ""
        values = ""
        for p in kwargs:
            if str(kwargs[p]) == "True":
                kwargs[p] = 1
            if str(kwargs[p]) == "False":
                kwargs[p] = 0
            params = params + p + ", "
            values = values + "'" + str(kwargs[p]) + "', "
        if len(params) > 1:
            params = params[0:-2]
            values = values[0:-2]

            conn = sqlite3.connect(self.db_file)
            c = conn.cursor()

            c.execute("INSERT INTO saved_models_" + model_type.lower() + " (model_id, occurrences_threshold, " + params
                      + ") VALUES ('" + model_id + "', '" + str(occurrences_threshold) + "', " + values + ")")

            conn.commit()
            conn.close()

    def create_result_table(self, length, model_id):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()

        c.execute("DROP TABLE IF EXISTS '" + model_id + "_results'")

        fields = ""
        for f in range(length):
            fields += "value" + str(f + 1) + ", "
        c.execute("CREATE TABLE '" + model_id + "_results' (comment_id INTEGER, "
                  + fields + "FOREIGN KEY(comment_id) REFERENCES reddit_comments(comment_id))")

        conn.commit()
        conn.close()

    def save_result(self, vector, length,  model_id, comment_id):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()

        values = dict(vector)
        fields = ""
        for f in range(length):
            try:
                fields = fields + ", " + str(values[f])
            except KeyError:
                fields = fields + ", null"

        c.execute("INSERT INTO '" + model_id + "_results' VALUES(" + str(comment_id) + fields + ")")

        conn.commit()
        conn.close()


    def create_usarname_tag_onehot_tables(self, model_id, usernames, tags):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()

        c.execute("DROP TABLE IF EXISTS {0}_input_tags".format(model_id))
        c.execute("CREATE TABLE {0}_input_tags (tag_id INTEGER, tag_onehot_index INTEGER, FOREIGN KEY (tag_id) REFERENCES reddit_comments (tag_id))".format(model_id))
        tag_ids = {}
        onehot_idx = -1
        for el in tags:  # add to table
            # c.execute("INSERT INTO input_tags (tag_id) VALUES (SELECT tag_id FROM tags WHERE tag = '" + el + "')")
            c.execute("SELECT tag_id FROM tags WHERE tag = '" + el + "'")
            found = [r for r in c]
            if len(found) > 0:
                onehot_idx += 1
                c.execute("INSERT INTO {0}_input_tags (tag_id, tag_onehot_index) VALUES (".format(model_id) + str(found[0][0]) + ", " + str(onehot_idx) + ")")
                tag_ids[found[0][0]] = onehot_idx

        c.execute("DROP TABLE IF EXISTS {0}_input_usernames".format(model_id))
        c.execute("CREATE TABLE {0}_input_usernames (username_id INTEGER, username_onehot_index INTEGER, FOREIGN KEY (username_id) REFERENCES reddit_comments (username_id))")
        username_ids = {}
        onehot_idx = -1
        for el in usernames:  # add to table
            # c.execute("INSERT INTO input_usernames (username_id) VALUES (SELECT username_id FROM usernames WHERE username = '" + el + "')")
            c.execute("SELECT username_id FROM usernames WHERE username = '" + el + "'")
            found = [r for r in c]
            if len(found) > 0:
                onehot_idx += 1
                c.execute("INSERT INTO {0}_input_usernames (username_id, username_onehot_index) VALUES (".format(model_id) + str(found[0][0]) + ", " + str(onehot_idx) + ")")
                username_ids[found[0][0]] = onehot_idx

        enc_tags = OneHotEncoder()
        enc_tags.fit([[x] for x in range(0, len(tag_ids))])  # TODO try without OneHotEncoder
        enc_usernames = OneHotEncoder()
        enc_usernames.fit([[x] for x in range(0, len(username_ids))])

        c.execute("DROP TABLE IF EXISTS {0}_output_onehot_tag".format(model_id))
        c.execute("SELECT tag_id FROM {0}_input_tags".format(model_id))
        found = [r for r in c]
        fields_tags_def = ""
        fields_tags_usg = ""
        for i in found:
            fields_tags_def = fields_tags_def + ", tag_" + str(tag_ids[i[0]]) + " REAL"
            fields_tags_usg = fields_tags_usg + ", tag_" + str(tag_ids[i[0]])
        c.execute("CREATE TABLE {0}_output_onehot_tag (comment_id INTEGER".format(model_id) + fields_tags_def +
                  ", FOREIGN KEY (comment_id) REFERENCES reddit_comments (comment_id))")


        c.execute("DROP TABLE IF EXISTS {0}_output_onehot_username".format(model_id))
        c.execute("SELECT username_id FROM input_usernames")
        found = [r for r in c]
        fields_usernames_def = ""
        fields_usernames_usg = ""
        for i in found:
            fields_usernames_def = fields_usernames_def + ", username_" + str(username_ids[i[0]]) + " REAL"
            fields_usernames_usg = fields_usernames_usg + ", username_" + str(username_ids[i[0]])
        c.execute("CREATE TABLE {0}_output_onehot_username (comment_id INTEGER".format(model_id) + fields_usernames_def +
                  ", FOREIGN KEY (comment_id) REFERENCES reddit_comments (comment_id))")


        # select docs using input* tables, compute onehots, insert into output_onehot*

        c.execute("""SELECT reddit_comments.comment_id, {0}_input_tags.tag_onehot_index, {0}_input_usernames.username_onehot_index
                         FROM (
                             (reddit_comments LEFT OUTER JOIN {0}_input_tags ON reddit_comments.tag_id = {0}_input_tags.tag_id)
                             LEFT OUTER JOIN {0}_input_usernames ON reddit_comments.username_id = {0}_input_usernames.username_id 
                         )
                         WHERE (reddit_comments.tag_id IN (SELECT tag_id FROM {0}_input_tags)) 
                         OR (reddit_comments.username_id IN (SELECT username_id FROM {0}_input_usernames))""".format(model_id))
        c2 = conn.cursor()
        for r in c:
            if r[1] is not None:
                vector_tags = enc_tags.transform([[r[1]]]).toarray()  # tag_onehot_index of comment
                c2.execute("INSERT INTO {0}_output_onehot_tag (comment_id".format(model_id) + fields_tags_usg + " )" +
                           " VALUES (" + str(r[0]) + ", " + (repr(vector_tags.tolist()))[2:-2] + ")")  # TODO test! This depends on behaviour od repr()

            if r[2] is not None:
                vector_usernames = enc_usernames.transform([[r[2]]]).toarray()  # username_onehot_index of comment
                c2.execute("INSERT INTO {0}_output_onehot_username (comment_id".format(model_id) + fields_usernames_usg + " )" +
                           " VALUES (" + str(r[0]) + ", " + (repr(vector_usernames.tolist()))[2:-2] + ")")  # TODO test! This depends on behaviour od repr()

        conn.commit()
        conn.close()


    # def create_general_onehot_tables(self):
    #     conn = sqlite3.connect(self.db_file)
    #     c = conn.cursor()
    #
    #
    #     # TODO compute onehots
    #
    #     conn.commit()
    #     conn.close()

    def find_coments(self, tag_id, from_date, to_date=None):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        addition_command = ''
        if to_date:
            addition_command = 'AND time < TIMESTAMP({0})'.format(to_date)
        c.execute(
                "SELECT comment_id, username_id FROM reddit_comments WHERE time >= TIMESTAMP({0}) AND tag_id={1} {2}"
                    .format(from_date, tag_id, addition_command))
        comments = [comment for comment in c]
        conn.close()
        return comments

    def get_count_comments_by_tag(self, tag_id, from_date, to_date=None):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        addition_command = ''
        if to_date:
            addition_command = ' AND time < {0)'.format(to_date.timestamp())
        request = 'SELECT count(*) FROM reddit_comments WHERE reddit_comments.time >= {0} AND username_id = {1}{2};'.format(from_date.timestamp(), tag_id, addition_command)
        c.execute(request)
        counts = [result for result in c]
        conn.close()
        return counts[0]

    def get_average_users_onehot_by_tag(self, model_id,tag_id, from_date, to_date=None):
        average_vector = None
        count_coument = 0
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        addition_command = ''
        if to_date:
            addition_command = ' AND time < {0)'.format(to_date.timestamp())
        request = """SELECT {3}_output_onehot_username.* 
                  FROM reddit_comments 
                  LEFT JOIN {3}_output_onehot_username ON reddit_comments.comment_id = {3}_output_onehot_username.comment_id 
                  WHERE reddit_comments.time >= {0} AND reddit_comments.tag_id = {1}{2};""".format(
            from_date.timestamp(), tag_id, addition_command, model_id)
        c.execute(request)
        for coment_onehot in c:
            if coment_onehot[0] is None:
                continue
            count_coument += 1
            curent_onehot = numpy.array(coment_onehot[1:])
            if average_vector is not None:
                average_vector += curent_onehot
            else:
                average_vector = curent_onehot
        average_vector /= count_coument
        conn.close()
        return average_vector

    def get_average_tag_onehot_by_user(self, model_id, user_id, from_date, to_date=None):
        average_vector = None
        count_coument = 0
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        addition_command = ''
        if to_date:
            addition_command = ' AND time < {0)'.format(to_date.timestamp())
        request = """SELECT {3}_output_onehot_tag.* 
                  FROM reddit_comments 
                  LEFT JOIN {3}_output_onehot_tag ON reddit_comments.comment_id = {3}_output_onehot_tag.comment_id 
                  WHERE reddit_comments.time >= {0} AND reddit_comments.username_id = {1}{2};""".format(
            from_date.timestamp(), user_id, addition_command, model_id)
        c.execute(request)
        for coment_onehot in c:
            if coment_onehot[0] is None:
                continue
            count_coument += 1
            curent_onehot = numpy.array(coment_onehot[1:])
            if average_vector is not None:
                average_vector += curent_onehot
            else:
                average_vector = curent_onehot
        average_vector /= count_coument
        conn.close()
        return average_vector

    def get_last_coment_by_tag(self, tag_id, to_date=None):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        addition_command = ''
        if to_date:
            addition_command = ' AND time < {0)'.format(to_date.timestamp())
        request = 'SELECT TOP(1) time FROM reddit_comments WHERE tag_id = {0} {1} ORDERED BY -time'.format(tag_id, addition_command)
        c.execute(request)
        result = [r for r in c]
        conn.close()
        return result[0]

    def get_last_coment_by_user(self, user_id, tag_id, to_date=None):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        addition_command = ''
        if to_date:
            addition_command = ' AND time < {0}'.format(to_date.timestamp())
        request = 'SELECT TOP(1) time FROM reddit_comments WHERE tag_id = {0} AND username_id = {2} {1} ORDERED BY -time'.format(tag_id,
                                                                                                          addition_command, user_id)
        c.execute(request)
        result = [r for r in c]
        conn.close()
        return result[0]

    def get_average_tag_of_user(self,  model_id, user_id, tag_id, to_date=None):
        count_coument = 0
        average_vector = None
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        addition_command = ''
        if to_date:
            addition_command = ' AND time < {0)'.format(to_date.timestamp())
        request = '''
                SELECT 
                    {1}_results.*
                FROM 
                    reddit_comments 
                        LEFT JOIN 
                    {1}_results ON reddit_comments.comment_id = {1}_results.comment_id 
                WHERE 
                    reddit_comments.user_id = {0} AND reddit_comments.tag_id = {3}{2}'''.format(
            user_id, model_id, addition_command, tag_id
        )
        for coment_onehot in c:
            if coment_onehot[0] is None:
                continue
            count_coument += 1
            curent_onehot = numpy.array(coment_onehot[1:])
            if average_vector is not None:
                average_vector += curent_onehot
            else:
                average_vector = curent_onehot
        average_vector /= count_coument
        conn.close()

    def get_average_topic(self, model_id, from_date, to_date=None):
        count_coument = 0
        average_vector = None
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        addition_command = ''
        if to_date:
            addition_command = ' AND time < {0)'.format(to_date.timestamp())
        request = '''
        SELECT 
            {1}_results.* 
        FROM 
            reddit_comments
                LEFT JOIN 
            {1}_results ON reddit_comments.comment_id = {1}_results.comment_id 
        WHERE 
            reddit_comments.time >= {0}{2}'''.format(
            from_date.timestamp(), model_id, addition_command
        )
        c.execute(request)
        for coment_onehot in c:
            if coment_onehot[0] is None:
                continue
            count_coument += 1
            curent_onehot = numpy.array(coment_onehot[1:])
            if average_vector is not None:
                average_vector += curent_onehot
            else:
                average_vector = curent_onehot
        average_vector /= count_coument
        conn.close()
        return average_vector

    def get_average_topic_by_tag(self, model_id, tag_id, from_date, to_date=None):
        count_coument = 0
        average_vector = None
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        addition_command = ''
        if to_date:
            addition_command = ' AND time < {0}'.format(to_date.timestamp())
        request = '''
        SELECT 
            {1}_results.*
        FROM 
            reddit_comments 
                LEFT JOIN 
            {1}_results ON reddit_comments.comment_id = {1}_results.comment_id 
        WHERE 
            reddit_comments.time >= {0} AND reddit_comments.tag_id = {3}{2}'''.format(
            from_date.timestamp(), model_id, addition_command, tag_id
        )
        c.execute(request)
        for coment_onehot in c:
            if coment_onehot[0] is None:
                continue
            count_coument += 1
            curent_onehot = numpy.array(coment_onehot[1:])
            if average_vector is not None:
                average_vector += curent_onehot
            else:
                average_vector = curent_onehot
        average_vector /= count_coument
        conn.close()
        return average_vector

    def get_average_topic_by_user(self, model_id, user_id, from_date, to_date=None):
        count_coument = 0
        average_vector = None
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        addition_command = ''
        if to_date:
            addition_command = ' AND time < {0)'.format(to_date.timestamp())
        request = 'SELECT {1}_results.* ' \
                  'FROM reddit_comments LEFT JOIN {1}_results ON reddit_comments.comment_id = {1}_results.comment_id ' \
                  'WHERE reddit_comments.time >= {0} AND reddit_comments.username_id = {3} {2};'.format(
            from_date.timestamp(), model_id, addition_command, user_id)
        c.execute(request)
        for coment_onehot in c:
            if coment_onehot[0] is None:
                continue
            count_coument += 1
            curent_onehot = numpy.array(coment_onehot[1:])
            if average_vector is not None:
                average_vector += curent_onehot
            else:
                average_vector = curent_onehot
        average_vector /= count_coument
        conn.close()
        return average_vector

    def get_average_users_onehot(self, model_id, from_date, to_date=None):
        average_vector = None
        count_coument = 0
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        addition_command = ''
        if to_date:
            addition_command = ' AND time < {0)'.format(to_date.timestamp())
        request = """SELECT {2}_output_onehot_username.* 
                  FROM reddit_comments 
                  LEFT JOIN {2}_output_onehot_username ON reddit_comments.comment_id = {2}_output_onehot_username.comment_id 
                  WHERE reddit_comments.time >= {0}{1};""".format(from_date.timestamp(), addition_command, model_id)
        c.execute(request)
        for coment_onehot in c:
            if coment_onehot[0] is None:
                continue
            count_coument += 1
            curent_onehot = numpy.array(coment_onehot[1:])
            if average_vector is not None:
                average_vector += curent_onehot
            else:
                average_vector = curent_onehot
        average_vector /= count_coument
        conn.close()
        return average_vector

    def get_average_tag_onehot(self, model_id, from_date, to_date=None):
        average_vector = None
        count_coument = 0
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        addition_command = ''
        if to_date:
            addition_command = ' AND time < {0)'.format(to_date.timestamp())
        request = """SELECT {2}_output_onehot_tag.* 
        FROM reddit_comments 
        LEFT JOIN {2}_output_onehot_tag ON reddit_comments.comment_id = {2}_output_onehot_tag.comment_id 
        WHERE reddit_comments.time >= {0}{1};""".format(
            from_date.timestamp(), addition_command, model_id)
        c.execute(request)
        for coment_onehot in c:
            if coment_onehot[0] is None:
                continue
            count_coument += 1
            curent_onehot = numpy.array(coment_onehot[1:])
            if average_vector is not None:
                average_vector += curent_onehot
            else:
                average_vector = curent_onehot
        average_vector /= count_coument
        conn.close()
        return average_vector

    def get_ordered_table_by_name(self, name, order):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        request = "SELECT * FROM {} ORDER BY {}".format(name, order)
        c.execute(request)
        result = [r for r in c]
        conn.close()
        return result



class Corpus_db:


    def __init__(self, parent):
        self._current = -1
        self.parent = parent


    def __getitem__(self, key):
        conn = sqlite3.connect(self.parent.db_file)
        c = conn.cursor()

        if self.parent.occ_threshold == 0:
            c.execute("SELECT word_id, occurrences FROM occurrences WHERE comment_id=" + str(key + 1))
        else:
            c.execute("SELECT word_id_index_" + str(self.parent.occ_threshold) + ".word_id_out, occurrences FROM occurrences "
                      + "INNER JOIN word_id_index_" + str(self.parent.occ_threshold) +
                      " ON occurrences.word_id = word_id_index_" + str(self.parent.occ_threshold) + ".word_id"
                      + " WHERE comment_id=" + str(key+1))
        corpus = []
        for row in c:
            corpus.append((row[0]-1, row[1]))

        conn.close()
        if corpus:
            return corpus
        else:
            raise IndexError("index not found in db")

    # def __next__(self):
    #     self._current += 1
    #     try:
    #         return self[self._current]
    #     except IndexError:
    #         # self._current = -1  # TODO test and check whether such implementation is valid
    #         raise StopIteration

    def __iter__(self):
        return Corpus_db_iter(self.parent, self)

        # self._current = -1  # TODO test and check whether such implementation is valid
        # return self

    def __len__(self):
        conn = sqlite3.connect(self.parent.db_file)
        c = conn.cursor()

        if self.parent.occ_threshold == 0:
            c.execute("SELECT count(*) FROM reddit_comments")
        else:
            c.execute("SELECT count(*) FROM ("
                      + "SELECT DISTINCT occurrences.comment_id "
                      + "FROM occurrences INNER JOIN word_id_index_" + str(self.parent.occ_threshold)
                      + " ON occurrences.word_id = word_id_index_" + str(self.parent.occ_threshold) + ".word_id)")
            # count of documents that contain at least one word listed in word_id_index

        result = [row[0] for row in c]
        conn.close()

        if result:
            return result[0]
        else:
            return 0


class Corpus_db_iter:

    def __init__(self, parent_dict, parent_corpus):
        self.parent = parent_dict
        self.corpus = parent_corpus
        self._current = -1

    def __iter__(self):
        return self

    def __next__(self):
        self._current += 1
        try:
            return self.corpus[self._current]
        except IndexError:
            # self._current = -1  # TODO test and check whether such implementation is valid
            raise StopIteration



