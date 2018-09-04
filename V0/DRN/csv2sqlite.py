
import csv
import datetime
import os.path
import sqlite3
from os import listdir

from config import config_processing
from processing import Processing


# old debug & test code

def redditcsv2sqlite(csv_file_path, db_file_path, table_name='reddit_comments'):  # function copied to modeling.py as add_csv_data(...) method, with few additions

    if not os.path.isfile(db_file_path):
        directory = '/'.join(db_file_path.split('/')[0:-1])
        if not os.path.exists(directory):
            os.makedirs(directory)

        conn = sqlite3.connect(db_file_path)
        c = conn.cursor()

        c.execute("PRAGMA foreign_keys = ON")

        c.execute('''
        CREATE TABLE {tn} (
            comment_id INTEGER PRIMARY KEY,
            time TIMESTAMP(14),
            username TEXT,
            comment TEXT,
            tag TEXT
        )'''.format(tn=table_name))

        c.execute('''
        CREATE TABLE global_dict (
            word_id INTEGER PRIMARY KEY,
            word TEXT,
            type TEXT,
            global_occurrences INTEGER,
            UNIQUE (word , type))''')

        c.execute('''
        CREATE TABLE occurrences (
            word_id INTEGER,
            comment_id INTEGER,
            occurrences INTEGER,
            FOREIGN KEY (word_id)
                REFERENCES global_dict (word_id),
            FOREIGN KEY (comment_id)
                REFERENCES {tn} (comment_id)
        )'''.format(tn=table_name))

        conn.commit()
        conn.close()

    processor = Processing(**config_processing)

    conn = sqlite3.connect(db_file_path)
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
        comment_id = None
        tag_ = row[3].replace("'", "''")  # assume there are 4 fields in every line
        try:
            c.execute("INSERT INTO " + table_name + " (time, username, comment, tag) VALUES ('" + time_ +
                      "', '" + username_ + "', '" + comment_ + "', '" + tag_ + "')")
            comment_id = c.lastrowid
        except sqlite3.IntegrityError as err:
            print("Error adding comment issued at " + time_ + ": " + str(err))
            comment_id = None
        # to process text and insert result
        _, words = processor(comment_)  # TODO move function to Model and there pass _ to appropriate method
        # print(words)
        for w in words:
            # c.execute("IF EXISTS (SELECT * FROM global_dict WHERE word='" + w[0] + "' AND type='" + w[1] + "') " +
            #           "UPDATE global_dict SET global_occuerrences=global_occuerrences+" + str(w[2]) +
            #           " WHERE word='" + w[0] + "' AND type='" + w[1] + "' " +
            #           "ELSE INSERT INTO global_dict (word, type, global_occuerrences) VALUES ('" + w[0] + "', '" + w[1] + "', " + str(w[2]) + ")")
            # # added to global dictionatyor updated number of occurrences
            try:
                c.execute("INSERT INTO global_dict (word, type, global_occurrences) VALUES ('" + w[0] + "', '" +
                          w[1] + "', " + str(w[2]) + ")")

            except sqlite3.IntegrityError as err1:
                # UNIQUE constraint prevents from adding, trying updating
                try:
                    c.execute("UPDATE global_dict SET global_occurrences=global_occurrences+" + str(w[2]) +
                              " WHERE word='" + w[0] + "' AND type='" + w[1] + "' ")
                except sqlite3.IntegrityError as err2:
                    print("!! failed both to insert and update word.\n   - error message on INSERT: " + str(err1)
                          + "\n   - error message on UPDATE: " + str(err2))
            c.execute("SELECT * FROM global_dict WHERE word='" + w[0] + "' AND type='" + w[1] + "'")
            word_id = None
            try:
                word_id = c.fetchone()[0]
            except:
                pass
            try:
                c.execute("INSERT INTO occurrences (word_id, comment_id, occurrences) VALUES ('" + str(word_id) + "', '" +
                          str(comment_id) + "', " + str(w[2]) + ")")
            except sqlite3.IntegrityError as err:
                print("!! failed to insert record into 'occurrences' table.\n   - error message: " + str(err))

    conn.commit()
    conn.close()
    pass



def redditcsv2sqlite_batch(csv_folder_path, db_file_path):
    filenames = listdir(csv_folder_path)
    for f in filenames:
        if f.endswith(".csv"):  # assume all .csv files in the folder have appropriate format
            redditcsv2sqlite(os.path.join(csv_folder_path, f), db_file_path)



def view_sqlite(db_file_path, table_name='reddit_comments'):  # prints ALL the table to console!
    conn = sqlite3.connect(db_file_path)
    c = conn.cursor()

    c.execute('SELECT * FROM {tn}'.format(tn=table_name))
    for row in c:
        print(str(row[0]) + " | " + str(row[1]) + " | " + str(row[2]) + " | " + str(row[3]))

    conn.commit()
    conn.close()



def view_user_messages(db_file_path, username, table_name='reddit_comments'):
    conn = sqlite3.connect(db_file_path)
    c = conn.cursor()

    c.execute("SELECT * FROM " + table_name + " WHERE username='" + username + "'")
    for row in c:
        print(str(row[0]) + " | " + str(row[1]) + " | " + str(row[2]) + " | " + str(row[3]))

    conn.commit()
    conn.close()


def count_user_messages(db_file_path, table_name='reddit_comments'):
    conn = sqlite3.connect(db_file_path)
    c = conn.cursor()

    c.execute("DROP TABLE IF EXISTS users")
    # c.execute("CREATE TABLE users (username TEXT, number INT)")
    c.execute("CREATE TABLE users AS SELECT username, count(*) FROM " + table_name + " GROUP BY username")

    # c.execute("SELECT username, count(*) FROM " + table_name + " GROUP BY username")
    c.execute("SELECT * FROM users")

    for row in c:
        print(str(row[0]) + ": " + str(row[1]))

    conn.commit()
    conn.close()


if __name__=="__main__":

    # create
    db_file_path = "data/db/reddit.sqlite"

    csv_file_path = 'data/csv/reddit_01_17_posts.csv'
    # csv_file_path = 'data/csv/test.csv'

    redditcsv2sqlite(csv_file_path, db_file_path)

    # redditcsv2sqlite_batch("data/csv", db_file_path)


    # view
    view_sqlite(db_file_path)

    # # view for user
    # # db_file_path = "data/db/reddit_all.sqlite"
    # print("posts of Tosacc_com")
    # view_user_messages(db_file_path, "Tosacc_com")

    # count for each user
    # db_file_path = "data/db/reddit_all.sqlite"
    print("\nnumber of posts for each user")
    count_user_messages(db_file_path)
