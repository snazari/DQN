
import datetime

from modeling import Model

if __name__ == '__main__':

    # create untrained Model object based on config.py
    from config import config
    model = Model(config=config)

    # or load previously created (using the same DB) model with its own configuration
    # model = Model(model_file="/home/user/repos/python_deep_reinforcement_sequence/data/LDA/model/1d439b93-fbc9-4ae4-b76d-7d1332259344_model.pickle")

    # create main dictionary from .csv: this complete replaces data in DB.
    # Call add_csv_data ONLY ONCE before the first launch of training!!
    # model.add_csv_data('data/csv/test_seasons.csv')  # this drops DB, but not deletes saved model files, built using it

    # or create it from all .csv files in a given folder, assuming all of them have the proper format
    model.add_csv_data('data/csv/test.csv')  # this drops db, but not deletes saved model files, built using it
    # a, b = model.get_part_of_data(parts=10)
    model.compute_context_features([datetime.timedelta(days=2), datetime.timedelta(hours=1)])
    print(1)
    # train model
    # If the description of a model with appropriate config parameters is found in db, the training phase will be skipped
    # Otherwise, the new model will be trained and saved for further usage, model_id and config parameters will be added to DB
    # Also this function creates word_id index according to occurrences_threshold parameter
    model.train_model()


    # get vector for the given text
    text = "jun jul aug sep" # "This is the random text the following words are added in order to be recognized by dictionary: read, blog, post"
    vectors = model.transforming(text)
    print(vectors)


    # # view vectors for the entire corpus, old way
    # vectors = model.transforming(model._dictionary.corpus)
    # print("\nall results:")
    # print(vectors)

    # save to DB vectors for the entire corpus
    model.get_results()



    # # mem usage
    # import os
    # import psutil
    # process = psutil.Process(os.getpid())
    # print(process.memory_info().rss)
    pass



