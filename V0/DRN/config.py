
import datetime

config_processing = {  #  after changing this, the db must be rebuilt by calling .add_csv_data('path/to/file.csv')
    'delete_punctuation_marks': True,
    'delete_numeral': True,
    'delete_single_words': True,
    'initial_form': True, #False,
    'stop_words': ['for', 'a', 'of', 'the', 'and', 'to', 'in', 'from',
                   'commit', 'initial', 'create', 'request', 'merge', 'pull', 'travisyml', 'add', 'adding'],
}

config_models = \
    {
        'TfIdf': {
            'normalize': True,
            'num_topics': 5
        },
        'LSI': {
            'num_topics': 5,
            'power_iters': 2,
            'extra_samples': 100
        },
        'RP': {
            'num_topics': 5
        },
        'LDA': {
            'num_topics': 5,
            'distributed': False,
            'alpha': 'symmetric',
            'eta': None
        },
        'HDP': {
            'gamma': 1,
            'kappa': 1.0,
            'tau': 64.0,
            'K': 15,
            'T': 150,
            'eta': 0.01,
            'num_topics': 5
        }
    }

config_general = {
    'occurrences_threshold': 0,
    'model_type': 'LDA' #"TfIdf" # "LSI" # "LDA"  # 'HDP' # "RP"

}

config_onehot = {
    "usernames": ["summer", "autumn", "winter", "newseason"],
    "tags": ["tag1", "tag2"]
}

config_dqnn = {
    'steps_time': [datetime.timedelta(hours=1), datetime.timedelta(hours=2),
                   datetime.timedelta(days=1), datetime.timedelta(weeks=1),
                   datetime.timedelta(days=30)],
    "categories": ['shoe', "skam", "node"],
    'period': {
        'days': 2,
        'hours': 3,
        'minutes': 1,
    }
}


# cross-validation configuration
config_cv = {
    'percent': 0,
    'parts_count': 5
}

config = {
    'name': 'conf1',
    'db_filename': 'data/db/reddit.sqlite',
    'batch_size': 3,
    'delete_table': True,
    'silent_mode': False,
    'config_general': config_general,
    'config_processing': config_processing,
    'config_model': config_models[config_general['model_type']],
    'config_dqnn': config_dqnn,
    'config_cv': config_cv
}