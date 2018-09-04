

from modeling import Model
from cross_validation import CrossValidationDQNN

if __name__ == '__main__':
    from config import config

    model = Model(config=config)
    model.add_csv_data('data/csv/test.csv')
    model.train_model()
    model.init_onehot()
    cross_validation = CrossValidationDQNN(config=config)
    estimate = cross_validation.run()
