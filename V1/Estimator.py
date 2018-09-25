import numpy as np


class Estimator:
    def __init__(self, threshold, k=20):
        if isinstance(k, list):
            self.k = np.array(k)
        else:
            self.k = np.array([k])
        self.true_positive_count_arr = np.zeros(len(self.k))
        self.false_positive_count_arr = np.zeros(len(self.k))
        self.true_count_arr = np.zeros(len(self.k))
        self.reciprocal_rank_arr = np.zeros(len(self.k))

        self.true_positive_count = 0
        self.false_positive_count = 0
        self.true_count = 0
        self.reciprocal_rank = 0

        self.data_len = 0
        self.treshold = threshold

    def add_data(self, predictions, answers):
        # data = np.array(list(zip(predictions, answers)))

        nrow, ncol = np.shape(predictions)
        # for i in range(0, nrow):
        #     data = np.column_stack((predictions[i],answers[i]))
        #     self.true_positive_count += np.sum([(predictions[i]>self.treshold) & (answers[i] == 1)])
        #     self.false_positive_count += np.sum([(predictions[i]>self.treshold) & (answers[i] == 0)])
        #     self.true_count += np.sum([x == 1 for _, x in data])
        #     self.reciprocal_rank += 1. / abs(np.subtract(data.transpose()[0], data.transpose()[1])).mean()
        #     self.data_len += 1
        self.true_positive_count = np.sum(np.logical_and(predictions > self.treshold, answers == 1))
        self.false_positive_count = np.sum(np.logical_and(predictions > self.treshold, answers == 0))
        self.true_count = np.sum(answers)
        self.reciprocal_rank = 1./abs(np.subtract(predictions, answers)).mean()
        self.data_len += nrow
        new_answers = []

        for index, value in enumerate(self.k):
            # Select top i feature (sorted by prediction value)
            # for i in range(0, nrow):
            #     # data = np.array(list(zip(predictions, answers)))
            #     data = np.column_stack((predictions[i], answers[i]))
            #     data = data[data[:,0].argsort()][::-1][0:value]
            #     self.true_positive_count_arr[index] += np.sum([(y == 1) for x, y in data])
            #     self.false_positive_count_arr[index] += np.sum([(y == 0) for x, y in data])
            #     self.true_count_arr[index] += np.sum([x == 1 for _, x in data])
            #     self.reciprocal_rank_arr[index] += 1. / abs(np.subtract(data.transpose()[0], data.transpose()[1])).mean()

            # alternative find n (n=2) max in each string
            new_predictions = np.array([])
            new_answers = np.array([])
            indexes = (np.argsort(predictions)[:,::-1][:,:value])
            for i in range(0,nrow):
                new_predictions = np.append(new_predictions, predictions[i][indexes[i]])
                new_answers = np.append(new_answers, answers[i][indexes[i]])
            self.true_positive_count_arr[index] += np.sum(np.logical_and(new_predictions > self.treshold, new_answers == 1))
            self.false_positive_count_arr[index] += np.sum(np.logical_and(new_predictions > self.treshold, new_answers == 0))
            self.true_count_arr[index] += np.sum(new_answers)
            self.reciprocal_rank_arr[index] += 1. / abs(np.subtract(new_predictions, new_answers)).mean()

    def estimate(self):
        # Calculate precision for all k
        precision_k = self.true_positive_count_arr/(self.false_positive_count_arr + self.true_positive_count_arr)
        # Calculate recall for all k
        recall_k = self.true_positive_count_arr/self.true_count_arr

        mrr_k = self.reciprocal_rank / self.data_len

        precision = self.true_positive_count / (self.false_positive_count + self.true_positive_count)
        recall = self.true_positive_count / self.true_count
        mrr = self.reciprocal_rank / self.data_len
        return precision, recall, mrr, precision_k, recall_k, mrr_k


if __name__ == '__main__':
    pred = np.array([[0.8, 0.9, 0.6, 0.39, 0.8],
                      [0.97, 0.48, 0.5, 0.8, 0.7],
                      [0.6, 0.7, 0.4, 0.7, 0.6],
                      [0.5, 0.86, 0.3, 0.6, 0.5],
                      [0.4, 0.6, 0.2, 0.5, 1.1]])
    ans =  np.asarray([[0, 0, 0, 1, 1],
                       [1, 1, 0, 1, 0],
                       [0, 0, 1, 0, 1],
                       [1, 1, 0, 1, 1],
                       [0, 0, 0, 0, 1]])

    estimator = Estimator(0.85, [2, 4])
    estimator.add_data(pred, ans)
    print(estimator.estimate())