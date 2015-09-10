__author__ = 'shen'

import numpy as np
import operator


class knn():
    def data_set(self):
        data = np.array([[1, 1, 1], [1, 1.2, 1.5], [0, 0.5, 0], [0, 0.2, 0.5]])
        labels = [1, 1, -1, -1]
        return data, labels

    def knn_classifier(self, test_data, train_data, labels, K):
        m, n = train_data.shape
        difference = np.tile(test_data, (m, 1)) - train_data
        difference = difference ** 2
        distance = difference.sum(axis=1)
        distance = distance ** 0.5
        sort_diff = distance.argsort()

        k_nearest_labels = {}
        for i in range(K):
            label = labels[sort_diff[i]]
            k_nearest_labels[label] = k_nearest_labels.get(label, 0)+1
        sort_label = sorted(k_nearest_labels.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sort_label[0]

    def nearest_center(self, test_data, train_data, labels, threshhold):
        m, n = train_data.shape
        aver = np.zeros([1, n])
        number = 0
        for i in range(m):
            if labels[i] == 1:
                number += 1
                aver += train_data[i]
        aver = aver * 1.0 / (number + 1e-6)
        difference = np.array(aver - test_data)
        difference = difference ** 2
        distance = difference.sum(axis=1)
        distance = distance ** 0.5
        if distance > threshhold:
            return -1
        else:
            return 1

k = knn() #create KNN object
data, labels = k.data_set()
cls = k.knn_classifier([0, 0.8, 0.2], data, labels, 3)
print cls, cls[0]

cls = k.nearest_center([0, 0.8, 0.2], data, labels, 0.2)
print cls












