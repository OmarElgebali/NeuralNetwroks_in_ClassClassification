import matplotlib.pyplot as plt

class Evaluation:
    def __init__(self, y_predict, y_actual, algorithm):
        self.y_actual = y_actual
        self.y_predict = y_predict
        self.algorithm = algorithm

    def confusion_matrix(self):
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        for actual, predicted in zip(self.y_actual, self.y_predict):
            if actual == 1 and predicted == 1:
                true_positives += 1
            elif actual == 0 and predicted == 1:
                false_positives += 1
            elif actual == 0 and predicted == 0:
                true_negatives += 1
            elif actual == 1 and predicted == 0:
                false_negatives += 1
        confusion_matrix = [
            [true_positives, false_positives],
            [false_negatives, true_negatives]
        ]
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        plt.title(f"Confusion Matrix of {self.algorithm} Algorithm")
        plt.ylabel("True labels")
        plt.xlabel("Predicted labels")
        plt.xticks([0, 1], ["Predicted 0", "Predicted 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.show()
        # for row in confusion_matrix:
        #     print(row)


# dataset = read_csv('Datasets/Dry_Bean_Dataset_ADALINE.csv')
# x_1 = dataset['MajorAxisLength'].to_numpy().reshape(-1, 1)
# x_2 = dataset['MinorAxisLength'].to_numpy().reshape(-1, 1)
# y = dataset['Class'].tolist()
# y_bin = [1 if label == 'CALI' else 0 for label in y]
#
# minMax = [MinMaxScaler() for _ in range(2)]
# x_1_scaled = minMax[0].fit_transform(x_1)
# x_2_scaled = minMax[1].fit_transform(x_2)
# adaline = Adaline(x_1_scaled, x_2_scaled, y_bin, 1, 1000, .1, .01)
# adaline.fit()
#
# feature1 = [295.4698306,
#             274.8633573,
#             313.570417,
#             301.392791,
#             357.1890036,
#             330.155474,
#             307.9956112,
#             336.9594211,
#             354.7254337,
#             342.74242472,
#             255.0735621,
#             244.102719,
#             233.8049677,
#             240.9695005,
#             260.0898266,
#             257.4648986,
#             265.4699791,
#             252.7723761,
#             257.4891433,
#             240.3770452
#             ]
#
# feature2 = [
#     196.3118225,
#     211.8851392,
#     197.8987002,
#     208.3464437,
#     179.8346914,
#     198.4615253,
#     213.6247263,
#     199.0051127,
#     193.354325,
#     203.259513,
#     157.80274,
#     168.5224911,
#     179.500919,
#     175.6450402,
#     163.1027196,
#     165.036857,
#     161.0284951,
#     169.4350399,
#     166.5592919,
#     178.6807736
# ]
# labelTest = [
#     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
# ]
#
# feature1 = np.array(feature1).reshape(-1, 1)
# feature2 = np.array(feature2).reshape(-1, 1)
# feature1_scaled = minMax[0].transform(feature1)
# feature2_scaled = minMax[1].transform(feature2)
# sample1 = [240.3770452]
# sample1 = np.array(sample1).reshape(-1, 1)
#
# sample2 = [178.6807736]
# sample2 = np.array(sample2).reshape(-1, 1)
#
# f1 = minMax[0].transform(sample1)
# f2 = minMax[1].transform(sample2)
# predict = adaline.predict([f1, f2])
# print("predict", predict)
#
# adaline.test(feature1_scaled, feature2_scaled, labelTest)
#
# eval = Evaluation(feature1_scaled, feature2_scaled, y_bin)
# eval.confusion_matrix()
