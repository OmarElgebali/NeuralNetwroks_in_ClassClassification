# import Core
#
# algorithm_names = ['Perceptron', 'Adaline']
# features_names = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes"]
# classes = [1, 2, 3]
# classes_encoding = {
#     1: '(BOMBAY) & (CALI)',
#     2: '(BOMBAY) & (SIRA)',
#     3: '(CALI) & (SIRA)'
# }
#
# epochs = 1000
# eta = 0.1
# mse = 0.01
# bias = 1
#
# for algorithm in algorithm_names:
#     for classes_encode_number in classes:
#         for feature_1_name in features_names:
#             for feature_2_name in features_names:
#                 if feature_2_name == feature_1_name:
#                     continue
#                 Core.prepare(algorithm, feature_1_name, feature_2_name, classes_encode_number)
#                 Core.fit(algorithm, epochs, eta, mse, bias)
#
#
# # algorithm = algorithm_names[1]
# # classes_encode_number = 1
# # feature_2_name = features_names[3]
# # feature_1_name = features_names[2]
# # Core.prepare(algorithm, feature_1_name, feature_2_name, classes_encode_number)
# # Core.fit(algorithm, epochs, eta, mse, bias)
