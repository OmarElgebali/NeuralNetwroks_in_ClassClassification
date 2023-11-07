from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import Adaline
import Perceptron
import Preprocessing
import evaluation

encoder_model = LabelEncoder()
scaler_f1_model = MinMaxScaler()
scaler_f2_model = MinMaxScaler()

encoded_class_train = []
norm_feature_1_train = []
norm_feature_2_train = []
encoded_class_test = []
norm_feature_1_test = []
norm_feature_2_test = []

label_test_prediction = []
label_train_prediction = []
adaline_model = None
model_weights = []


label_encode = {
    1: ['BOMBAY', 'CALI'],
    2: ['BOMBAY', 'SIRA'],
    3: ['CALI', 'SIRA']
}
f1_name = ''
f2_name = ''


def prepare(algorithm, feature_1_name, feature_2_name, classes_encode_number):
    global encoder_model, scaler_f1_model, scaler_f2_model, encoded_class_train, norm_feature_1_train, norm_feature_2_train, encoded_class_test, norm_feature_1_test, norm_feature_2_test, f1_name, f2_name
    dataset = Preprocessing.readFile(classes_encode_number)
    f1_name = feature_1_name
    f2_name = feature_2_name
    feature, target_class = Preprocessing.get_feature(feature_1_name, feature_2_name, dataset)
    feature_1_train, feature_2_train, feature_1_test, feature_2_test, class_train, class_test = Preprocessing.train_test(feature, target_class, feature_1_name, feature_2_name)
    encoder_model, scaler_f1_model, scaler_f2_model, encoded_class_train, norm_feature_1_train, norm_feature_2_train = Preprocessing.preprocessing_training(algorithm, feature_1_train, feature_2_train, class_train)
    encoded_class_test, norm_feature_1_test, norm_feature_2_test = Preprocessing.preprocessing_test(algorithm, feature_1_test, feature_2_test, class_test, scaler_f1_model, scaler_f2_model, encoder_model)


def fit(algorithm, epochs, eta, mse, bias):
    global label_test_prediction, label_train_prediction, adaline_model, model_weights
    if algorithm == 'Perceptron':
        model_weights = Perceptron.perceptron_train(norm_feature_1_train, norm_feature_2_train, encoded_class_train, eta, epochs)
        label_test_prediction = Perceptron.perceptron_test(norm_feature_1_test, norm_feature_2_test, encoded_class_test, model_weights)
    elif algorithm == 'Adaline':
        adaline_model = Adaline.Adaline(norm_feature_1_train, norm_feature_2_train, encoded_class_train, bias, epochs, eta, mse)
        adaline_model.fit()
        label_test_prediction = adaline_model.test(norm_feature_1_test, norm_feature_2_test)
        # label_test_prediction = adaline_model.test_with_eval(norm_feature_1_test, norm_feature_2_test,encoded_class_test)
        model_weights = adaline_model.weights
    plot_draw(algorithm)


def predict(algorithm, x1, x2, labels_encode_number):
    label1, label2 = label_encode[labels_encode_number]
    if algorithm == 'Perceptron':
        predicted_value = Perceptron.perceptron_predict(x1, x2, model_weights)
        predicted_label = Perceptron.revert_to_string(predicted_value, label1, label2)
    elif algorithm == 'Adaline':
        predicted_label = adaline_model.predict([x1, x2])
    return predicted_label


def plot_draw(algorithm):
    evaluation_model = evaluation.Evaluation(y_predict=label_test_prediction, y_actual=encoded_class_test, algorithm=algorithm, f1_name=f1_name, f2_name=f2_name)
    evaluation_model.plot_decision_boundary(feature1=norm_feature_1_train, feature2=norm_feature_2_train, weights=model_weights, labels=encoded_class_train)
    evaluation_model.confusion_matrix()
