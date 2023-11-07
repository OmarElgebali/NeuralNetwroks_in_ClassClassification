import tkinter as tk
from tkinter import messagebox
import Core

algorithm_names = ['Perceptron', 'Adaline']
features_names = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes"]
classes_encoding = {
    1: '(BOMBAY) & (CALI)',
    2: '(BOMBAY) & (SIRA)',
    3: '(CALI) & (SIRA)'
}
check_if_fitted = {
    'algorithm': '',
    'feature_1_name': '',
    'feature_2_name': '',
    'classes_encode_number': 0
}


# check_if_fitted = {
#     alg: {
#         feature1: {feature2: 0 for feature2 in features_names if feature2 != feature1} for feature1 in features_names
#     }
#     for alg in algorithm_names
# }


def start_fitting(algorithm, epochs, eta, mse, bias, classes_encode_number, selected_features):
    feature_1_name, feature_2_name = selected_features
    Core.prepare(algorithm, feature_1_name, feature_2_name, classes_encode_number)
    Core.fit(algorithm, epochs, eta, mse, bias)


main = tk.Tk()
main.title('Perceptron & Adaline')

Task_1_frame = tk.Frame(main)
Task_1_frame.columnconfigure(0, weight=1)
Task_1_frame.columnconfigure(1, weight=1)
Task_1_frame.columnconfigure(2, weight=1)
Task_1_frame.columnconfigure(3, weight=1)
Task_1_frame.columnconfigure(4, weight=1)
Task_1_frame.columnconfigure(5, weight=1)

algorithm_lbl = tk.Label(Task_1_frame, text="Algorithm", font=('Times New Roman', 16))
algorithm_lbl.grid(row=0, column=0, columnspan=3, sticky=tk.W + tk.E)
radio_algorithm = tk.IntVar()
tk.Radiobutton(Task_1_frame, font=('Arial', 12), text="Perceptron", variable=radio_algorithm, value=1).grid(row=0,
                                                                                                            column=3,
                                                                                                            sticky=tk.W + tk.E)
tk.Radiobutton(Task_1_frame, font=('Arial', 12), text="Adaline", variable=radio_algorithm, value=2).grid(row=0,
                                                                                                         column=5,
                                                                                                         sticky=tk.W + tk.E)

epochs_lbl = tk.Label(Task_1_frame, text="Epochs", font=('Times New Roman', 16))
epochs_lbl.grid(row=1, column=0, columnspan=3, sticky=tk.W + tk.E)
epochs_txt = tk.Entry(Task_1_frame)
epochs_txt.grid(row=1, column=3, columnspan=3, sticky=tk.W + tk.E)

eta_lbl = tk.Label(Task_1_frame, text="Learning Rate", font=('Times New Roman', 16))
eta_lbl.grid(row=2, column=0, columnspan=3, sticky=tk.W + tk.E)
eta_txt = tk.Entry(Task_1_frame)
eta_txt.grid(row=2, column=3, columnspan=3, sticky=tk.W + tk.E)

mse_lbl = tk.Label(Task_1_frame, text="MSE Threshold (For Adaline Only)", font=('Times New Roman', 16))
mse_lbl.grid(row=3, column=0, columnspan=3, sticky=tk.W + tk.E)
mse_txt = tk.Entry(Task_1_frame)
mse_txt.grid(row=3, column=3, columnspan=3, sticky=tk.W + tk.E)

is_bias = tk.IntVar()
bias_lbl = tk.Label(Task_1_frame, text="Bias (For Adaline Only)", font=('Times New Roman', 16))
bias_lbl.grid(row=4, column=0, columnspan=3, sticky=tk.W + tk.E)
bias_chk = tk.Checkbutton(Task_1_frame, text="Add Bias", font=('Arial', 12), variable=is_bias)
bias_chk.grid(row=4, column=3, columnspan=3, sticky=tk.W + tk.E)

classes_lbl = tk.Label(Task_1_frame, text="Classes", font=('Times New Roman', 16))
classes_lbl.grid(row=5, column=0, columnspan=3, sticky=tk.W + tk.E)
radio_classes = tk.IntVar()
tk.Radiobutton(Task_1_frame, font=('Arial', 10), text="(BOMBAY) & (CALI)", variable=radio_classes, value=1).grid(row=5,
                                                                                                                 column=3,
                                                                                                                 sticky=tk.W + tk.E)
tk.Radiobutton(Task_1_frame, font=('Arial', 10), text="(BOMBAY) & (SIRA)", variable=radio_classes, value=2).grid(row=5,
                                                                                                                 column=4,
                                                                                                                 sticky=tk.W + tk.E)
tk.Radiobutton(Task_1_frame, font=('Arial', 10), text="(CALI) & (SIRA)", variable=radio_classes, value=3).grid(row=5,
                                                                                                               column=5,
                                                                                                               sticky=tk.W + tk.E)

features_lbl = tk.Label(Task_1_frame, text="Features", font=('Times New Roman', 16))
features_lbl.grid(row=6, column=0, columnspan=3, sticky=tk.W + tk.E)
features_listbox = tk.Listbox(Task_1_frame, selectmode=tk.MULTIPLE, font=('Arial', 10))
features_listbox.config(height=5)
features_listbox.grid(row=6, column=3, columnspan=3, sticky=tk.W + tk.E)
for feature_name in features_names:
    features_listbox.insert(tk.END, feature_name)

lbl_pred_x1 = tk.Label(Task_1_frame, text="X1", font=('Times New Roman', 16))
lbl_pred_x1.grid(row=7, column=0, columnspan=2, sticky=tk.W + tk.E)
txt_pred_x1 = tk.Entry(Task_1_frame)
txt_pred_x1.grid(row=7, column=2, sticky=tk.W + tk.E)

lbl_pred_x2 = tk.Label(Task_1_frame, text="X2", font=('Times New Roman', 16))
lbl_pred_x2.grid(row=7, column=3, sticky=tk.W + tk.E)
txt_pred_x2 = tk.Entry(Task_1_frame)
txt_pred_x2.grid(row=7, column=4, sticky=tk.W + tk.E)

lbl_pred_y = tk.Label(Task_1_frame, text="Output Prediction", font=('Times New Roman', 16))
lbl_pred_y.grid(row=9, column=0, sticky=tk.W + tk.E)
txt_pred_y = tk.Label(Task_1_frame, text="", font=('Helvetica', 20), fg="green")
txt_pred_y.grid(row=9, column=3, sticky=tk.W + tk.E)


def check_fitting():
    if not epochs_txt.get() or not eta_txt.get() or not mse_txt.get():
        messagebox.showerror(title="Error",
                             message="Ensure that these fields not empty\n1. Epochs\n2. Learning Rate\n3. Mse Threshold")
        return

    algorithm_index = radio_algorithm.get()
    if algorithm_index not in [1, 2]:
        messagebox.showerror(title="Error", message="Select an Algorithm")
        return

    epochs = int(float(epochs_txt.get()))
    if epochs < 0:
        messagebox.showerror(title="Error", message="(Epochs) must be +ve number")
        return

    eta = float(eta_txt.get())
    if eta <= 0:
        messagebox.showerror(title="Error", message="(Learning Rate) must be +ve number")
        return

    mse = float(mse_txt.get())
    if mse < 0:
        messagebox.showerror(title="Error", message="(MSE Threshold) must be +ve number")
        return

    classes_encode_number = radio_classes.get()
    if classes_encode_number not in [1, 2, 3]:
        messagebox.showerror(title="Error", message="Select 1 of The 3 Combinations of Classes")
        return

    selected_features_indices = features_listbox.curselection()
    if len(features_listbox.curselection()) != 2:
        messagebox.showerror(title="Error", message="Select only 2 features")
        return

    bias = is_bias.get()

    algorithms = {1: 'Perceptron', 2: 'Adaline'}
    algorithm = algorithms[algorithm_index]
    selected_features = [features_listbox.get(index) for index in selected_features_indices]
    start_fitting(algorithm, epochs, eta, mse, bias, classes_encode_number, selected_features)
    feature_1_name, feature_2_name = selected_features
    check_if_fitted['algorithm'] = algorithm
    check_if_fitted['feature_1_name'] = feature_1_name
    check_if_fitted['feature_2_name'] = feature_2_name
    check_if_fitted['classes_encode_number'] = classes_encode_number
    lbl_pred_x1.config(text=f"X1 ({feature_1_name})")
    lbl_pred_x2.config(text=f"X1 ({feature_2_name})")


def start_predicting():
    algorithm_index = radio_algorithm.get()
    if algorithm_index not in [1, 2]:
        messagebox.showerror(title="Error", message="Select an Algorithm")
        return

    classes_encode_number = radio_classes.get()
    if classes_encode_number not in [1, 2, 3]:
        messagebox.showerror(title="Error", message="Select 1 of The 3 Combinations of Classes")
        return

    selected_features_indices = features_listbox.curselection()
    if len(features_listbox.curselection()) != 2:
        messagebox.showerror(title="Error", message="Select only 2 features")
        return

    algorithms = {1: 'Perceptron', 2: 'Adaline'}
    algorithm = algorithms[algorithm_index]
    selected_features = [features_listbox.get(index) for index in selected_features_indices]
    feature_1_name, feature_2_name = selected_features
    if check_if_fitted['algorithm'] != algorithm or check_if_fitted['classes_encode_number'] != classes_encode_number or check_if_fitted['feature_1_name'] != feature_1_name or check_if_fitted['feature_2_name'] != feature_2_name:
        messagebox.showerror(title="Error",
                             message=f"This Model is not fitted yet\nAlgorithm: {algorithm}\nClasses: {classes_encoding[classes_encode_number]}\nFeature 1: {feature_1_name}\nFeature 2: {feature_2_name}")
        return

    if not txt_pred_x1.get() or not txt_pred_x2.get():
        messagebox.showerror(title="Error", message="Ensure that X1 & X2 are not empty")
        return
    feature_1_value = float(txt_pred_x1.get())
    feature_2_value = float(txt_pred_x2.get())
    pred_output = Core.predict(algorithm=algorithm, x1=feature_1_value, x2=feature_2_value, labels_encode_number=classes_encode_number)
    txt_pred_y.config(text=f"{pred_output}")


def plot_evaluation():
    algorithm_index = radio_algorithm.get()
    if algorithm_index not in [1, 2]:
        messagebox.showerror(title="Error", message="Select an Algorithm")
        return

    classes_encode_number = radio_classes.get()
    if classes_encode_number not in [1, 2, 3]:
        messagebox.showerror(title="Error", message="Select 1 of The 3 Combinations of Classes")
        return

    selected_features_indices = features_listbox.curselection()
    if len(features_listbox.curselection()) != 2:
        messagebox.showerror(title="Error", message="Select only 2 features")
        return

    algorithms = {1: 'Perceptron', 2: 'Adaline'}
    algorithm = algorithms[algorithm_index]
    selected_features = [features_listbox.get(index) for index in selected_features_indices]
    feature_1_name, feature_2_name = selected_features
    if check_if_fitted['algorithm'] != algorithm or check_if_fitted['classes_encode_number'] != classes_encode_number or check_if_fitted['feature_1_name'] != feature_1_name or check_if_fitted['feature_2_name'] != feature_2_name:
        messagebox.showerror(title="Error",
                             message=f"This Model is not fitted yet\nAlgorithm: {algorithm}\nClasses: {classes_encoding[classes_encode_number]}\nFeature 1: {feature_1_name}\nFeature 2: {feature_2_name}")
        return
    Core.plot_draw(algorithm)


btn_fit = tk.Button(Task_1_frame, text="Fit", font=('Arial', 12), command=check_fitting)
btn_fit.grid(row=8, column=0, columnspan=2, sticky=tk.W + tk.E)

btn_plot_eval = tk.Button(Task_1_frame, text="Plot Evaluation", font=('Arial', 12), command=plot_evaluation)
btn_plot_eval.grid(row=8, column=2, columnspan=2, sticky=tk.W + tk.E)

btn_predict = tk.Button(Task_1_frame, text="Predict", font=('Arial', 12), command=start_predicting)
btn_predict.grid(row=8, column=4, columnspan=2, sticky=tk.W + tk.E)

Task_1_frame.pack(fill='x')

main.mainloop()
