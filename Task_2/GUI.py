import tkinter as tk
from tkinter import messagebox

import Core

activation_functions = {1: 'Sigmoid', 2: 'Hyper-Tangent'}


def start_fitting(activation_function, epochs, eta, bias, num_layers, num_neurons_in_each_layer):
    """
    :param activation_function: 'Sigmoid', 'Hyper-Tangent'
    :param epochs: integer
    :param eta: float
    :param bias: 0, 1
    :param num_layers: integer
    :param num_neurons_in_each_layer: [#neurons]
    """
    Core.preprocessing(activation_function, bias)
    Core.fit(activation_function, epochs, eta, bias, num_layers, num_neurons_in_each_layer)
    # pass


main = tk.Tk()
main.title('Multi-Layer Perceptron')

Task_1_frame = tk.Frame(main)
Task_1_frame.columnconfigure(0, weight=1)
Task_1_frame.columnconfigure(1, weight=1)
Task_1_frame.columnconfigure(2, weight=1)
Task_1_frame.columnconfigure(3, weight=1)
Task_1_frame.columnconfigure(4, weight=1)
Task_1_frame.columnconfigure(5, weight=1)

activation_lbl = tk.Label(Task_1_frame, text="Activation Function", font=('Times New Roman', 16))
activation_lbl.grid(row=0, column=0, columnspan=3, sticky=tk.W + tk.E)
radio_activation = tk.IntVar()
tk.Radiobutton(Task_1_frame, font=('Arial', 12), text="Sigmoid", variable=radio_activation, value=1).grid(row=0,
                                                                                                          column=3,
                                                                                                          sticky=tk.W + tk.E)
tk.Radiobutton(Task_1_frame, font=('Arial', 12), text="Hyper-Tangent", variable=radio_activation, value=2).grid(row=0,
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

is_bias = tk.IntVar()
bias_lbl = tk.Label(Task_1_frame, text="Bias", font=('Times New Roman', 16))
bias_lbl.grid(row=3, column=0, columnspan=3, sticky=tk.W + tk.E)
bias_chk = tk.Checkbutton(Task_1_frame, text="Add Bias", font=('Arial', 12), variable=is_bias)
bias_chk.grid(row=3, column=3, columnspan=3, sticky=tk.W + tk.E)

num_layers_lbl = tk.Label(Task_1_frame, text="Number of Layers", font=('Times New Roman', 16))
num_layers_lbl.grid(row=4, column=0, columnspan=3, sticky=tk.W + tk.E)
num_layers_txt = tk.Entry(Task_1_frame)
num_layers_txt.grid(row=4, column=3, columnspan=3, sticky=tk.W + tk.E)

start_row_counter = 5
num_neurons_lbl = []
num_neurons_txt = []


def check_fitting():
    if not epochs_txt.get() or not eta_txt.get() or not num_layers_txt.get():
        messagebox.showerror(title="Error",
                             message="Ensure that these fields not empty\n1. Epochs\n2. Learning Rate\n3. Num of Layers")
        return

    for neuron in num_neurons_txt:
        if not neuron.get():
            messagebox.showerror(title="Error",
                                 message="Ensure that each layer have at least 1 neuron")
            return

    activation_index = radio_activation.get()
    if activation_index not in [1, 2]:
        messagebox.showerror(title="Error", message="Select an Activation Function")
        return

    epochs = int(float(epochs_txt.get()))
    if epochs < 0:
        messagebox.showerror(title="Error", message="(Epochs) must be +ve number")
        return

    eta = float(eta_txt.get())
    if eta <= 0:
        messagebox.showerror(title="Error", message="(Learning Rate) must be +ve number")
        return
    #
    num_layers = int(float(num_layers_txt.get()))
    if num_layers < 0:
        messagebox.showerror(title="Error", message="(Number of Layers) must be +ve number")
        return

    num_neurons_in_each_layer = []
    for layer_index, neuron in enumerate(num_neurons_txt):
        num_neurons_in_each_layer.append(int(float(neuron.get())))
        if num_neurons_in_each_layer[-1] < 0:
            messagebox.showerror(title="Error",
                                 message=f"Number of Neurons in layer #{layer_index + 1} must be +ve number")
            return

    bias = is_bias.get()
    activation_function = activation_functions[activation_index]
    start_fitting(activation_function, epochs, eta, bias, num_layers, num_neurons_in_each_layer)


btn_fit = tk.Button(Task_1_frame, text="Fit", font=('Arial', 12), command=check_fitting)
btn_predict = tk.Button(Task_1_frame, text="Predict", font=('Arial', 12))


def create_neuron_entries():
    for entry, lbl in zip(num_neurons_txt, num_neurons_lbl):
        entry.destroy()
        lbl.destroy()
    num_neurons_lbl.clear()
    num_neurons_txt.clear()

    for layer in range(int(float(num_layers_txt.get()))):
        row_counter = start_row_counter + layer + 1
        num_neurons_lbl.append(
            tk.Label(Task_1_frame, text=f"Number of Neurons in Layer #{layer + 1}", font=('Times New Roman', 16)))
        num_neurons_lbl[layer].grid(row=row_counter, column=0, columnspan=3, sticky=tk.W + tk.E)
        num_neurons_txt.append(tk.Entry(Task_1_frame))
        num_neurons_txt[layer].grid(row=row_counter, column=3, columnspan=3, sticky=tk.W + tk.E)

    btn_fit.grid(row=start_row_counter + len(num_neurons_lbl) + 1, column=0, columnspan=3, sticky=tk.W + tk.E)
    btn_predict.grid(row=start_row_counter + len(num_neurons_lbl) + 1, column=3, columnspan=3, sticky=tk.W + tk.E)


create_neuron_btn = tk.Button(Task_1_frame, text="Create Neuron Entries", command=create_neuron_entries)
create_neuron_btn.grid(row=5, column=0, columnspan=6)


# lbl_pred_x1 = tk.Label(Task_1_frame, text="X1", font=('Times New Roman', 16))
# lbl_pred_x1.grid(row=7, column=0, columnspan=2, sticky=tk.W + tk.E)
# txt_pred_x1 = tk.Entry(Task_1_frame)
# txt_pred_x1.grid(row=7, column=2, sticky=tk.W + tk.E)
#
# lbl_pred_x2 = tk.Label(Task_1_frame, text="X2", font=('Times New Roman', 16))
# lbl_pred_x2.grid(row=7, column=3, sticky=tk.W + tk.E)
# txt_pred_x2 = tk.Entry(Task_1_frame)
# txt_pred_x2.grid(row=7, column=4, sticky=tk.W + tk.E)

# lbl_pred_y = tk.Label(Task_1_frame, text="Output Prediction", font=('Times New Roman', 16))
# lbl_pred_y.grid(row=9, column=0, sticky=tk.W + tk.E)
# txt_pred_y = tk.Label(Task_1_frame, text="", font=('Helvetica', 20), fg="green")
# txt_pred_y.grid(row=9, column=3, sticky=tk.W + tk.E)

def start_predicting():
    activation_index = radio_activation.get()
    if activation_index not in [1, 2]:
        messagebox.showerror(title="Error", message="Select an Activation Function")
        return

    # classes_encode_number = radio_classes.get()
    # if classes_encode_number not in [1, 2, 3]:
    #     messagebox.showerror(title="Error", message="Select 1 of The 3 Combinations of Classes")
    #     return
    #
    # selected_features_indices = features_listbox.curselection()
    # if len(features_listbox.curselection()) != 2:
    #     messagebox.showerror(title="Error", message="Select only 2 features")
    #     return
    #
    # algorithms = {1: 'Perceptron', 2: 'Adaline'}
    # algorithm = algorithms[algorithm_index]
    # selected_features = [features_listbox.get(index) for index in selected_features_indices]
    # feature_1_name, feature_2_name = selected_features
    # if not txt_pred_x1.get() or not txt_pred_x2.get():
    #     messagebox.showerror(title="Error", message="Ensure that X1 & X2 are not empty")
    #     return
    # feature_1_value = float(txt_pred_x1.get())
    # feature_2_value = float(txt_pred_x2.get())
    # pred_output = Core.predict(algorithm=algorithm, x1=feature_1_value, x2=feature_2_value, labels_encode_number=classes_encode_number)
    # txt_pred_y.config(text=f"{pred_output}")


Task_1_frame.pack(fill='x')

main.mainloop()
