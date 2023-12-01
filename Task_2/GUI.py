import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import pandas as pd
from tqdm import tqdm

import Core
import Preprocessing

activation_functions = {1: 'Sigmoid', 2: 'Hyper-Tangent'}
features_names = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes"]

main = tk.Tk()
main.title('Multi-Layer Perceptron')

Task_1_frame = tk.Frame(main)
Task_1_frame.columnconfigure(0, weight=1)
Task_1_frame.columnconfigure(1, weight=1)
Task_1_frame.columnconfigure(2, weight=1)
Task_1_frame.columnconfigure(3, weight=1)
Task_1_frame.columnconfigure(4, weight=1)
Task_1_frame.columnconfigure(5, weight=1)
progress = ttk.Progressbar(Task_1_frame, orient=tk.HORIZONTAL, mode='determinate')
progress_label = tk.Label(Task_1_frame, text="Progress: 0.00%")
time_label = tk.Label(Task_1_frame, text="Time: 0.00s")

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


def start_fitting(activation_function, epochs, eta, bias, num_layers, num_neurons_in_each_layer):
    Core.preprocessing(activation_function, bias)

    def update_progress(epoch, total_epochs):
        progress_value = (epoch * 100) / total_epochs
        progress['value'] = progress_value
        progress_label.config(text=f"Progress: {progress_value:.2f}%")
        Task_1_frame.update_idletasks()

    def fit_and_track_progress():
        start_time = time.time()
        Core.fit(epochs, eta, bias, num_layers, num_neurons_in_each_layer, update_progress)
        end_time = time.time()
        duration = end_time - start_time
        time_label.config(text=f"Time: {duration:.2f}s")

    training_thread = threading.Thread(target=fit_and_track_progress)
    training_thread.start()


# def start_fitting(activation_function, epochs, eta, bias, num_layers, num_neurons_in_each_layer):
#     """
#     :param activation_function: 'Sigmoid', 'Hyper-Tangent'
#     :param epochs: integer
#     :param eta: float
#     :param bias: 0, 1
#     :param num_layers: integer
#     :param num_neurons_in_each_layer: [#neurons]
#     """
#     Core.preprocessing(activation_function, bias)
#     Core.fit(epochs, eta, bias, num_layers, num_neurons_in_each_layer)
#     for epoch in tqdm(range(epochs), desc='Training Progress'):
#         time.sleep(0.1)  # Simulate training time
#         # Update progress bar
#         progress['value'] = (epoch + 1) * (100 / epochs)
#         Task_1_frame.update_idletasks()
#     print("Training completed!")
# pass

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


txt_predict = []
lbl_predict = []
lbl_predict_output_label = tk.Label()
lbl_predict_output_value = tk.Label()


def start_predicting():
    each_feature_value = []
    for index, feature_value in enumerate(txt_predict):
        each_feature_value.append(float(feature_value.get()))
    print(pd.DataFrame([each_feature_value], columns=features_names))
    predict_output = Core.classify(pd.DataFrame([each_feature_value], columns=features_names))
    print("predict_output: ", predict_output)
    predict_result = Preprocessing.inverse_target_encoder([predict_output])[0]
    print("predict_result: ", predict_result)
    lbl_predict_output_value.config(text=f"{predict_result}")


def predict_window():
    global lbl_predict_output_label, lbl_predict_output_value
    for entry, lbl in zip(txt_predict, lbl_predict):
        entry.destroy()
        lbl.destroy()
    lbl_predict.clear()
    txt_predict.clear()

    popup = tk.Toplevel()
    popup.title("Prediction Window")
    prediction_frame = tk.Frame(popup)
    prediction_frame.columnconfigure(0, weight=1)
    prediction_frame.columnconfigure(1, weight=1)
    for index, curr_row in enumerate(features_names):
        lbl_predict.append(tk.Label(prediction_frame, text=curr_row, font=('Arial', 16)))
        lbl_predict[-1].grid(row=index, column=0, sticky=tk.W + tk.E)
        txt_predict.append(tk.Entry(prediction_frame))
        txt_predict[-1].grid(row=index, column=1, sticky=tk.W + tk.E)
    lbl_predict_output_label = tk.Label(prediction_frame, text="Class", font=('Arial', 16))
    lbl_predict_output_label.grid(row=5, column=0, sticky=tk.W + tk.E)
    lbl_predict_output_value = tk.Label(prediction_frame, text="", font=('Helvetica', 20), fg="green")
    lbl_predict_output_value.grid(row=5, column=1, sticky=tk.W + tk.E)
    btn_predict = tk.Button(prediction_frame, text="Predict", font=('Arial', 12), command=start_predicting)
    btn_predict.grid(row=6, column=0, sticky=tk.W + tk.E)
    prediction_frame.pack(fill='x')


btn_fit = tk.Button(Task_1_frame, text="Fit", font=('Arial', 12), command=check_fitting)
btn_open_predict = tk.Button(Task_1_frame, text="Open Predict Window", font=('Arial', 12), command=predict_window)


def create_neuron_entries():
    for entry, lbl in zip(num_neurons_txt, num_neurons_lbl):
        entry.destroy()
        lbl.destroy()
    num_neurons_lbl.clear()
    num_neurons_txt.clear()

    for layer in range(int(float(num_layers_txt.get()))):
        row_counter = start_row_counter + layer + 1
        num_neurons_lbl.append(
            tk.Label(Task_1_frame, text=f"# of Neurons in Layer ({layer + 1})", font=('Times New Roman', 16)))
        num_neurons_lbl[layer].grid(row=row_counter, column=0, columnspan=3, sticky=tk.W + tk.E)
        num_neurons_txt.append(tk.Entry(Task_1_frame))
        num_neurons_txt[layer].grid(row=row_counter, column=3, columnspan=3, sticky=tk.W + tk.E)

    btn_fit.grid(row=start_row_counter + len(num_neurons_lbl) + 1, column=0, columnspan=3, sticky=tk.W + tk.E)
    btn_open_predict.grid(row=start_row_counter + len(num_neurons_lbl) + 1, column=3, columnspan=3, sticky=tk.W + tk.E)
    progress.grid(row=start_row_counter + len(num_neurons_lbl) + 2, column=0, columnspan=4, sticky=tk.W + tk.E)
    progress_label.grid(row=start_row_counter + len(num_neurons_lbl) + 2, column=4, columnspan=1, sticky=tk.W + tk.E)
    time_label.grid(row=start_row_counter + len(num_neurons_lbl) + 2, column=5, columnspan=1, sticky=tk.W + tk.E)


create_neuron_btn = tk.Button(Task_1_frame, text="Create Neuron Layers", command=create_neuron_entries)
create_neuron_btn.grid(row=5, column=0, columnspan=6)

Task_1_frame.pack(fill='x')

main.mainloop()
