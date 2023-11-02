# import tkinter as tk
# from tkinter import W
# from tkinter import *
#
#
# main = tk.Tk()
# main.title('Perceptorn & Adaline')
# boxs = tk.Listbox(main)
# boxs.pack()
# boxs.insert(0, 'Area')
# boxs.insert(1, 'Perimeter')
# boxs.insert(2, 'MajorAxisLength')
# boxs.insert(3, 'MinorAxisLength')
# boxs.insert(4, 'roundnes')
# radio = tk.IntVar()
# tk.Radiobutton(main, text="C1 & C2", variable=radio, value=1).pack()
# tk.Radiobutton(main, text="C1 & C2", variable=radio, value=2).pack()
# tk.Radiobutton(main, text="C3 & C2", variable=radio, value=3).pack()
#
# tk.Label(main, text='Learning Rate').pack()
# LearningRate = tk.Entry(main)
# LearningRate.pack()
#
# tk.Label(main, text='Epochs').pack()
# Epochs = tk.Entry(main)
# Epochs.pack()
#
#
# button = tk.Button(main, text="Perceptron", width=10, height=3, command=lambda: pr.big(box1, box2, rad, learn, ep))
# button.pack()
#
# main.mainloop()
#
# selected_indices = boxs.curselection()
# box1 = boxs.get(selected_indices[0])
# box2 = boxs.get(selected_indices[1])
# rad = int(radio.get())
# learn = float(LearningRate.get())
# ep = int(Epochs.get())
