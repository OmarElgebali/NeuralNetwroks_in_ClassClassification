import tkinter as tk
from tkinter import W
from tkinter import *
from Preprocessing import *

selected_indices = []
box1 = ''
box2 = ''
rad = 0
learn = 0
ep = 0


def getValue():
    global selected_indices, box1, box2, rad, learn, ep
    selected_indices = boxs.curselection()
    box1 = str(boxs.get(selected_indices[0]))
    print(box1)
    box2 = str(boxs.get(selected_indices[1]))
    print(box2)
    rad = int(radio.get())
    print(rad)
    learn = float(LearningRate.get())
    print(learn)
    ep = int(Epochs.get())
    print(ep)
    execute(rad, box1, box2, learn, ep)


main = tk.Tk()
main.title('Perceptorn & Adaline')
boxs = tk.Listbox(main, selectmode=tk.MULTIPLE)
boxs.pack()
boxs.insert(0, 'Area')
boxs.insert(1, 'Perimeter')
boxs.insert(2, 'MajorAxisLength')
boxs.insert(3, 'MinorAxisLength')
boxs.insert(4, 'roundnes')
radio = tk.IntVar()
tk.Radiobutton(main, text="C1 & C2", variable=radio, value=1).pack()
tk.Radiobutton(main, text="C1 & C3", variable=radio, value=2).pack()
tk.Radiobutton(main, text="C3 & C3", variable=radio, value=3).pack()

tk.Label(main, text='Learning Rate').pack()
LearningRate = tk.Entry(main)
LearningRate.pack()

tk.Label(main, text='Epochs').pack()
Epochs = tk.Entry(main)
Epochs.pack()

button = tk.Button(main, text="Perceptron", width=10, height=3, command=lambda: getValue())
button.pack()

main.mainloop()
