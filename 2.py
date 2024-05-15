import numpy as np
import tkinter as tk

number_neurons_1 = 2
number_neurons_2 = 5
number_outputs = 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(output, y_true):
    return np.mean((output - y_true) ** 2)

class NeuralNetwork:
    def __init__(self):
        self.weights_1 = np.random.normal(size=(number_neurons_1, number_neurons_2))
        self.bias_1 = np.random.normal(size=(number_neurons_2,))
        self.weights_2 = np.random.normal(size=(number_neurons_2, number_outputs))
        self.bias_2 = np.random.normal(size=(number_outputs,))

    def activation(self, x, weights, bias):
        z = np.dot(x, weights) + bias
        return sigmoid(z), z

    def feedforward(self, x):
        a1, z1 = self.activation(x, self.weights_1, self.bias_1)
        a2, z2 = self.activation(a1, self.weights_2, self.bias_2)        
        return a2

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 10000

        for epoch in range(epochs):
            for i in range(len(data)):
                x = np.array([data[i]])
                y_true = np.array([all_y_trues[i]])

                a1, z1 = self.activation(x, self.weights_1, self.bias_1)
                a2, z2 = self.activation(a1, self.weights_2, self.bias_2)

                loss = mse_loss(a2, y_true)
                output_delta = (y_true - a2) * deriv_sigmoid(z2)
                neurons_2_delta = np.dot(output_delta, self.weights_2.T) * deriv_sigmoid(z1)

                self.weights_2 += learn_rate * np.dot(a1.T, output_delta)
                self.bias_2 += learn_rate * np.sum(output_delta)
                self.weights_1 += learn_rate * np.dot(x.T, neurons_2_delta)
                self.bias_1 += learn_rate * np.sum(neurons_2_delta)

network = NeuralNetwork()

data = [[1, 0], [0, 0], [1, 1], [0, 1]]
all_y_trues = [0, 0, 1, 0]

network.train(data, all_y_trues)

app = tk.Tk()
app.title("Простой интерфейс нейронной сети")

def predict_output():
    input_data = [0.0, 0.0]
    try:
        input_data = [float(entry1.get()), float(entry2.get())]
    except ValueError:
        pass

    x = np.array(input_data)
    output = network.feedforward(x)
    label_result.config(text="Результат: " + str(output))

label_instruction = tk.Label(app, text="Введите значения для нейронной сети:")
label_instruction.pack()

entry1 = tk.Entry(app)
entry1.pack()

entry2 = tk.Entry(app)
entry2.pack()

button_predict = tk.Button(app, text="Предсказать", command=predict_output)
button_predict.pack()

label_result = tk.Label(app, text="Результат: ")
label_result.pack()

app.mainloop()