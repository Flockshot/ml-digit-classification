import threading
import time

import torch
import torch.nn as nn
import numpy as np
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(time.gmtime(time.time()))
# we load all the datasets of Part 3
x_train, y_train = pickle.load(open("data/mnist_train.data", "rb"))
x_validation, y_validation = pickle.load(open("data/mnist_validation.data", "rb"))
x_test, y_test = pickle.load(open("data/mnist_test.data", "rb"))

x_train = x_train / 255.0
x_train = x_train.astype(np.float32)

x_test = x_test / 255.0
x_test = x_test.astype(np.float32)

x_validation = x_validation / 255.0
x_validation = x_validation.astype(np.float32)

# and converting them into Pytorch tensors in order to be able to work with Pytorch
x_train = torch.from_numpy(x_train).to(device)
y_train = torch.from_numpy(y_train).to(torch.long).to(device)

x_validation = torch.from_numpy(x_validation).to(device)
y_validation = torch.from_numpy(y_validation).to(torch.long).to(device)

x_test = torch.from_numpy(x_test).to(device)
y_test = torch.from_numpy(y_test).to(torch.long).to(device)


class MLPModel(nn.Module):
    def __init__(self, hidden_layers, learning_rate, activation_function, neurons_per_layer):
        super(MLPModel, self).__init__()

        # hidden_layer_neurons = 784 // 2
        hidden_layer_neurons = neurons_per_layer
        self.input_layer = nn.Linear(784, hidden_layer_neurons).to(device)
        self.hidden_layers = []

        for i in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_layer_neurons, hidden_layer_neurons).to(device))
            # self.hidden_layers.append(nn.Linear(hidden_layer_neurons, hidden_layer_neurons // 2).to(device))
            # hidden_layer_neurons = hidden_layer_neurons // 2

        self.output_layer = nn.Linear(hidden_layer_neurons, 10).to(device)
        self.activation_function = activation_function.to(device)
        self.learning_rate = learning_rate
        self.loss_function = nn.CrossEntropyLoss().to(device)
        self.softmax_function = torch.nn.Softmax(dim=1).to(device)

        self.patience_counter = 0
        self.max_patience = 10
        self.best_validation_accuracy = float('inf')

    def calc_accuracy(self, predictions, labels):
        soft_predictions = self.softmax_function(predictions).to(device)
        return ((torch.argmax(soft_predictions, 1) == labels).float().mean()) * 100

    def check_early_stopping(self, validation_accuracy):
        if validation_accuracy > self.best_validation_accuracy:
            self.patience_counter += 1
            if self.patience_counter > self.max_patience:
                return True
        else:
            self.patience_counter = 0
            self.best_validation_accuracy = validation_accuracy

    def forward(self, data):
        hidden_layer_output = self.activation_function(self.input_layer(data))
        for single_hidden_layer in self.hidden_layers:
            hidden_layer_output = self.activation_function(single_hidden_layer(hidden_layer_output))
        output_layer = self.output_layer(hidden_layer_output)

        return output_layer

    def run(self, train_data, train_labels, validation_data, validation_labels):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        ITERATION = 15000
        for iteration in range(1, ITERATION + 1):
            optimizer.zero_grad()
            train_predictions = self(train_data)
            train_loss_value = self.loss_function(train_predictions, train_labels)
            train_loss_value.backward()
            optimizer.step()

            with torch.no_grad():

                train_accuracy = self.calc_accuracy(train_predictions, train_labels)

                validation_predictions = self(validation_data)
                validation_loss = self.loss_function(validation_predictions, validation_labels)
                validation_accuracy = self.calc_accuracy(validation_predictions, validation_labels)

                if iteration % 25 == 0:
                    print(
                        "Iteration : %d - Train Loss %.4f - Train Accuracy : %.2f - Validation Loss : %.4f Validation Accuracy : %.2f" %
                        (iteration, train_loss_value.item(), train_accuracy, validation_loss.item(),
                         validation_accuracy))

                if self.check_early_stopping(validation_loss):
                    print("Early Stopping at iteration %d" % iteration)
                    break

    def test(self, test_data, test_labels):
        with torch.no_grad():
            test_predictions = self(test_data)
            test_accuracy = self.calc_accuracy(test_predictions, test_labels)
            print("Test - Accuracy %.2f" % test_accuracy)


threadLock = threading.RLock()


class myThread(threading.Thread):
    def __init__(self, threadName, hidden_layer, learning_rate, activation_function, neurons_per_layer,
                 train_data=x_train, train_labels=y_train, validation_data=x_validation,
                 validation_labels=y_validation):
        threading.Thread.__init__(self, name=threadName)
        self.hidden_layer = hidden_layer
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.neurons_per_layer = neurons_per_layer
        self.train_data = train_data
        self.train_labels = train_labels
        self.validation_labels = validation_labels
        self.validation_data = validation_data

    def run(self):
        print("Starting " + self.name + "\n")

        one_start_time = time.time()
        model_accuracies = []
        for i in range(10):
            print("Hidden Layer: %d - Learning Rate: %f - Activation Function: %s - Neurons: %d Iteration: %d" % (
                self.hidden_layer, self.learning_rate, self.activation_function.__class__.__name__,
                self.neurons_per_layer, i + 1))
            nn_model = MLPModel(self.hidden_layer, self.learning_rate, self.activation_function,
                                self.neurons_per_layer).to(device)
            nn_model.run(self.train_data, self.train_labels, self.validation_data, self.validation_labels)
            nn_model.test(x_test, y_test)
            print("")
            model_accuracies.append(nn_model.calc_accuracy(nn_model(x_test), y_test).cpu())

        mean = np.mean(model_accuracies)
        std = np.std(model_accuracies)
        lower_bound = calc_lower_bound(mean, std)
        upper_bound = calc_upper_bound(mean, std)
        stats_list = [self.hidden_layer, self.learning_rate, self.neurons_per_layer,
                      self.activation_function.__class__.__name__, mean, std,
                      lower_bound, upper_bound]

        # Get lock to synchronize threads
        threadLock.acquire()
        stats.append(stats_list)
        threadLock.release()
        print("Statistics for Hidden Layer: %d - Learning Rate: %f - Neuron: %d - Activation Function: %s" %
              (self.hidden_layer, self.learning_rate, self.neurons_per_layer,
               self.activation_function.__class__.__name__))
        print("Mean: %.2f" % mean)
        print("Std: %.2f" % std)
        print("Lower Bound: %.2f" % lower_bound)
        print("Upper Bound: %.2f" % upper_bound)
        print("Time Taken: %.2f seconds" % (time.time() - one_start_time))
        print("------------------------------------------------------------")


def calc_lower_bound(mean, std):
    return mean - ((1.96 * std) / np.sqrt(10))


def calc_upper_bound(mean, std):
    return mean + ((1.96 * std) / np.sqrt(10))


hidden_layers = [4, 6]
learning_rates = [0.001, 0.0001]
neurons = [128, 256]
activation_functions = [nn.Sigmoid(), nn.Tanh()]

stats = []

start_time = time.time()
j = 0
threads = []
for hidden_layer in hidden_layers:
    for learning_rate in learning_rates:
        for neuron in neurons:
            for activation_function in activation_functions:
                pass
                thread = myThread("Thread-%d" % j, hidden_layer, learning_rate, activation_function, neuron)
                threads.append(thread)
                thread.start()
                j += 1

for thread in threads:
    thread.join()

best_model_index = np.argmax(stats, axis=0)[4]
print("Best Model: Hidden Layer: %d - Learning Rate: %f - Neurons: %d - Activation Function: %s" %
      (stats[best_model_index][0], stats[best_model_index][1], stats[best_model_index][2], stats[best_model_index][3]))
print("Mean: %.2f" % stats[best_model_index][4])
print("Std: %.2f" % stats[best_model_index][5])
print("Lower Bound: %.2f" % stats[best_model_index][6])
print("Upper Bound: %.2f" % stats[best_model_index][7])

elapsed_time = time.time() - start_time
print("Total Time Taken: %.2f seconds" % elapsed_time)

# Best Model: Hidden Layer: 4 - Learning Rate: 0.001000 - Neurons: 256 - Activation Function: Tanh

best_train = torch.cat((x_train, x_validation), 0).to(device)
best_labels = torch.cat((y_train, y_validation), 0).to(device)


def get_func_from_name(func_name):
    if func_name == "Sigmoid":
        return nn.Sigmoid()
    elif func_name == "Tanh":
        return nn.Tanh()


best_model_thread = myThread("Thread-%d" % j, stats[best_model_index][0], stats[best_model_index][1],
                             get_func_from_name(stats[best_model_index][3]), stats[best_model_index][2], best_train,
                             best_labels, x_test, y_test)

# = myThread("Thread-%d" % j, 4, 0.001, nn.Tanh(), 256, best_train, best_labels, x_test, y_test)
best_model_thread.start()
best_model_thread.join()

best_model_test_index = -1
print("Best Model: Hidden Layer: %d - Learning Rate: %f - Neurons: %d - Activation Function: %s" %
      (stats[best_model_test_index][0], stats[best_model_test_index][1], stats[best_model_test_index][2],
       stats[best_model_test_index][3]))
print("Mean: %.2f" % stats[best_model_test_index][4])
print("Std: %.2f" % stats[best_model_test_index][5])
print("Lower Bound: %.2f" % stats[best_model_test_index][6])
print("Upper Bound: %.2f" % stats[best_model_test_index][7])
