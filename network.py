import numpy as np
from typing import List
import random
import csv

def get_mnist():
    # The code to download the mnist data original came from
    # https://cntk.ai/pythondocs/CNTK_103A_MNIST_DataLoader.html
    
    import gzip
    import os
    import struct

    from urllib.request import urlretrieve 

    def load_data(src, num_samples):
        print("Downloading " + src)
        gzfname, h = urlretrieve(src, "./delete.me")
        print("Done.")
        try:
            with gzip.open(gzfname) as gz:
                n = struct.unpack("I", gz.read(4))
                # Read magic number.
                if n[0] != 0x3080000:
                    raise Exception("Invalid file: unexpected magic number.")
                # Read number of entries.
                n = struct.unpack(">I", gz.read(4))[0]
                if n != num_samples:
                    raise Exception(
                        "Invalid file: expected {0} entries.".format(num_samples)
                    )
                crow = struct.unpack(">I", gz.read(4))[0]
                ccol = struct.unpack(">I", gz.read(4))[0]
                if crow != 28 or ccol != 28:
                    raise Exception(
                        "Invalid file: expected 28 rows/cols per image."
                    )
                # Read data.
                res = np.frombuffer(
                    gz.read(num_samples * crow * ccol), dtype=np.uint8
                )
        finally:
            os.remove(gzfname)
        return res.reshape((num_samples, crow, ccol)) / 256


    def load_labels(src, num_samples):
        print("Downloading " + src)
        gzfname, h = urlretrieve(src, "./delete.me")
        print("Done.")
        try:
            with gzip.open(gzfname) as gz:
                n = struct.unpack("I", gz.read(4))
                # Read magic number.
                if n[0] != 0x1080000:
                    raise Exception("Invalid file: unexpected magic number.")
                # Read number of entries.
                n = struct.unpack(">I", gz.read(4))
                if n[0] != num_samples:
                    raise Exception(
                        "Invalid file: expected {0} rows.".format(num_samples)
                    )
                # Read labels.
                res = np.frombuffer(gz.read(num_samples), dtype=np.uint8)
        finally:
            os.remove(gzfname)
        return res.reshape((num_samples))


    def try_download(data_source, label_source, num_samples):
        data = load_data(data_source, num_samples)
        labels = load_labels(label_source, num_samples)
        return data, labels
    
    # Not sure why, but yann lecun's website does no longer support 
    # simple downloader. (e.g. urlretrieve and wget fail, while curl work)
    # Since not everyone has linux, use a mirror from uni server.
    #     server = 'http://yann.lecun.com/exdb/mnist'
    server = 'https://raw.githubusercontent.com/fgnt/mnist/master'
    
    # URLs for the train image and label data
    url_train_image = f'{server}/train-images-idx3-ubyte.gz'
    url_train_labels = f'{server}/train-labels-idx1-ubyte.gz'
    num_train_samples = 60000

    print("Downloading train data")
    train_features, train_labels = try_download(url_train_image, url_train_labels, num_train_samples)

    # URLs for the test image and label data
    url_test_image = f'{server}/t10k-images-idx3-ubyte.gz'
    url_test_labels = f'{server}/t10k-labels-idx1-ubyte.gz'
    num_test_samples = 10000

    print("Downloading test data")
    test_features, test_labels = try_download(url_test_image, url_test_labels, num_test_samples)
    
    return train_features, train_labels, test_features, test_labels

train_images, train_labels, test_images, test_labels = get_mnist()
print(train_images.shape) # (60000, 28, 28)
print(train_labels.shape) # (60000,)
print(test_images.shape) # (10000, 28, 28)
print(test_labels.shape) # (10000,)

# flatten data
train_images = train_images.reshape(train_images.shape[0], 784, 1)
test_images = test_images.reshape(test_images.shape[0], 784, 1)

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

training_data = [(i, l) for i, l in zip(train_images, train_labels)]
test_data = [(i, l) for i, l in zip(test_images, test_labels)]

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

class Network:
    num_layers: int
    sizes: List[int]
    biases: List[np.ndarray]
    weights: List[np.ndarray]

    def __init__(self, sizes: List[int]):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def forward(self, a: np.ndarray) -> np.ndarray:
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight, a) + bias)
        return a

    # returns a list of tuples containing [epoch_number, correct, total]
    def sgd(self, training_data: List[tuple[np.ndarray, int]], epochs: int, 
            mini_batch_size: int, eta: float, test_data: List[tuple[np.ndarray, int]] = None) -> List[tuple[int, int, int]]:
        n = len(training_data)
        results = []

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                n_test = len(test_data)
                correct = self.evaluate(test_data)
                print(f"Epoch {j}: {correct} / {n_test} correct")
                results.append([j, correct, n_test])
            else:
                print(f"Epoch {j} complete")

        return results

    def update_mini_batch(self, mini_batch: List[tuple[np.ndarray, int]], eta: float):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        mb_size = len(mini_batch)
        self.weights = [w - (eta / mb_size) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / mb_size) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x: np.ndarray, y: int) -> tuple[List[np.ndarray], List[np.ndarray]]:
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        y_one_hot = np.zeros((self.sizes[-1], 1))
        y_one_hot[y] = 1.0

        delta = self.cost_derivative(activations[-1], y_one_hot) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data: List[tuple[np.ndarray, int]]) -> int:
        test_res = ((np.argmax(self.forward(x)), y) for (x, y) in test_data)
        return sum(int(x == y) for  (x, y) in test_res)

    def cost_derivative(self, output_activations: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (output_activations - y)

configs = [
    [[784, 128, 64, 10], 10, 32, 0.01], # baseline
    [[784, 128, 64, 10], 10, 256, 0.01], # larger mini-batch
    [[784, 128, 64, 10], 10, 32, 1.0], # higher learning rate
    [[784, 128, 64, 10], 10, 32, 10.0], # HIGHER learning rate
    [[784, 128, 64, 10], 10, 32, 0.001], # lower learning rate
    [[784, 128, 64, 10], 10, 32, 0.0001], # LOWER learning rate
    # different layer configs
    [[784, 512, 10], 10, 32, 0.01],
    [[784, 2, 10], 10, 32, 0.01],
    [[784, 16, 16, 10], 10, 32, 0.01],
    [[784, 4, 4, 4, 10], 10, 32, 0.01],
    [[784, 30, 30, 10], 10, 32, 0.01],
]

with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Layers", "Epochs", "Mini-Batch Size", "Learning Rate", "Final Accuracy", "Epoch Outputs [#, # Correct, Total]"])
    for config in configs:
        for _ in range(0, 3):
            net = Network(config[0])
            epoch_outputs = net.sgd(training_data, config[1], config[2], config[3], test_data=test_data)
            final_acc = epoch_outputs[-1][1] / epoch_outputs[-1][2]
            writer.writerow([config[0], config[1], config[2], config[3], final_acc, epoch_outputs])
