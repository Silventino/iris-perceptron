import numpy as np
import random
import pprint

class Perceptron(object):
    """Implements a perceptron network"""
    def __init__(self, input_size, lr=1, epochs=100):
        self.W = np.zeros(input_size+1)
        # add one for bias
        self.epochs = epochs
        self.lr = lr
    
    def activation_fn(self, x):
        # 1 = 'Iris-setosa'
        # 0 = 'Iris-versicolor'
        return 1 if x >= 0 else 0
 
    def predict(self, x):
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a
 
    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(len(X)):
                x = np.insert(X[i], 0, 1)
                y = self.predict(x)
                e = d[i] - y
                self.W = self.W + self.lr * e * x

dataset = []
with open('./iris.txt', 'r') as arquivo:
    linhas = arquivo.readlines()
    for linha in linhas:
        dados = linha.rstrip().split(',')
        for i in range(len(dados)-1):
            dados[i] = float(dados[i])
        if(len(dados) == 5):
            if(dados[4] != 'Iris-virginica'):
                dataset.append(dados)
random.shuffle(dataset)

resultado = []
for i in range(len(dataset)):
    if(dataset[i][-1] == 'Iris-setosa'):
        resultado.append(1)
    else:
        resultado.append(0)
    dataset[i] = dataset[i][:-1]

cut = int(len(dataset)*0.05)

training = dataset[:cut]
training_answear = resultado[:cut]

pprint.pprint(training)

p = Perceptron(4, 0.1, 100)
p.fit(training, training_answear)

test = dataset[cut:]
test_answear = resultado[cut:]

# print(test)

acertos = 0
for i in range(len(test)):
    test[i].insert(0,0)
    predicted = p.predict(test[i])

    if(predicted == test_answear[i]):
        acertos += 1
        
    if(predicted == 1):
        predicted = "Iris-setosa"
    else:
        predicted = "Iris-versicolor"
    
    if(test_answear == 1):
        r = "Iris-setosa"
    else:
        r = "Iris-versicolor"

    print("Predicted: " + str(predicted))
    print("Answear: " + str(r) + "\n")

    

print("\nTotal de acertos: " + str(acertos))
print("de " + str(len(test)) + " tentativas.")

# 1 = 'Iris-setosa'
# 0 = 'Iris-versicolor'