import numpy as np

def forwardPass(masukan,bobot, bias) :
    jml_bobot = np.dot(masukan, bobot) + bias
    aktivasi = jml_bobot
    return aktivasi

bobot = np.array([
    [2.99999928]
])
bias = np.array([
    [1.99999976]
])

inputan = np.array([
    [7],
    [8],
    [9],
    [10]
])

outputan = forwardPass(inputan, bobot, bias)

print('Output layer output (linear)')
print('=============================')
print(outputan, "\n")