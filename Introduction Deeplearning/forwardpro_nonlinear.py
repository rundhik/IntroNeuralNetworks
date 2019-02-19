import numpy as np

def forwardPass(masukan,bobot, bias, akt = 'linear') :
    jml_bobot = np.dot(masukan, bobot) + bias
    if akt is 'relu' :
        aktivasi = np.maximum(jml_bobot, 0)
    else:
        aktivasi = jml_bobot
    return aktivasi

bobot_H = np.array([
    [0.00192761, -0.78845304, 0.30310717, 0.44131625, 0.32792646, -0.02451803, 1.43445349, -1.12972116]
])
bias_H = np.array([-0.02657719, -1.15885878, -0.79183501, -0.33550513, -0.23438406, -0.25078532, 0.22305705, 0.80253315])

bobot_o = np.array([
    [-0.77540326], [ 0.5030424 ], [ 0.37374797], [-0.20287184], [-0.35956827], [-0.54576212], [ 1.04326093], [ 0.8857621 ]
])
bias_o = np.array([0.04351173])

inputan = np.array([[-2],[0],[2]])

keluaran_hiddenlayer = forwardPass(inputan, bobot_H, bias_H, 'relu')

print('Hidden layer output (Relu)')
print('=============================')
print(keluaran_hiddenlayer, "\n")


keluaran_final = forwardPass(keluaran_hiddenlayer, bobot_o, bias_o, 'linear')

print('Output layer output (linear)')
print('=============================')
print(keluaran_final, "\n")