# ¿Cómo reducir el underfitting?

Código fuente de [este](https://youtu.be/m-gAi6u-iNQ) video, en donde se muestra un tutorial paso a paso en Keras para reducir el underfitting en un modelo de Deep Learning.

El objetivo es clasificar el set [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), que contiene 60.000 imágenes a color pertenecientes a 10 posibles categorías. Al usar modelos como redes neuronales se evidencia el problema del underfitting, alcanzando niveles de precisión (con los sets de entrenamiento y validación) que no superan el 60%. Sin embargo, al cambiar el modelo por una red convolucional, la precisión en ambos sets se incrementa hasta poco más del 85%.

## Dependencias
Keras==2.2.4
numpy==1.16.3