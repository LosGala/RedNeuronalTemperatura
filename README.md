#RedNeuronalTemperatura


````markdown
# Predicción de Temperatura: Celsius a Fahrenheit

Este proyecto utiliza una red neuronal con TensorFlow para predecir la temperatura en Fahrenheit a partir de un valor en Celsius.

## Requisitos

Instala las bibliotecas necesarias:

```bash
pip install tensorflow numpy matplotlib
````

## Instrucciones

1. **Definir los datos**: El proyecto usa un conjunto de datos de temperaturas en Celsius y Fahrenheit.
2. **Crear el modelo**: El modelo tiene dos capas ocultas y una capa de salida para hacer la predicción.
3. **Entrenar el modelo**: Se entrena el modelo durante 1000 épocas con los datos de entrada.
4. **Visualizar la pérdida**: Se grafica la magnitud de la pérdida durante el entrenamiento.
5. **Predecir la temperatura**: Se puede predecir la temperatura en Fahrenheit para cualquier valor en Celsius.

## Código

```python
import tensorflow as tf
import numpy as np

# Datos de entrada
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Definir el modelo
oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

# Compilar el modelo
modelo.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

# Entrenar el modelo
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)

# Visualizar la pérdida
import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])

# Hacer una predicción
resultado = modelo.predict([100.0])
print("El resultado es " + str(resultado) + " Fahrenheit")
```

## Agradecimientos

* **TensorFlow** para la red neuronal.
* **NumPy** para manejar los datos.
* **Matplotlib** para graficar la pérdida.

## Licencia

Este proyecto está bajo la Licencia MIT.


