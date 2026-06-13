# Deep Learning — Matemáticas Fundamentales

El deep learning es una rama del machine learning que utiliza redes neuronales profundas para aprender representaciones jerárquicas de los datos. En lugar de definir reglas a mano, la red ajusta sus parámetros internos (pesos y sesgos) a través de optimización basada en gradientes, lo que le permite modelar relaciones complejas y no lineales en datos de cualquier tipo: imágenes, texto, audio, series de tiempo, etc.

Los bloques matemáticos que lo sostienen son simples pero poderosos: funciones lineales y afines, álgebra lineal (multiplicación de matrices), funciones no lineales de activación, y derivadas/gradientes que guían el aprendizaje.

---

<details>
<summary>📘 Tarea 1.1 — Math Warm-Up</summary>

## Tarea 1.1 — Math Warm-Up

Notebook de calentamiento matemático que cubre los conceptos fundamentales necesarios para entender cómo funciona una red neuronal desde adentro.

### Temas cubiertos

**1. Funciones lineales en 1D**
Implementación de $y = \beta + \omega x$, donde $\beta$ es el intercepto y $\omega$ la pendiente. Se exploró el efecto de cada parámetro sobre la recta.

**2. Funciones lineales en 2D e hiperplanos**
Extensión a dos entradas: $y = \beta + \omega_1 x_1 + \omega_2 x_2$. Visualización mediante gráficos de contorno para entender cómo cada parámetro inclina o desplaza el plano.

**3. Álgebra lineal — operaciones por lote**
Implementación de la operación matricial $Y = \beta + X\Omega^\top$, que permite calcular múltiples funciones lineales sobre múltiples puntos de datos en una sola operación. Base de lo que hace una capa densa en cualquier red neuronal.

**4. Funciones polinomiales y no linealidad**
Comparación visual entre funciones lineales, cuadráticas y cúbicas. Se estableció por qué las activaciones no lineales son indispensables: apilar capas lineales sin ellas sigue siendo una función lineal.

**5. Derivadas y líneas tangentes**
Cálculo de la derivada como pendiente instantánea. Implementación de la **diferencia central**:

$$f'(x) \approx \frac{f(x+\varepsilon) - f(x-\varepsilon)}{2\varepsilon}$$

Se comparó la derivada numérica contra la analítica sobre $\sin(x)$.

**6. Gradientes**
Extensión de la derivada a múltiples variables. Visualización del campo gradiente de una función cuenco, observando cómo los vectores apuntan cuesta arriba y cómo el descenso de gradiente los sigue en dirección contraria.

</details>

---

<details>
<summary>📗 Tarea 1.2 — Regresión Lineal</summary>

## Tarea 1.2 — Regresión Lineal

Notebook que introduce el modelo de regresión lineal 1D y la función de pérdida de mínimos cuadrados, sentando las bases del entrenamiento supervisado.

### Temas cubiertos

**1. Modelo de regresión lineal 1D**
Implementación de $f(x, \phi_0, \phi_1) = \phi_0 + \phi_1 x$, donde $\phi_0$ es el intercepto y $\phi_1$ la pendiente. Se generalizó la notación de $\beta/\omega$ a $\phi_0/\phi_1$, que es la convención usada en el resto del curso.

**2. Función de pérdida — Mínimos cuadrados**
Implementación de la pérdida de la ecuación 2.5:

$$L[\phi_0, \phi_1] = \sum_{i=1}^{I} \left(y_i - f(x_i, \phi_0, \phi_1)\right)^2$$

Mide la suma de los cuadrados de los residuos entre los valores reales y las predicciones del modelo.

**3. Descenso coordinado**
Optimización manual e implementación automática del descenso coordinado: se alterna la minimización respecto a $\phi_0$ (con $\phi_1$ fijo) y respecto a $\phi_1$ (con $\phi_0$ fijo), repitiendo hasta convergencia.

**4. Visualización de la superficie de pérdida**
Generación del mapa de calor de $L[\phi_0, \phi_1]$ sobre una grilla 2D de parámetros. Se identifica el mínimo global y se verifica que los parámetros encontrados por descenso coordinado se acercan a él.

</details>

---

<details>
<summary>📙 Tarea 2.1 — Redes Neuronales Superficiales</summary>

## Tarea 2.1 — Redes Neuronales Superficiales

Notebook que construye y explora redes neuronales de una sola capa oculta (shallow networks), partiendo de la función de activación ReLU hasta arquitecturas con múltiples entradas y salidas.

### Temas cubiertos

**1. Función de activación ReLU**
Implementación de $\text{ReLU}(z) = \max(0, z)$ usando `np.maximum`. Es la función de activación más utilizada en redes modernas: devuelve 0 para entradas negativas (neurona "apagada") y el valor sin cambios para entradas positivas (neurona "activa").

**2. Red neuronal (1 entrada, 1 salida, 3 neuronas ocultas)**
Implementación completa del flujo hacia adelante (*forward pass*):

$$y = \phi_0 + \phi_1 \cdot \text{ReLU}(\theta_{10} + \theta_{11}x) + \phi_2 \cdot \text{ReLU}(\theta_{20} + \theta_{21}x) + \phi_3 \cdot \text{ReLU}(\theta_{30} + \theta_{31}x)$$

Produce una función **lineal por partes** con hasta 4 segmentos. Se visualizaron preactivaciones, activaciones y activaciones ponderadas de cada neurona.

**3. Exploración de parámetros**
Análisis del efecto de modificar cada grupo de parámetros: $\phi_0$ desplaza la salida verticalmente; escalar $\phi_1, \phi_2, \phi_3$ escala la amplitud; escalar simultáneamente $\theta_{ix}$ y $\phi_i$ en forma inversa produce la misma función (invarianza por reescalado).

**4. Pérdida de mínimos cuadrados sobre la red**
Implementación de $L[\boldsymbol{\phi}] = \sum_{i=1}^{I}(y_i - f[x_i, \boldsymbol{\phi}])^2$ aplicada a la salida de la red neuronal. Se ajustaron los parámetros manualmente para minimizar la pérdida sobre un conjunto de 20 puntos de entrenamiento.

**5. Red neuronal (2 entradas, 1 salida, 3 neuronas ocultas)**
Extensión a dos entradas: cada preactivación define un hiperplano en $\mathbb{R}^2$:

$$\text{pre}_i = \theta_{i0} + \theta_{i1} x_1 + \theta_{i2} x_2$$

Las fronteras donde $\text{pre}_i = 0$ son líneas rectas que dividen el plano de entrada en regiones — los **politopos lineales**. Con 3 neuronas se crean hasta 7 regiones en posición general.

**6. Red neuronal (2 entradas, 2 salidas, 3 neuronas ocultas)**
Arquitectura con dos salidas que comparten las mismas preactivaciones y activaciones, pero con conjuntos de pesos de salida independientes $\{\phi_{1x}\}$ y $\{\phi_{2x}\}$. Ambas salidas tienen la misma estructura de politopos pero diferente forma dentro de cada región.

</details>
