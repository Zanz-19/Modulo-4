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