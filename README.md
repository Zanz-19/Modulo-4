# Deep Learning — Matemáticas Fundamentales

El deep learning es una rama del machine learning que utiliza redes neuronales profundas para aprender representaciones jerárquicas de los datos. En lugar de definir reglas a mano, la red ajusta sus parámetros internos (pesos y sesgos) a través de optimización basada en gradientes, lo que le permite modelar relaciones complejas y no lineales en datos de cualquier tipo: imágenes, texto, audio, series de tiempo, etc.

Los bloques matemáticos que lo sostienen son simples pero poderosos: funciones lineales y afines, álgebra lineal (multiplicación de matrices), funciones no lineales de activación, y derivadas/gradientes que guían el aprendizaje.

---

<details>
<summary>📁 1er_Tarea</summary>

## 1er_Tarea

### 📘 1.1 — Math Warm-Up

Notebook de calentamiento matemático que cubre los conceptos fundamentales necesarios para entender cómo funciona una red neuronal desde adentro.

**Temas cubiertos**

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
<summary>📁 2da_Tarea</summary>

## 2da_Tarea

### 📗 1.2 — Regresión Lineal

Notebook que introduce el modelo de regresión lineal 1D y la función de pérdida de mínimos cuadrados, sentando las bases del entrenamiento supervisado.

**Temas cubiertos**

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

### 📙 2.1 — Redes Neuronales Superficiales

Notebook que construye y explora redes neuronales de una sola capa oculta (shallow networks), partiendo de la función de activación ReLU hasta arquitecturas con múltiples entradas y salidas.

**Temas cubiertos**

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

---

<details>
<summary>📁 3ra_Tarea</summary>

## 3ra_Tarea

### 📕 2.2 — Composing Neural Networks

Notebook que explora qué ocurre al **componer** (encadenar) redes superficiales, alimentando la salida de una red como entrada de otra.

**Temas cubiertos**

**1. Composición de dos redes superficiales**
Se alimenta la salida de la Red 1 como entrada de la Red 2: `net12_out = shallow_1_1_3(net1_out, ...)`. La curva resultante muestra más segmentos lineales que cualquiera de las dos redes por separado.

**2. Efecto de modificar los parámetros de salida ($\phi$)**
Invertir el signo de un peso de salida de la Red 2 cambia la forma de uno de sus segmentos, alterando la composición final en las regiones donde la salida de la Red 1 cae dentro de ese segmento.

**3. Efecto de escalar los parámetros de salida de la primera red**
Reducir la amplitud de la Red 1 comprime el rango de entrada que llega a la Red 2, recorriendo una porción más pequeña de su dominio.

**4. Composición de una red consigo misma**
Al encadenar la Red 1 con una copia idéntica de sí misma, la no linealidad de la ReLU genera más puntos de quiebre que la red original.

**5. Crecimiento exponencial de regiones lineales**
Al encadenar $N$ copias de una red de 3 neuronas (hasta 4 segmentos cada una) antes de una red final, el número máximo de regiones lineales crece como $4^{N+1}$ — la idea central detrás del poder expresivo de las redes profundas con pocos parámetros.

### 📕 2.3 — Deep Neural Networks

Notebook que introduce la **notación matricial** para redes neuronales y la extiende a arquitecturas profundas con múltiples capas ocultas.

**Temas cubiertos**

**1. Forma matricial de una red superficial**
Implementación de la ecuación 4.15: $\mathbf{h}_1 = \text{ReLU}(\boldsymbol\beta_0 + \boldsymbol\Omega_0 \mathbf{x})$, $y = \boldsymbol\beta_1 + \boldsymbol\Omega_1 \mathbf{h}_1$, agrupando los parámetros $\theta$ y $\phi$ en matrices y vectores.

**2. Composición de dos redes en forma matricial**
Se deriva cómo los parámetros $\phi$ de la primera red se "absorben" dentro de $\boldsymbol\beta_1$ y $\boldsymbol\Omega_1$ de la segunda red al componerlas. Verificado numéricamente: la composición matricial coincide con la versión escalar con un error del orden de $10^{-16}$ (precisión de punto flotante).

**3. Red profunda de 3 capas ocultas**
Construcción de una red con $D_i=4$ entradas, capas ocultas de tamaños $D_1=5$, $D_2=2$, $D_3=4$, y $D_o=1$ salida. Se determinaron las formas correctas de cada matriz $\boldsymbol\beta_k$ ($D_k \times 1$) y $\boldsymbol\Omega_k$ ($D_k \times D_{k-1}$).

### 📕 2.4.1 — Loss Function I (Regresión con ruido gaussiano)

Notebook que deriva la función de pérdida de mínimos cuadrados a partir de primeros principios estadísticos, asumiendo ruido gaussiano en las observaciones.

**Temas cubiertos**

**1. Distribución normal**
Implementación directa de la densidad gaussiana (ecuación 5.7): $\text{Pr}(y\mid\mu,\sigma) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y-\mu)^2}{2\sigma^2}\right)$.

**2. Likelihood**
Cálculo de la likelihood conjunta de los datos como el producto de las probabilidades individuales: $L(\boldsymbol\phi) = \prod_i \text{Pr}(y_i\mid\mu_i,\sigma)$.

**3. Log-likelihood negativa**
Conversión del producto de probabilidades en una suma de logaritmos para evitar subdesbordamiento numérico: $-\log L(\boldsymbol\phi) = -\sum_i \log\text{Pr}(y_i\mid\mu_i,\sigma)$.

**4. Suma de cuadrados**
Implementación de la pérdida clásica de mínimos cuadrados y comprobación de que **maximizar la likelihood gaussiana, minimizar la log-likelihood negativa y minimizar la suma de cuadrados son equivalentes**: las tres métricas convergen exactamente en el mismo valor óptimo del parámetro.

### 📕 2.4.2 — Loss Function II (Clasificación binaria)

Notebook que extiende el marco de máxima verosimilitud al caso de clasificación binaria, derivando la entropía cruzada como función de pérdida.

**Temas cubiertos**

**1. Función sigmoide**
Implementación de $\text{sig}(z) = \frac{1}{1+e^{-z}}$, que transforma la salida arbitraria de la red en una probabilidad $\lambda \in (0,1)$.

**2. Distribución de Bernoulli**
Implementación de $\text{Pr}(y\mid\lambda) = \lambda^{y}(1-\lambda)^{1-y}$, que da la probabilidad de la clase observada dado el parámetro predicho por la red.

**3. Likelihood y log-likelihood negativa**
Cálculo de la likelihood conjunta y su versión logarítmica negativa, que resulta ser exactamente la **entropía cruzada binaria** — la función de pérdida estándar para clasificación binaria en redes neuronales.

</details>

---

<details>
<summary>📁 4ta_Tarea</summary>

## 4ta_Tarea

### 📕 2.4.3 — Loss Function III (Clasificación multiclase)

Notebook que cierra la serie de funciones de pérdida derivadas por máxima verosimilitud, extendiendo el marco de Bernoulli a múltiples clases mediante la distribución categórica.

**Temas cubiertos**

**1. Función softmax**
Implementación de $\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$, aplicada por columna, con resta del máximo para estabilidad numérica y evitar overflow en `np.exp`. Transforma las salidas arbitrarias de la red en probabilidades no negativas que suman uno.

**2. Distribución categórica**
Implementación de $\text{Pr}(y=k\mid\boldsymbol\lambda) = \lambda_k$, que devuelve la probabilidad de la clase observada según los parámetros predichos por la red.

**3. Likelihood**
Cálculo de la likelihood conjunta como el producto de las probabilidades categóricas de cada punto: $L(\boldsymbol\phi) = \prod_i \text{Pr}(y_i\mid\boldsymbol\lambda_i)$.

**4. Log-likelihood negativa**
Conversión a suma de logaritmos negativos para evitar el subdesbordamiento numérico que produce el producto de muchas probabilidades pequeñas: $-\log L(\boldsymbol\phi) = -\sum_i \log\text{Pr}(y_i\mid\boldsymbol\lambda_i)$.

**5. Verificación del óptimo**
Barrido del parámetro $\beta_1$ manteniendo el resto fijo, graficando likelihood y NLL en función de su valor. Se comprueba que **el máximo de la likelihood y el mínimo de la NLL ocurren exactamente en el mismo punto**, confirmando por qué en la práctica se optimiza la NLL en vez de la likelihood directa.

### 📗 3.1 — Optimización I (Descenso por gradiente sobre un modelo Gabor)

Notebook exploratorio que anima el proceso de descenso por gradiente sobre un modelo no convexo, para observar de forma visual e intuitiva el efecto de la tasa de aprendizaje.

**Temas cubiertos**

**1. Modelo y datos sintéticos**
Ajuste de una función tipo Gabor $f(x,\phi_0,\phi_1) = \sin(z)\cdot e^{-z^2/8}$, con $z = \phi_0 + 0.06\,\phi_1 x$, sobre datos generados con parámetros verdaderos conocidos más ruido gaussiano.

**2. Gradiente analítico**
Derivación de $\partial L/\partial\phi_0$ y $\partial L/\partial\phi_1$ vía regla de la cadena sobre $z$, para la pérdida MSE $L = \tfrac{1}{2}\text{mean}(r^2)$.

**3. Animación del descenso**
Visualización simultánea de la trayectoria de los parámetros sobre las curvas de nivel de la superficie de pérdida (panel izquierdo) y el ajuste del modelo a los datos conforme avanzan las iteraciones (panel derecho).

**4. Efecto de la tasa de aprendizaje**
Experimentación con distintos puntos iniciales y tasas de aprendizaje (`lr`). Se comprobó que un `lr` muy pequeño converge de forma estable pero lenta, uno intermedio ofrece el mejor equilibrio entre velocidad y estabilidad, y uno demasiado alto introduce oscilaciones notorias en la trayectoria de los parámetros — comportamiento verificado corriendo el experimento con varias semillas de ruido.

### 📙 4.1 — Descenso por gradiente

Notebook que implementa desde cero el algoritmo de descenso por gradiente para ajustar un modelo lineal simple, incluyendo búsqueda lineal para el tamaño de paso.

**Temas cubiertos**

**1. Modelo lineal y pérdida de suma de cuadrados**
Implementación de $f(x,\boldsymbol\phi) = \phi_0 + \phi_1 x$ y $L(\boldsymbol\phi) = \sum_i(f(x_i,\boldsymbol\phi)-y_i)^2$, verificada contra un valor de referencia conocido.

**2. Gradiente analítico**
Derivación de $\partial L/\partial\phi_0 = 2\sum_i(\hat y_i - y_i)$ y $\partial L/\partial\phi_1 = 2\sum_i(\hat y_i-y_i)x_i$, verificado con diferencias finitas.

**3. Búsqueda lineal**
Rutina que evalúa la pérdida en varios puntos a lo largo de la dirección de descenso para elegir el tamaño de paso $\alpha$ que la minimiza, en vez de fijar un valor arbitrario.

**4. Paso de descenso por gradiente**
Ensamble de los pasos anteriores: calcular el gradiente, encontrar $\alpha$ óptimo por búsqueda lineal, y actualizar $\boldsymbol\phi \leftarrow \boldsymbol\phi - \alpha\nabla L$. Se visualiza la trayectoria de los parámetros sobre la superficie de pérdida a lo largo de las iteraciones.

### 📙 4.2 — Descenso por gradiente estocástico

Notebook que compara distintas variantes del descenso por gradiente sobre un modelo Gabor no convexo: búsqueda lineal, paso fijo, mini-batch y *scheduler* de tasa de aprendizaje.

**Temas cubiertos**

**1. Modelo Gabor y su gradiente**
Implementación de la pérdida de suma de cuadrados sobre $f(x,\boldsymbol\phi)=\sin(z)e^{-z^2/32}$, con $z=\phi_0+0.06\phi_1 x$, y derivación explícita de $\partial L/\partial\phi_0$ y $\partial L/\partial\phi_1$ vía regla de la cadena.

**2. Descenso con búsqueda lineal**
Reutilización de la búsqueda lineal para elegir $\alpha$ en cada paso; se observa que, al ser la superficie no convexa, **el punto de convergencia depende de la inicialización** (distintos valles/mínimos locales).

**3. Descenso con paso fijo**
Actualización $\boldsymbol\phi \leftarrow \boldsymbol\phi - \alpha\nabla L$ sin búsqueda lineal. Se comprobó que un $\alpha$ alto provoca oscilaciones que saltan las zonas de baja pérdida, mientras que uno muy bajo converge de forma estable pero lenta.

**4. Mini-batch SGD**
Estimación del gradiente usando solo un subconjunto aleatorio de los datos ($\mathcal{B}_t$) en cada paso, mediante `np.random.permutation`. Batches pequeños introducen más variabilidad en la trayectoria pero permiten más actualizaciones por época; batches grandes se acercan más al gradiente completo.

**5. Scheduler de tasa de aprendizaje**
Reducción de $\alpha$ por un factor $\beta$ cada $M$ iteraciones, sacrificando exploración al final del entrenamiento para ganar estabilidad y evitar saltos grandes cerca del mínimo.

### 📘 4.3 — Momentum

Notebook que compara SGD estándar, Momentum y Momentum de Nesterov sobre el mismo modelo Gabor no convexo.

**Temas cubiertos**

**1. SGD estándar (referencia)**
Actualización directa $\boldsymbol\phi \leftarrow \boldsymbol\phi - \alpha\, g_t$ usando el gradiente de un mini-batch aleatorio, como línea base de comparación.

**2. Momentum**
Implementación de la velocidad acumulada $m_t = \beta m_{t-1} + (1-\beta)g_t$ y actualización $\boldsymbol\phi_{t+1} = \boldsymbol\phi_t - \alpha m_t$. El promedio móvil del gradiente suaviza la trayectoria y reduce las oscilaciones típicas del SGD puro.

**3. Momentum de Nesterov**
Variante que evalúa el gradiente en una posición **adelantada** por la velocidad acumulada previa ($\boldsymbol\phi_t - \alpha\beta m_{t-1}$) antes de actualizar $m_t$, anticipando hacia dónde se moverá el parámetro. En este experimento no superó a Momentum estándar.

### 📔 4.4 — Adam

Notebook que construye el optimizador Adam paso a paso, partiendo de descenso con paso fijo, pasando por gradientes normalizados por coordenada, hasta el algoritmo completo con corrección de sesgo.

**Temas cubiertos**

**1. Descenso con paso fijo (referencia)**
Línea base $\boldsymbol\phi_{t+1} = \boldsymbol\phi_t - \alpha\nabla L$ sobre una función de pérdida con curvatura muy distinta en cada dimensión, motivando la necesidad de un tamaño de paso adaptativo por coordenada.

**2. Gradientes normalizados**
Actualización por coordenada $m_t=g_t$, $v_t=g_t^2$, $\boldsymbol\phi_{t+1}=\boldsymbol\phi_t-\alpha\, m_t/(\sqrt{v_t}+\epsilon)$. Cada dimensión avanza con un tamaño de paso ajustado a la magnitud de su propio gradiente, mitigando el problema de curvaturas desiguales.

**3. Adam**
Promedios móviles del gradiente y del gradiente al cuadrado, $m_t=\beta m_{t-1}+(1-\beta)g_t$ y $v_t=\gamma v_{t-1}+(1-\gamma)g_t^2$, con corrección de sesgo $\hat m_t = m_t/(1-\beta^t)$, $\hat v_t = v_t/(1-\gamma^t)$ — necesaria porque $m_0=v_0=0$ sesgan las primeras iteraciones hacia cero. Actualización final: $\boldsymbol\phi_{t+1}=\boldsymbol\phi_t-\alpha\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$.

</details>


---

<details>
<summary>📁 5ta_Tarea</summary>

## 5ta_Tarea

### 📓 6.1 — Convolución 1D

Notebook que implementa desde cero, sin usar rutinas de biblioteca, la operación de convolución 1D en sus distintas variantes de kernel, paso y dilatación.

**Temas cubiertos**

**1. Convolución con kernel 3, paso 1, dilatación 1**
Implementación con relleno de ceros: cada salida combina el punto actual con sus dos vecinos inmediatos, ponderados por los pesos del kernel $\omega$.

**2. Convolución con kernel 3, paso 2, dilatación 1**
El centro del kernel para la salida $i$ cae en la posición de entrada $2i$; se verificó que el resultado coincide exactamente con tomar uno de cada dos valores de la convolución equivalente con paso 1.

**3. Convolución con kernel 5, paso 1, dilatación 1**
Kernel más ancho: el offset del filtro va de $-2$ a $+2$ alrededor de cada posición.

**4. Convolución con kernel 3, paso 1, dilatación 2**
Se deja un hueco entre elementos consecutivos del filtro: el offset avanza de dos en dos ($-2, 0, +2$) en vez de uno en uno.

**5. Representación matricial de la convolución**
Construcción de la matriz de convolución $\boldsymbol\Omega$ (kernel 3, paso 1) colocando los pesos alrededor de la diagonal en cada fila. Se comprobó que $\boldsymbol\Omega\mathbf{x}$ produce exactamente el mismo resultado que la convolución calculada directamente.

### 📓 6.2 — Convolución para MNIST-1D

Notebook que construye y entrena una red convolucional 1D sobre el dataset sintético MNIST-1D, comparándola contra el enfoque totalmente conectado.

**Temas cubiertos**

**1. Arquitectura convolucional**
Construcción de la red con `torch.nn`: tres capas `Conv1d` (kernel 3, paso 2, sin relleno) intercaladas con `ReLU`, seguidas de `Flatten` y una capa `Linear` final. Se verificó a mano cómo se reduce la longitud de la secuencia en cada capa: $40 \to 19 \to 9 \to 4$, dando $4\times 15=60$ valores antes de la capa lineal.

**2. Entrenamiento**
Entrenamiento con `CrossEntropyLoss`, optimizador SGD con momentum y un *scheduler* que reduce la tasa de aprendizaje a la mitad cada 20 épocas. A lo largo de 100 épocas el error de validación bajó de ~77% a ~9%, confirmando que la arquitectura aprende correctamente.

### 📓 6.3 — Convolución 2D

Notebook que implementa la convolución 2D en NumPy en cuatro niveles progresivos de complejidad, validando cada uno contra `torch.nn.functional.conv2d`.

**Temas cubiertos**

**1. Convolución 2D básica**
Un canal de entrada, un canal de salida, una sola imagen: recorrido anidado sobre alto y ancho de la salida, y sobre alto y ancho del kernel.

**2. Convolución con paso (stride)**
Se generaliza el índice de la imagen a `c_y * stride + c_kernel_y` (y análogo en x), permitiendo pasos mayores a 1.

**3. Múltiples canales de entrada y salida**
Se añaden los recorridos sobre canal de entrada y canal de salida, acumulando la suma sobre todos los canales de entrada para cada canal de salida.

**4. Convolución completa (batch, multicanal, con paso)**
Versión final que añade el recorrido sobre el tamaño de lote (batch), quedando equivalente a una capa `Conv2d` estándar. En los cuatro niveles, el error absoluto medio contra PyTorch fue del orden de $10^{-7}$, confirmando la correcta implementación.

</details>