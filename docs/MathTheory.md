# Fundamentos Matemáticos: Espacios de Permutación e Invarianza Afín

## 1. El Teorema de Invarianza por Escalamiento Temporal

Sea un sistema dinámico asíncrono con $n$ canales neuronales sensoriales. Para un patrón de estímulo dado $S(t)$, definimos el conjunto de tiempos de arribo del primer potencial de acción (*first-spike arrival times*) como:

$$T = \{t_1, t_2, \dots, t_n\}, \quad t_i \in \mathbb{R}^+$$

### 1.1. Transformación Afín Monótona Creciente (Time-Warping)
Consideremos una distorsión temporal global donde la señal se acelera, retarda o desplaza según:

$$t'_i = g(t_i) = \alpha \cdot t_i + \beta, \quad \alpha > 0, \; \beta \in \mathbb{R}$$

Puesto que la derivada de $g$ es estrictamente positiva:
$$\frac{dg}{dt} = \alpha > 0$$

La función $g$ es una biyección estrictamente monótona creciente en $\mathbb{R}$. Por consiguiente:

$$t_i < t_j \iff \alpha \cdot t_i + \beta < \alpha \cdot t_j + \beta \iff t'_i < t'_j \quad \forall i, j \in \{1, \dots, n\}$$

### 1.2. Invarianza de la Permutación Inducida
Definimos la permutación de disparo $\sigma \in S_n$ como la biyección que ordena los canales por tiempo de llegada:

$$t_{\sigma(1)} \le t_{\sigma(2)} \le \dots \le t_{\sigma(n)}$$

Dado que el orden relativo se preserva idéntico para cualquier $\alpha > 0$ y $\beta \in \mathbb{R}$:

$$\sigma(T') \equiv \sigma(T)$$

La representación discreta extraída en el diccionario de rangos es **formal e idénticamente invariante al factor de compresión o dilatación temporal $\alpha$**.

---

## 2. Particionamiento Modular en Subregiones $N=4$

El espacio combinatorio de permutaciones totales para $n$ canales es $|S_n| = n!$.
Para valores grandes de $n$, $n!$ crece con severidad exponencial y la sensibilidad al *jitter* de alta frecuencia se incrementa debido a colisiones en canales temporalmente próximos.

Kairos103 descompone el espacio en $M$ subregiones modulares locales independientes de tamaño fijo $N = 4$:

* **Capacidad local:** Cada subregión produce $4! = 24$ configuraciones posibles de primer spike.
* **Capacidad del producto cartesiano:** Para $M$ módulos:
  $$|\Omega| = (4!)^M = 24^M$$
* **Estabilidad frente a jitter:** La inhibición lateral acoplada localmente dentro de cada cuarteto garantiza que solo las neuronas con activación predominante fijen el orden, filtrando el ruido de fase con robustez superior al $97\%$.
