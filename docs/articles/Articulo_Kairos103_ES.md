# Kairos103: Redes Neuronales Asíncronas por Orden de Disparo con Invarianza Estricta a la Dilatación Temporal (Time-Warping)

* **Autor:** **Erisnel Lora Sugve**  
* **Filiación:** Investigador Independiente, Santiago de Cuba, Cuba  
* **Email:** [`lorasugve@gmail.com`](mailto:lorasugve@gmail.com)  
* **ORCID:** [0009-0007-2047-9345](https://orcid.org/0009-0007-2047-9345)  
* **Ecosistema:** .NET 10 | Computación Neuromórfica | Aprendizaje Orientado a Eventos

> **Idioma / Language:** [English (Inglés)](Article_Kairos103_EN.md) | **Español**

---

## Resumen Ejecutivo

¿Qué ocurre cuando una señal continua —como un latido cardíaco (ECG), una vibración mecánica o un comando de voz— se emite al doble de velocidad ($2.0\times$) o a la mitad ($0.5\times$)? 

Para las arquitecturas dominantes de la Inteligencia Artificial moderna, como los **Transformers** (con codificaciones posicionales fijas $PE$) y las **redes recurrentes (RNN/LSTM)**, el resultado es el colapso: su precisión se degrada entre un **25% y un 50%**, a menos que se recurra a costosas técnicas de aumento de datos (*data augmentation*) o reentrenamientos específicos.

**Kairos103** introduce un cambio de paradigma radical: una arquitectura neuronal asíncrona (*clockless*) de ultra-bajo consumo desarrollada en **C# / .NET 10**, inspirada en la hipótesis del primer potencial de acción de la corteza sensorial humana (Thorpe et al., 1996) y fundamentada en la **teoría combinatoria B103**.

Al reemplazar el tiempo métrico rígido (*Chronos*) por el orden de precedencia causal relativa de los pulsos (*Kairos*), Kairos103 demuestra **invarianza matemática estricta del orden de disparo ante transformaciones afines temporales** y alta estabilidad empírica end-to-end, manteniendo una precisión del **98%–100%** en el espectro temporal moderado ($0.5\times$ a $1.2\times$) y reteniendo un **91.62%** ante estrés extremo ($2.0\times$), con una reducción computacional de hasta **28×** frente a la auto-atención densa de referencia sin necesidad de aumento artificial de datos (*zero time-warp augmentation*).

---

## 1. El Dilema Fundamental: Chronos vs. Kairos

En la filosofía griega clásica coexisten dos concepciones del tiempo:
* **Chronos ($\chi\rho\acute{o}\nu o\varsigma$):** El tiempo métrico, secuencial y uniforme, medido por el tic-tac implacable de un reloj ($\Delta t$).
* **Kairos ($\kappa\alpha\iota\rho\acute{o}\varsigma$):** El momento cualitativo: la oportunidad, la sincronía y el **orden relativo** en que suceden los acontecimientos.

### La Trampa de Chronos en los Modelos Contemporáneos
Los Transformers convierten el tiempo en coordenadas espaciales proyectadas: cada instante de muestreo $t$ recibe un vector posicional estático ($PE_{(t)}$). Si la señal se comprime en el tiempo ($2\times$), las características que antes coincidían en el índice $t=100$ ahora aparecen en $t=50$. La matriz de atención densa ($QK^T / \sqrt{d_k}$) se desalinea completamente respecto a los pesos aprendidos durante el entrenamiento nominal ($1.0\times$).

Por su parte, las LSTMs sufren deriva acumulativa en sus compuertas de olvido e integración ante fluctuaciones en la tasa de muestreo efectivo.

### El Paradigma Kairos
En el cerebro biológico, la corteza visual y auditiva resuelve decisiones complejas en menos de $150\,\text{ms}$, un tiempo insuficiente para promediar tasas de disparo (*rate coding*). La información viaja en el **orden de llegada del primer spike (*first-spike arrival times*)** a través de subpoblaciones neuronales.

Kairos103 formaliza este principio: no importa si el fenómeno duró 50 milisegundos o 200 milisegundos; si la onda $A$ precede a la onda $B$ y esta a la onda $C$, la topología informacional es idéntica.

---

## 2. Demostración Matemática y Distinción Metodológica de Invarianza

Para analizar la robustez temporal con rigor científico, es fundamental separar tres niveles del sistema:

1. **Invarianza Matemática Estricta del Codificador de Orden:** Demostrada formalmente para instantes continuos.
2. **Estabilidad Física del Front-End Sensorial LIF:** Efectos de la discretización digital ($\Delta t$), recorte de ventanas y dinámicas de fuga.
3. **Precisión Empírica End-to-End:** Rendimiento final del clasificador lineal sobre datos reales y sintéticos.

### 2.1. El Teorema de Invarianza Afín del Orden de Disparo
Consideremos un conjunto de $n$ neuronas sensoriales que emiten su primer pulso en los instantes:
$$T = \{t_1, t_2, \dots, t_n\}, \quad t_i \in \mathbb{R}^+$$

Si la señal experimenta una transformación afín monótona creciente (aceleración, desaceleración o desfase temporal global):
$$t'_i = g(t_i) = \alpha \cdot t_i + \beta, \quad \text{con } \alpha > 0, \; \beta \in \mathbb{R}$$

Puesto que $\frac{dg}{dt} = \alpha > 0$, la función $g$ es estrictamente monótona creciente. Por consiguiente:
$$t_i < t_j \iff \alpha t_i + \beta < \alpha t_j + \beta \iff t'_i < t'_j \quad \forall i, j$$

La permutación de disparo inducida $\sigma \in S_n$ satisface idénticamente:
$$\sigma(T') \equiv \sigma(T)$$

**Corolario:** En el espacio abstracto de tiempos de llegada continuos, la firma de permutaciones de Kairos103 es **100% invariante por construcción**.

### 2.2. Front-End Sensorial y Realidades de la Discretización
En series temporales digitales del mundo real, la señal no es continua: se muestrea en una cuadrícula fija $\Delta t$. Además, las neuronas LIF integran corriente con una constante de fuga pasiva ($\tau_{\text{leak}}$). 
Bajo compresiones extremas ($\alpha = 2.0\times$), si la frecuencia efectiva sube significativamente respecto a $\tau_{\text{leak}}$ y la ventana finita corta información de borde, el orden de disparo puede experimentar pequeñas perturbaciones. Esta distinción explica con total transparencia por qué en el dataset sintético a $2.0\times$ la precisión se sitúa en **$91.62\%$** en lugar de un teórico $100\%$, reteniendo aún así una ventaja contundente frente al colapso de los modelos de base.

---

## 3. Arquitectura del Sistema: Módulos $N=4$ y Proyección Biyectiva

El espacio total de permutaciones para $n$ neuronas es $n!$. Un valor elevado de $n$ provocaría una explosión combinatoria inmanejable y alta sensibilidad al ruido (*jitter*) entre canales casi simultáneos.

Para resolverlo, Kairos103 estructura el procesamiento en **subregiones modulares independientes de $N=4$ neuronas** acopladas con inhibición lateral:

1. **Front-End de Fase Normalizada:** Transduce el soporte de la señal a una coordenada de fase relativa $\tau \in [0, 1]$, proyectando ventanas multiescala sobre filtros aleatorios normalizados L2.
2. **Neuronas LIF con Disparo Único e Inhibición Cruzada:** Cuando la neurona con mayor afinidad alcanza su umbral, emite su spike y eleva transitoriamente el umbral de las 3 neuronas restantes del cuarteto. Esto congela la secuencia y filtra fluctuaciones espurias de alta frecuencia.
3. **Mapeo Combinatorio Inyectivo B103 (Código de Lehmer):**
   Cada cuarteto genera exactamente $4! = 24$ permutaciones posibles. Mediante aritmética factorial, cada permutación se convierte de forma biyectiva en un índice entero $k \in [0, 23]$:
   $$\text{Índice} = \sum_{i=0}^{n-1} L_i \cdot (n - 1 - i)!$$
4. **Readout Lineal por Acumulación Discreta:** El vector de características disperso activa selectivamente pesos sin requerir multiplicaciones flotantes densas en la etapa clasificadora.

```
 Señal Continua / Flujo Sensorial
               │
               ▼
 ┌──────────────────────────┐
 │ Front-End: Fase Relativa │  (Muestreo normalizado multiescala)
 └─────────────┬────────────┘
               │ Corrientes continuas
               ▼
 ┌──────────────────────────┐
 │ Módulos N=4 con LIF      │  (Inhibición lateral por disparo)
 └─────────────┬────────────┘
               │ Primeros spikes (t1 < t2 < t3 < t4)
               ▼
 ┌──────────────────────────┐
 │ Diccionario B103 / Lehmer│  (Mapeo biyectivo: 4! = 24 estados)
 └─────────────┬────────────┘
               │ Vector disperso de características
               ▼
 ┌──────────────────────────┐
 │ Clasificador Lineal      │  (Sumas acumuladas enteras / discretas)
 └──────────────────────────┘
```

---

## 4. Resultados Experimentales

Todos los modelos fueron entrenados **exclusivamente a velocidad nominal ($1.0\times$)**, sin aumento de datos de dilatación temporal (*zero time-warp augmentation*), y fueron sometidos a pruebas de estrés con factores $\alpha \in \{0.5\times, 0.8\times, 1.0\times, 1.2\times, 2.0\times\}$.

### A. Dataset Sintético: Morfología y Secuencia Temporal (WarpStress 1D)

| Factor Warp | Velocidad | Kairos103 (Rank) | Mini-Transformer | Simple LSTM |
|:-----------:|:---------:|:----------------:|:----------------:|:-----------:|
| **0.5x** | Doble lentitud ($2\times$ longitud) | **99.00%** | 45.38% | 75.62% |
| **0.8x** | 20% más lenta | **98.88%** | 63.62% | 97.38% |
| **1.0x** | Nominal (Control) | **98.75%** | **99.88%** | **99.88%** |
| **1.2x** | 20% más rápida | **98.38%** | 45.62% | 99.38% |
| **2.0x** | Doble rapidez ($0.5\times$ longitud) | **91.62%** | 46.38% | 51.75% |

> [!NOTE]
> **Naturaleza Metodológica de los Baselines:** Para garantizar una comparación reproducible y autónoma en C# sin dependencias externas de librerías heterogéneas, el Mini-Transformer y la Simple LSTM se estructuraron como **baselines de referencia canónicos (zero-augmentation)**. Ambos convergen a una precisión nominal sobresaliente ($99.88\%$), demostrando capacidad plena de aprendizaje del espacio de estados. No obstante, al acelerar o desacelerar la señal sin reentrenamiento específico, su rendimiento decae significativamente hacia $45\%–51\%$, evidenciando la fragilidad intrínseca del marco temporal métrico (*Chronos*) frente a la robustez estructural del orden relativo (*Kairos*).

* **Observación:** Kairos103 mantiene una estabilidad prácticamente plana ($\ge 98.3\%$) en el intervalo operativo $0.5\times - 1.2\times$, y preserva un **$91.62\%$** en compresión extrema ($2.0\times$), superando a los baselines continuos por más de 40 puntos porcentuales.

### B. Dataset Biomédico Real: Detección de Arritmias en ECG (MIT-BIH)

Evaluado sobre latidos normales (**N**) frente a latidos ventriculares prematuros (**V**) provenientes de la base de datos clínica MIT-BIH Arrhythmia:

| Factor Warp | Kairos103 (Rank) | Mini-Transformer | Simple LSTM |
|:-----------:|:----------------:|:----------------:|:-----------:|
| **0.5x** | **100.00%** | 92.00% | 66.25% |
| **0.8x** | **99.75%** | 95.25% | 64.00% |
| **1.0x** | **100.00%** | 97.25% | 65.50% |
| **1.2x** | **100.00%** | 84.50% | 62.50% |
| **2.0x** | **100.00%** | 91.25% | 65.50% |

* **Observación:** En el complejo electrocardiográfico, el Mini-Transformer alcanza un **$97.25\%$** a velocidad nominal, pero se degrada progresivamente hasta el **$84.50\% - 91.25\%$** bajo dilataciones. Kairos103 mantiene una retención prácticamente perfecta ($\ge 99.75\%$) en todos los factores evaluados.

---

## 5. Eficiencia Operativa y Huella Energética

La evaluación cuantitativa del presupuesto de operaciones por decisión arrojó las siguientes métricas:

* **Mini-Transformer (Atención cuadrática $O(T^2 \cdot D)$):** $3.42 \times 10^9 - 3.59 \times 10^9$ operaciones MAC continuas de punto flotante en el lote completo.
* **Simple LSTM:** $4.37 \times 10^8 - 6.22 \times 10^8$ MACs continuos.
* **Kairos103 (Pipeline completo):**
  * Front-End analógico/digital: $1.22 \times 10^8 - 2.45 \times 10^8$ MACs continuos.
  * Spikes discretos generados: $\approx 117,881 - 255,994$ eventos.
  * Acumulaciones enteras en clasificación: $\approx 63,476 - 256,000$ operaciones (0 multiplicaciones flotantes en readout).
* **Factor de Reducción:** Kairos103 requiere entre **14.6× y 27.8× menos operaciones matemáticas globales** por inferencia frente a la auto-atención densa equivalente.

> [!TIP]
> **Precisiones sobre Memoria y Telemetría de Potencia:** 
> 1. Las métricas operacionales reflejan **conteos algorítmicos teóricos** de complejidad matemática (MACs vs. adiciones de eventos enteros). La medición directa de potencia física (Joules/Watts) se reserva para etapas posteriores de síntesis en silicio neuromórfico (e.g. Intel Loihi o FPGAs asíncronas).
> 2. Los reportes de memoria asignada en consola corresponden al **churn acumulado del recolector de basura (GC)** durante la ejecución masiva en serie de miles de muestras en el harness CLI de .NET. La memoria requerida por muestra individual en inferencia se mantiene en el orden de pocos kilobytes, apta para microcontroladores sin asignador dinámico de memoria.

---

## 6. Filosofía de Ingeniería: ¿Por qué C# y Consola Científica?

A diferencia de muchos proyectos que dependen de entornos Python heterogéneos con dependencias volátiles de CUDA, Kairos103 fue implementado en **C# puro sobre .NET 10** bajo una filosofía **CLI-First / Headless**:

1. **Aislamiento Métrico Riguroso:** Sin dispatched loops, bindings XAML ni GC overhead de interfaces visuales que contaminen los benchmarks de microsegundos y memoria (`GC.GetAllocatedBytesForCurrentThread()`).
2. **Reproducibilidad Total y Portabilidad:** Ejecutable sin modificaciones en Windows, Linux y macOS mediante un único comando `dotnet run`.
3. **Despliegue Directo a Edge:** Preparado para compilar mediante Native AOT (*Ahead-Of-Time*) para microcontroladores industriales y sensores biomédicos autónomos de ultra-bajo consumo.

---

## 7. Conclusión y Próximos Pasos

Kairos103 demuestra que el paradigma dominante de discretización métrica rígida del tiempo no es la única vía para el procesamiento de secuencias temporales, ni la más eficiente. 

Al rescatar la concepción cualitativa del tiempo (*Kairos*) y combinarla con modelos biológicos de primer disparo y matemáticas combinatorias compactas, es posible construir sistemas de aprendizaje que son **estructuralmente inmunes a las dilataciones temporales**, consumiendo una fracción mínima de energía.

El código fuente completo, los datasets de benchmarking (WarpStress v1 y MIT-BIH) y los protocolos reproducibles están disponibles de forma abierta en GitHub bajo la licencia libre **GNU General Public License v3.0 (GPL-3.0)**.

* **Repositorio Oficial:** [https://github.com/sharkcode1001/Kairos103](https://github.com/sharkcode1001/Kairos103)

---

## 8. Licencia y Contacto para Uso Comercial

Kairos103 opera bajo un esquema de **licenciamiento dual**:
* **Uso Académico y Código Abierto:** Distribuido bajo **GNU GPL v3**. Cualquier trabajo derivado publicado debe mantener la reciprocidad de código abierto.
* **Licenciamiento Comercial y Empresarial:** Para integración en productos propietarios, microcontroladores comerciales, dispositivos médicos cerrados o entornos industriales con exención de *copyleft*, contactar directamente al autor:
  * **Autor:** **Erisnel Lora Sugve**
  * **Email:** [`lorasugve@gmail.com`](mailto:lorasugve@gmail.com)
  * **ORCID:** [0009-0007-2047-9345](https://orcid.org/0009-0007-2047-9345)
  * **Ubicación:** Santiago de Cuba, Cuba
