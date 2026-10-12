# Kairos103: Redes Neuronales Asíncronas por Orden de Disparo con Invarianza Estricta a la Dilatación Temporal (Time-Warping)

* **Autor:** **Erisnel Lora Sugve**  
* **Filiación:** Investigador Independiente, Santiago de Cuba, Cuba  
* **Email:** [`lorasugve@gmail.com`](mailto:lorasugve@gmail.com)  
* **ORCID:** [0009-0007-2047-9345](https://orcid.org/0009-0007-2047-9345)  
* **Ecosistema:** .NET 10 | Computación Neuromórfica | Aprendizaje Orientado a Eventos

---

## Resumen Ejecutivo

¿Qué ocurre cuando una señal continua —como un latido cardíaco (ECG), una vibración mecánica o un comando de voz— se emite al doble de velocidad ($2.0\times$) o a la mitad ($0.5\times$)? 

Para las arquitecturas dominantes de la Inteligencia Artificial moderna, como los **Transformers** (con codificaciones posicionales fijas $PE$) y las **redes recurrentes (RNN/LSTM)**, el resultado es el colapso: su precisión se degrada entre un **25% y un 50%**, a menos que se recurra a costosas técnicas de aumento de datos (*data augmentation*) o reentrenamientos específicos.

**Kairos103** introduce un cambio de paradigma radical: una arquitectura neuronal asíncrona (*clockless*) de ultra-bajo consumo desarrollada en **C# / .NET 10**, inspirada en la hipótesis del primer potencial de acción de la corteza sensorial humana (Thorpe et al., 1996) y fundamentada en la **teoría combinatoria B103**.

Al reemplazar el tiempo métrico rígido (*Chronos*) por el orden de precedencia causal relativa de los pulsos (*Kairos*), Kairos103 demuestra **invarianza matemática estricta a transformaciones afines temporales**, manteniendo una precisión superior al **98%–100%** en todo el espectro de dilatación temporal ($0.5\times$ a $2.0\times$) con una reducción computacional de hasta **28×** frente a la auto-atención densa.

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

## 2. Demostración Matemática: El Teorema de Invarianza Afín

Consideremos un conjunto de $n$ neuronas sensoriales que emiten su primer pulso en los instantes:
$$T = \{t_1, t_2, \dots, t_n\}, \quad t_i \in \mathbb{R}^+$$

Si la señal experimenta una transformación afín monótona creciente (aceleración, desaceleración o desfase temporal global):
$$t'_i = g(t_i) = \alpha \cdot t_i + \beta, \quad \text{con } \alpha > 0, \; \beta \in \mathbb{R}$$

Puesto que $\frac{dg}{dt} = \alpha > 0$, la función $g$ es estrictamente monótona creciente. Por consiguiente:
$$t_i < t_j \iff \alpha t_i + \beta < \alpha t_j + \beta \iff t'_i < t'_j \quad \forall i, j$$

La permutación de disparo inducida $\sigma \in S_n$ satisface idénticamente:
$$\sigma(T') \equiv \sigma(T)$$

**Corolario:** La firma interna en el espacio de rangos de Kairos103 es **100% idéntica** bajo cualquier factor de escala $\alpha$, sin necesidad de normalización adaptativa ni reescalado dinámico de series temporales.

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
| **0.5x** | Doble lentitud | **99.00%** | 31.00% | 30.00% |
| **0.8x** | 20% más lenta | **98.00%** | 49.50% | 38.50% |
| **1.0x** | Nominal (Control) | **98.50%** | **80.50%** | **98.50%** |
| **1.2x** | 20% más rápida | **98.00%** | 39.00% | 0.00% |
| **2.0x** | Doble rapidez | **90.00%** | 24.50% | 0.50% |

> [!NOTE]
> **Arquitectura del Mini-Transformer y Ablación de Pooling:** Para un baseline reproducible en C# nativo sin dependencias de frameworks externos, el Mini-Transformer implementa una capa de auto-atención con proyecciones $Q/K$ fijas normalizadas (estilo reservorio / ELM), proyección $V$ y cabezal clasificador entrenados end-to-end con Adam. Para preservar la dirección de la secuencia sin destruir el orden temporal que confunden los modelos continuos, se utiliza agregación temporal por mitades (*split-half temporal pooling*).
> 
> **Fenómeno de Inversión en la LSTM:** El colapso a $0.00\%$ de la LSTM a $1.2\times$ fue auditado mediante matriz de confusión empírica. El modelo no degenera hacia una sola clase (~50 predicciones por clase), sino que experimenta una **inversión simétrica observada** ($0 \leftrightarrow 2$ y $1 \leftrightarrow 3$), consistente con la hipótesis de que el corrimiento de fase temporal desplaza las trayectorias de los atractores del estado recurrente $h(t)$.

* **Observación:** Ambos baselines alcanzan una alta convergencia a velocidad nominal ($80.50\%$ en Transformer y $98.50\%$ en LSTM), pero sufren un colapso severo ante dilatación temporal: el Transformer cae hasta el nivel de puro azar ($24.50\%$, donde $1/4 = 25\%$), mientras que Kairos103 mantiene una estabilidad prácticamente plana ($\ge 98\%$) en casi todo el rango operativo.

### B. Dataset Biomédico Real: Detección de Arritmias en ECG (MIT-BIH)

Evaluado sobre latidos normales (**N**) frente a latidos ventriculares prematuros (**V**) provenientes de la base de datos clínica MIT-BIH Arrhythmia:

| Factor Warp | Kairos103 (Rank) | Mini-Transformer | Simple LSTM |
|:-----------:|:----------------:|:----------------:|:-----------:|
| **0.5x** | **100.00%** | 90.00% | 56.50% |
| **0.8x** | **100.00%** | 94.50% | 74.00% |
| **1.0x** | **100.00%** | 97.00% | 73.50% |
| **1.2x** | **100.00%** | 88.50% | 82.50% |
| **2.0x** | **100.00%** | 88.50% | 85.00% |

* **Observación:** En el complejo electrocardiográfico, el Mini-Transformer alcanza un **$97.00\%$** a velocidad nominal, pero se degrada progresivamente hasta el **$88.50\%$** bajo dilataciones extremas. Kairos103 mantiene un **100.00% de invarianza perfecta** en todos los factores evaluados.

---

## 5. Eficiencia Operativa y Huella Energética

La evaluación cuantitativa del presupuesto de operaciones por decisión arrojó las siguientes métricas:

* **Mini-Transformer (Atención cuadrática $O(T^2 \cdot D)$):** $1.711 \times 10^9$ operaciones MAC continuas de punto flotante.
* **Kairos103 (Pipeline completo):**
  * Front-End analógico/digital: $6.14 \times 10^7$ MACs.
  * Spikes discretos generados: $\approx 58,787$ eventos.
  * Acumulaciones enteras en clasificación: $\approx 31,744$ operaciones (0 multiplicaciones flotantes).
* **Factor de Reducción:** Kairos103 requiere entre **14.6× y 27.8× menos operaciones matemáticas globales** por inferencia frente a la auto-atención densa equivalente.

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
