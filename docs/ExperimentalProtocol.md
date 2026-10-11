# Protocolo Experimental: Invarianza a Time-Warping (Kairos103 vs. Baselines)

Este documento formaliza las reglas y condiciones científicas de benchmarking aprobadas antes de la ejecución de pruebas.

---

## 1. Variables Experimentales

* **Factores de Escalamiento Temporal en Inferencia ($Test$):**
  $$\alpha \in \{0.5, 0.8, 1.0, 1.2, 2.0\}$$
* **Control de Ruido:**
  * Todos los conjuntos de test incluyen idéntico nivel de *jitter* temporal (perturbación gaussiana preservando monotonicidad temporal) y ruido de amplitud gaussiano ($\sigma_a = 0.05$).
  * La condición $\alpha = 1.0\times$ en test actúa como grupo de control para aislar el efecto del jitter frente a la dilatación de tiempo.
* **Condición de Entrenamiento ($Train$):**
  * Exclusivamente a velocidad nominal ($1.0\times$), sin jitter y con ruido residual leve ($\sigma_a = 0.03$).
  * **Cero aumento de datos con warp:** La invariancia debe emerger de la propia codificación asíncrona de rango temporal (*rank-order*), no de memorización de perturbaciones.

---

## 2. Métricas de Evaluación

Para cada modelo (Kairos103, Transformer, LSTM), para cada factor de velocidad y para cada uno de los dos conjuntos de datos:

1. **Precisión de Clasificación ($Accuracy$ vs. $\alpha$):**
   * Curva de degradación frente a dilatación y compresión.
2. **Costo Computacional por Decisión:**
   * **Baselines continuos (Transformer / LSTM):** Número de multiplicaciones de punto flotante acumuladas (MACs/FLOPs) por inferencia y memoria residente en heap.
   * **Kairos103:** Desglose estricto en dos componentes:
     * *Transducción Sensorial Front-End:* MACs continuos de muestreo local sobre las ventanas temporales.
     * *Red y Clasificador:* Número de eventos discretos (spikes de primer disparo) + operaciones de acumulación entera en lectura (sin multiplicaciones flotantes).
     * *Cómputo Total:* Suma de ambas etapas comparada contra el presupuesto global de atención densa del baseline.

---

## 3. Criterios de Éxito Formal

Para considerar verificada la hipótesis experimental, deben cumplirse **todas** las siguientes condiciones:

1. **Invarianza de Kairos103:** La precisión obtenida se mantiene dentro de un margen de $\pm 3\%$ respecto al valor nominal en $1.0\times$ para los cinco factores temporales, en ambos datasets.
2. **Colapso del Baseline:** Los modelos continuos (Transformer o LSTM) sufren una degradación $\ge 10\%$ en precisión en los extremos $0.5\times$ o $2.0\times$ con respecto a su rendimiento a $1.0\times$.
3. **Eficiencia Energética/Operacional:** Kairos103 ejecuta la inferencia con $\le \frac{1}{10}$ del cómputo por decisión requerido por el baseline.

---

## 4. Reglas de Parada e Integridad Científica

* **Rechazo de Hipótesis:** Si Kairos103 no presenta una curva sustancialmente plana de precisión a través de los factores temporales, se reportará el resultado de forma transparente sin maquillar ni ajustar selectivamente hiperparámetros.
* **Límite de Calidad Nominal:** Si Kairos103 no alcanza un mínimo de $80\%$ de precisión en la condición de control $1.0\times$, se detendrán los barridos de warp hasta refinar el esquema de codificación temporal o la inhibición lateral de subregiones.
