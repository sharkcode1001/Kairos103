# Kairos103: Clockless Rank-Order Neural Architecture with Strict Time-Warp Invariance

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform: .NET 10](https://img.shields.io/badge/.NET-10.0-purple.svg)](https://dotnet.microsoft.com/)
[![Application: Console CLI](https://img.shields.io/badge/Interface-CLI%20First%20(Scientific)-black.svg)](#)
[![Focus: Neuromorphic Computing](https://img.shields.io/badge/Focus-Event--Driven%20%7C%20Rank--Order-green.svg)](#)

---

## 1. Visión General y Filosofía

**Kairos103** es una arquitectura neuronal asíncrona (*clockless*) de ultra-bajo consumo desarrollada en **C# / .NET 10**, concebida bajo una filosofía **CLI-First / Headless**: un motor modular orientado a consola científica para garantizar reproducibilidad total, cero contaminación de memoria por interfaces gráficas y máxima precisión en el benchmarking.

En la filosofía griega clásica:
* **Chronos** ($\chi\rho\acute{o}\nu o\varsigma$) representa el tiempo métrico, secuencial y medido por un reloj rígido ($\Delta t$).
* **Kairos** ($\kappa\alpha\iota\rho\acute{o}\varsigma$) representa el momento cualitativo: la relación, la oportunidad y el **orden relativo** en que suceden los eventos.

Las arquitecturas contemporáneas dominantes —especialmente los **Transformers** (con codificaciones posicionales fijas $PE$) y las **RNNs/LSTMs**— dependen exclusivamente de un marco *Chronos*: discretizan la realidad en marcos o pasos de tiempo homogéneos $\Delta t$. Cuando una señal se acelera, se ralentiza o sufre fluctuaciones temporales no uniformes (*time-warping*), estas redes colapsan o exigen costosos reentrenamientos.

**Kairos103** opera bajo el paradigma *Kairos*: prescinde de relojes globales y coordenadas temporales absolutas. La información reside en las permutaciones de disparo relativo ($n!$) entre subregiones neuronales locales, logrando **invarianza matemática estricta a dilataciones temporales** y ejecutando inferencias mediante eventos dispersos sin multiplicaciones densas de punto flotante.

---

## 2. Decisión Arquitectónica: ¿Por qué CLI-First (Consola)?

Para validar científicamente una arquitectura frente a baselines consolidados como Transformers y LSTMs, se descarta una aplicación GUI monolítica de inicio en favor de una arquitectura en capas basada en **Consola Científica**:

1. **Aislamiento Métrico Riguroso (Zero UI Overhead):**
   * Medir microsegundos por inferencia o asignación de memoria en el heap (`GC.GetAllocatedBytesForCurrentThread()`) en WPF o WinForms introduce ruido estadístico proveniente del dispatcher, bucles de renderizado y bindings XAML.
   * La consola asegura que el 100% de los ciclos de CPU y bytes medidos corresponden exclusivamente a la red neuronal.
2. **Automatización de Barridos Masivos (Sweeps & CI):**
   * Facilita la ejecución automatizada de experimentos reproducibles mediante argumentos de línea de comandos (ej. 10 semillas aleatorias $\times$ 7 factores de escala temporal $\times$ 5 niveles de ruido).
3. **Generación Automatizada de Reportes y Gráficos:**
   * El CLI genera automáticamente matrices de confusión, curvas comparativas (*Accuracy vs. Time-Warp Factor*) y tablas Markdown listas para publicación técnica o repositorio.
4. **Desacoplamiento Futuro:**
   * Al mantener la lógica pura en `Kairos103.Core`, una interfaz gráfica visualizadora (WPF / Avalonia) podrá añadirse en el futuro como un consumidor opcional sin alterar el motor de cálculo.

---

## 3. El Problema Fundamental de los Transformers en Secuencias Temporales

1. **Dependencia de Marcos Rígidos de Muestreo ($\Delta t$):**
   Los Transformers convierten el tiempo en espacio proyectando cada instante a un vector continuo y sumando un vector posicional dependiente del índice $t$. Si una señal se emite al doble de velocidad ($2\times$) o a la mitad ($0.5\times$), los índices cambian drásticamente y la atención ($QK^T / \sqrt{d_k}$) se desalinea.
2. **Cómputo Cuadrático y Consumo Ineficiente:**
   La auto-atención densa calcula relaciones $O(T^2)$ sobre toda la longitud de la ventana, recalculando la atención incluso en periodos de señal estática donde no hay información nueva.
3. **Inviabilidad en Edge y Microcontroladores:**
   El footprint de memoria de la atención y la tabla de embeddings impide su ejecución en microcontroladores y sensores autónomos de baja potencia.

---

## 4. Hipótesis Científica y Lo Que Demostramos

### 4.1. Teorema de Invarianza por Escalamiento Temporal
Sea un conjunto de $n$ canales neuronales que emiten sus primeros pulsos (*first-spike arrival times*) en los instantes:
$$T = \{t_1, t_2, \dots, t_n\}$$

Si la señal de entrada sufre una transformación temporal afín monótona creciente (aceleración, desaceleración o desplazamiento temporal):
$$t'_i = \alpha \cdot t_i + \beta, \quad \text{con } \alpha > 0, \; \beta \in \mathbb{R}$$

Entonces, el orden relativo de llegada inducido $\sigma \in S_n$ satisface:
$$t_i < t_j \iff t'_i < t'_j \quad \forall i, j$$

$$\sigma(T') \equiv \sigma(T)$$

**Consecuencia:** La representación interna del diccionario de rangos de Kairos103 es **100% idéntica** bajo cualquier factor de dilatación o compresión temporal $\alpha$, sin necesidad de normalización ni reescalado dinámico.

### 4.2. Objetivos Empíricos Concretos a Demostrar
1. **Invarianza al Time-Warping ($0.5\times$ a $2.0\times$) en Épocas Acotadas:**
   * En tareas de clasificación de secuencias temporales acotadas (ráfagas sensoriales, vibración o complejos ECG delimitados por su intervalo de activación), Kairos103 mantiene una degradación inferior al $2\%$ en precisión sin reentrenamiento ni aumento de datos, mientras que baselines con codificación posicional rígida como Transformers y LSTMs sufren caídas superiores al $20\%-30\%$.
2. **Eficiencia Computacional Discreta y Desglose de MACs:**
   * En la etapa de red y readout clasificador, la inferencia se resuelve exclusivamente mediante sumas acumuladas sobre eventos dispersos ($\sim 10^4$ adiciones discretas sin multiplicaciones densas).
   * Al contabilizar el pipeline completo (incluyendo los filtros analógicos/digitales continuos del front-end sensorial), Kairos103 reduce el cómputo total en un orden de magnitud ($\sim 14.6\times$ menos operaciones MAC/FLOPs globales frente a la auto-atención densa cuadrática $O(T^2 \cdot D)$ de un Transformer equivalente).
3. **Tolerancia al Jitter mediante Subregiones Modulares:**
   * La segmentación en módulos locales de $N=4$ neuronas con inhibición lateral adaptativa filtra el ruido de fase de alta frecuencia, conservando una estabilidad de código superior al $97\%$.

---

## 5. Arquitectura del Flujo de Procesamiento

```
 Señal Continua / Flujo de Eventos (Sensorial o Sintético)
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────┐
 │ 1. Front-End: Filtros Temporales Locales                │
 │    (Estimulación por ráfagas / Cruce adaptativo)        │
 └──────────────────────────┬──────────────────────────────┘
                            │ Pulsos asíncronos
                            ▼
 ┌─────────────────────────────────────────────────────────┐
 │ 2. Subregiones Modulares N=4                            │
 │    - Neuronas LIF con disparo por compromiso            │
 │    - Inhibición lateral por salto de umbral             │
 └──────────────────────────┬──────────────────────────────┘
                            │ Primeros spikes (t1 < t2 < t3 < t4)
                            ▼
 ┌─────────────────────────────────────────────────────────┐
 │ 3. Extractor de Permutaciones (Rank Dictionary)         │
 │    - Vector de rangos normalizados (1/rango)            │
 │    - Capacidad combinatoria local: 4! = 24              │
 └──────────────────────────┬──────────────────────────────┘
                            │ Vector disperso de características
                            ▼
 ┌─────────────────────────────────────────────────────────┐
 │ 4. Capa de Coincidencias Jerárquicas & Readout          │
 │    - Detección de co-ocurrencias relativas              │
 │    - Clasificador supervisado lineal/disperso           │
 └─────────────────────────────────────────────────────────┘
```

---

## 6. Estructura de la Solución (.NET 10 Solution)

```text
Kairos103/
├── src/
│   ├── Kairos103.Core/               # Class Library: Motor neuronal puro agnóstico a UI
│   │   ├── Neurons/
│   │   │   ├── LifNeuron.cs          # Leaky Integrate-and-Fire con umbral dinámico
│   │   │   └── ModularSubregion.cs   # Unidad de N=4 neuronas con inhibición lateral
│   │   ├── Encoding/
│   │   │   ├── RankOrderEncoder.cs   # Mapeo de tiempos a vectores de rango (1/rango)
│   │   │   └── CombinatorialTable.cs # Mapeo combinatorio inyectivo B103
│   │   └── Models/
│   │       ├── KairosNetwork.cs      # Grafo modular asíncrono
│   │       └── PairwiseTrainer.cs    # Regla de aprendizaje supervisado por pares
│   │
│   ├── Kairos103.Benchmarks/         # Class Library: Protocolos de comparación y baselines
│   │   ├── Baselines/
│   │   │   ├── MiniTransformer.cs    # Baseline de atención multi-cabezal con PE
│   │   │   └── SimpleLstm.cs         # Baseline recurrente clásico
│   │   ├── Generators/
│   │   │   ├── SyntheticSignals.cs   # Generador de patrones con dilatación temporal (0.5x - 2.0x)
│   │   │   └── NoiseInjector.cs      # Inyección de jitter temporal y deriva
│   │   └── Metrics/
│   │       ├── EfficiencyTracker.cs  # Contador de Spikes vs. MACs y asignación de memoria
│   │       └── RobustnessMetrics.cs  # Curvas de degradación de precisión vs. time-warp
│   │
│   └── Kairos103.Cli/                # Console App: Punto de entrada principal para investigación
│       ├── Program.cs                # Enrutador de comandos (train, benchmark, sweep, export)
│       ├── Commands/
│       │   ├── RunBenchmarkCommand.cs
│       │   ├── SweepNoiseCommand.cs
│       │   └── ExportReportCommand.cs
│       └── Reporting/
│           └── AsciiPlotter.cs       # Renderizado de gráficos en terminal y exportación SVG/MD
│
├── tests/
│   └── Kairos103.Tests/              # xUnit: Pruebas unitarias matemáticas y de inyectividad
│       ├── InvarianceTests.cs        # Verificación formal: sigma(t) == sigma(alpha * t + beta)
│       └── CombinatorialTests.cs     # Verificación de límites B103: sum(C(N,k)*2^k) >= 2^R
│
├── docs/
│   ├── MathTheory.md                 # Demostración formal de los espacios de permutación
│   └── ExperimentalProtocol.md       # Protocolo para reproducir los benchmarks paso a paso
│
├── Kairos103.slnx
├── README.md
└── LICENSE
```

---

## 7. Interfaz de Línea de Comandos (CLI Usage)

La aplicación de consola ofrece comandos directos para reproducibilidad en terminal:

```bash
# 1. Ejecutar benchmark comparativo rápido (Kairos103 vs Transformer vs LSTM)
dotnet run --project src/Kairos103.Cli -- benchmark --fast

# 2. Ejecutar benchmark completo sobre todos los datos y factores
dotnet run --project src/Kairos103.Cli -- benchmark --warp-factors 0.5,0.8,1.0,1.2,2.0 --dataset synthetic

# 3. Ejecutar prueba de tolerancia a Jitter temporal
dotnet run --project src/Kairos103.Cli -- jitter-sweep --max-jitter-ms 50 --step 10

# 4. Exportar reporte técnico y figuras vectoriales (Markdown + SVG en ./results)
dotnet run --project src/Kairos103.Cli -- export-plots --output ./results/
```


---

## 8. Protocolo Experimental para Reproducibilidad

Todo el experimento está concebido para ejecutarse en cualquier ordenador portátil con el SDK de .NET instalado:

1. **Fase 1: Entrenamiento Nominal ($1.0\times$)**
   * Se entrena Kairos103 y los baselines (Mini-Transformer y LSTM) sobre el dataset a velocidad de muestreo nominal.
2. **Fase 2: Barrido de Dilatación Temporal (*Time-Warp Stress Test*)**
   * Se evalúa la inferencia aplicando factores de escala temporal $\alpha \in \{0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0\}$.
   * Se evalúa con adición progresiva de ruido gaussiano temporal (*jitter* $\sigma_j \in [0, 50\,\text{ms}]$).
3. **Fase 3: Medición de Huella Energética y Computacional**
   * Recuento estricto de eventos: número total de operaciones de suma por spike frente a operaciones flotantes (FLOPs/MACs) de los modelos continuos.
   * Asignación de memoria en el heap durante inferencia en régimen estacionario (`GC.GetAllocatedBytesForCurrentThread()`).

---

## 9. Referencias y Fundamentos

* **Thorpe, S., Fize, D., & Marlot, C. (1996).** *Speed of processing in the human visual system.* Nature. (Fundamentos de la hipótesis de primer spike).
* **Lora Sugve, E. (2026).** *Learning to Use Spike Order: Rank-Order Dictionaries Beat Binary Features by 17 Points at a Fraction of the Events.* SNN103 Technical Preprint.
* **B103 Theory (2026):** *Combinatorial State-Space Mapping with Semantic Activity Encodings.*

---

## 10. Licencia

Este proyecto está distribuido bajo la licencia **MIT**. Consulte el archivo `LICENSE` para más detalles.
