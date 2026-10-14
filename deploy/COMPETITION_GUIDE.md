# Kairos103: Guía de Competición Externa y Evaluación Abierta

Este documento detalla cómo llevar **Kairos103** a competir fuera de tu ordenador local en plataformas públicas y auditadas contra modelos del estado del arte (Transformers, LSTMs, ResNets y Spiking Neural Networks).

---

## 1. Dónde Recomiendo Probarlo Primero: Kaggle MIT-BIH Challenge

La prueba más accesible, inmediata y de mayor impacto público es el benchmark de arritmias cardíacas en Kaggle:

* **Enlace del Dataset:** [Kaggle: ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
* **Datos:** Basado directamente en la base de datos clínica **MIT-BIH** (109,446 latidos segmentados a 125 Hz).
* **Referencia de la comunidad:** Más de 200 notebooks públicos compitiendo con CNNs, ResNets, LSTMs y Transformers densos (la media de precisión SOTA se sitúa entre el **97.5% y el 98.4%**).

### Protocolo de Ejecución en Kaggle:
1. **Descarga los datos:**
   * `mitbih_train.csv` (87,554 muestras de entrenamiento)
   * `mitbih_test.csv` (21,892 muestras de prueba ciega)

2. **Entrena Kairos103 con la CLI:**
   ```bash
   dotnet run --project src/Kairos103.Cli -- train \
       --data mitbih_train.csv \
       --output model_mitbih.kairos \
       --epochs 25 \
       --subregions 16
   ```

3. **Genera el archivo de predicciones de competición (`submission.csv`):**
   ```bash
   python deploy/kaggle_evaluator.py \
       --model model_mitbih.kairos \
       --test mitbih_test.csv \
       --output submission.csv
   ```

4. **Publica tu Notebook en Kaggle:**
   * Crea un Notebook público titulado: *"Kairos103: Clockless Rank-Order SNN for Time-Warp Invariant ECG Classification"*.
   * Adjunta tu binario compilado o código fuente y muestra la tabla comparativa: **Rendimiento competitivo frente a arquitecturas continuas (ResNet / LSTM) preservando invarianza temporal estricta y evitando la auto-atención densa $O(T^2)$**.

---

## 2. Competición Oficial en Silicio y Edge AI: MLPerf Tiny (MLCommons)

Para certificar cuantitativamente la eficiencia energética y el presupuesto de operaciones de Kairos103:

* **Plataforma:** [MLCommons MLPerf Tiny Inference Benchmark](https://mlcommons.org/benchmarks/inference-tiny/)
* **Qué audita:**
  * Número real de operaciones aritméticas por decisión.
  * Tamaño del binario y memoria requerida en dispositivos empotrados (ARM Cortex-M / RISC-V).
  * Latencia de inferencia y consumo en microjulios ($\mu\text{J}$).
* **Categorías:** *Anomaly Detection* (detección de anomalías en vibración 1D) y *Sensory Classification*.
* **Cómo competir:**
  * Se compila el ejecutable para Linux o la arquitectura objetivo:
    ```bash
    dotnet publish src/Kairos103.Cli -c Release -r linux-x64 --self-contained true
    ```
  * Se ejecuta el arnés de medición de MLCommons sobre el binario exportado.

---

## 3. Desafíos Médicos Rigurosos: PhysioNet Challenges (CinC)

* **Plataforma:** [PhysioNet / Computing in Cardiology (CinC) Challenges](https://physionet.org/about/challenge/)
* **Qué es:** La competición científica más prestigiosa en cardiología computacional.
* **Mecanismo:**
  * Cada desafío anual define un esquema de entrada/salida y un conjunto de prueba completamente ciego.
  * Adaptando los scripts de inferencia al formato exigido por la edición, subes tu contenedor Docker:
    ```bash
    docker build -t kairos103 -f deploy/Dockerfile .
    ```
  * El sistema de evaluación ejecuta el contenedor de forma autónoma y publica el algoritmo en el ranking internacional oficial.

---

## 4. Demostración Web Pública: Hugging Face Space Interactivo

Para que cualquier persona o evaluador pueda probar Kairos103 directamente desde su navegador sin instalar nada:

1. Crea una cuenta en [Hugging Face](https://huggingface.co).
2. Crea un **Space** gratuito con SDK **Docker**.
3. Sube el código de Kairos103 y el `deploy/Dockerfile`.
4. El Space ofrecerá una interfaz visual con:
   * Gráfica interactiva de un latido ECG.
   * Un control deslizante de velocidad temporal: `[ 0.5x ----- 1.0x ----- 2.0x ]`.
   * Visualización en tiempo real:
     * **Transformer convencional:** Se degrada al variar la escala de tiempo.
     * **Kairos103:** Mantiene la detección correcta con disparos asíncronos y readout lineal disperso.

---

## 5. Herramientas Preparadas en este Directorio

| Archivo | Función |
| :--- | :--- |
| [`Dockerfile`](Dockerfile) | Contenedor multi-stage universal para compilar y ejecutar Kairos en cualquier entorno Linux. |
| [`.dockerignore`](.dockerignore) | Filtro de archivos para compilaciones Docker rápidas y limpias. |
| [`kaggle_evaluator.py`](kaggle_evaluator.py) | Script de Python (sin dependencias externas) para parsear datasets y generar `submission.csv`. |
| [`COMPETITION_GUIDE.md`](COMPETITION_GUIDE.md) | Guía detallada con enlaces y protocolos para cada plataforma externa. |
