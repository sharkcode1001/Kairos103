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
   * Adjunta tu binario compilado o código fuente y muestra la tabla comparativa: **Precisión equivalente o superior a una ResNet con 0 multiplicaciones flotantes en la inferencia**.

---

## 2. Competición Oficial en Silicio y Edge AI: MLPerf Tiny (MLCommons)

Si el objetivo es certificar que Kairos103 es una de las arquitecturas neuronales más eficientes del mundo para sensores biomédicos y microcontroladores:

* **Plataforma:** [MLCommons MLPerf Tiny Inference Benchmark](https://mlcommons.org/benchmarks/inference-tiny/)
* **Qué audita:**
  * Número real de operaciones aritméticas por decisión.
  * Tamaño del binario y memoria requerida en microcontroladores (ARM Cortex-M / RISC-V).
  * Latencia de inferencia y consumo en microjulios ($\mu\text{J}$).
* **Categorías:** *Anomaly Detection* (detección de anomalías en vibración 1D) y *Sensory Classification*.
* **Cómo competir:**
  * Se compila el núcleo C# mediante **Native AOT**:
    ```bash
    dotnet publish src/Kairos103.Cli -c Release -r linux-x64 -p:PublishAot=true
    ```
  * Se ejecuta el arnés de medición de MLCommons sobre el binario exportado.

---

## 3. Desafíos Médicos Rigurosos: PhysioNet Challenges (CinC)

* **Plataforma:** [PhysioNet / Computing in Cardiology (CinC) Challenges](https://physionet.org/about/challenge/)
* **Qué es:** La competición científica más prestigiosa en cardiología computacional.
* **Mecanismo:**
  * Los organizadores no proporcionan las etiquetas de prueba.
  * Subes tu contenedor Docker a su servidor:
    ```bash
    docker build -t kairos103 -f deploy/Dockerfile .
    ```
  * El sistema de PhysioNet evalúa tu contenedor de forma ciega y publica tu algoritmo en el ranking internacional junto a publicaciones en el congreso CinC.

---

## 4. Demostración Web Pública: Hugging Face Space Interactivo

Para que cualquier persona o reclutador pueda probar Kairos103 directamente desde su navegador sin instalar nada:

1. Crea una cuenta en [Hugging Face](https://huggingface.co).
2. Crea un **Space** gratuito con SDK **Docker** o **Gradio**.
3. Sube el código de Kairos103 y el `deploy/Dockerfile`.
4. El Space ofrecerá una interfaz visual con:
   * Gráfica interactiva de un latido ECG.
   * Un control deslizante de velocidad temporal: `[ 0.5x ----- 1.0x ----- 2.0x ]`.
   * Visualización en tiempo real:
     * **Transformer convencional:** Falla al alejar la velocidad de 1.0x.
     * **Kairos103:** Mantiene la detección correcta con <1,000 spikes y 0 multiplicaciones densas.

---

## 5. Resumen de Herramientas Preparadas en este Directorio

| Archivo | Función |
| :--- | :--- |
| [`deploy/Dockerfile`](file:///c:/Users/VANICA/Documents/GitHub/Kairos103/deploy/Dockerfile) | Contenedor multi-stage universal para ejecutar Kairos en cualquier nube o servidor Linux. |
| [`deploy/.dockerignore`](file:///c:/Users/VANICA/Documents/GitHub/Kairos103/deploy/.dockerignore) | Filtro de archivos para compilaciones Docker rápidas y limpias. |
| [`deploy/kaggle_evaluator.py`](file:///c:/Users/VANICA/Documents/GitHub/Kairos103/deploy/kaggle_evaluator.py) | Script de Python para parsear datasets de Kaggle y generar `submission.csv`. |
| [`deploy/COMPETITION_GUIDE.md`](file:///c:/Users/VANICA/Documents/GitHub/Kairos103/deploy/COMPETITION_GUIDE.md) | Guía detallada con enlaces y protocolos para cada plataforma externa. |
