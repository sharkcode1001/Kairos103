# Receta del mejor modelo observado

La captura muestra como mejor resultado visible `submission(30).csv`, con score
`0.58406`, descrito como `MasterEnsemble-8Models-5FoldPlus3FullTrain-TunedThresholds`.

Este repositorio ya implementa sus piezas: red Kairos103, validación cruzada
estratificada, calibración de thresholds y predicción por ensamble. La receta
reproducible está en `deploy/train_best_kaggle.ps1`.

Para el dataset real de la competencia, ubicado en `datasets/nppe2_dataset`,
ejecuta desde la raíz del repositorio:

```powershell
powershell -ExecutionPolicy Bypass -File deploy/train_best_kaggle.ps1 `
  -Train datasets/nppe2_dataset/train.csv `
  -Test datasets/nppe2_dataset/test.csv `
  -SampleSubmission datasets/nppe2_dataset/sample_submission.csv
```

El script produce `results/best_ensemble/submission_best_ensemble.csv` y verifica
automáticamente que tenga el encabezado `id,label`, el número correcto de filas
y etiquetas dentro de `[0,3]`.

Los tres modelos full-train usan semillas distintas (`42`, `123` y `2026`),
mientras que los cinco modelos de fold comparten los thresholds optimizados con
las predicciones out-of-fold. El score de Kaggle no puede garantizarse localmente
porque el conjunto de evaluación es oculto; `0.58406` es el resultado histórico
mostrado en la imagen, no una métrica calculable sin ese ground truth.

El script es reanudable: si se interrumpe, vuelve a ejecutarlo y omitirá los
modelos `.kairos` ya terminados. Al completar las tres semillas generará
`results/best_ensemble/submission_record_ovr_3seed.csv`.
