# Conversión MIT-BIH a NPPE2

El conversor `mitbih_to_nppe2.py` transforma registros WFDB (`.hea`, `.dat`,
`.atr`) en un CSV con las mismas columnas que `datasets/nppe2_dataset/train.csv`:
250 muestras de señal, `pre_rr`, `post_rr`, `rr_ratio` y `label`.

Instalación de dependencias:

```powershell
py -m pip install -r deploy/requirements-mitbih.txt
```

Conversión:

```powershell
py deploy/mitbih_to_nppe2.py `
  --input datasets/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0 `
  --output datasets/mitbih_nppe2/mitbih_beats.csv `
  --audit datasets/mitbih_nppe2/mitbih_audit.json
```

El script usa el canal `MLII` cuando existe, ventanas de 250 muestras y
normalización min-max por latido. Las etiquetas iniciales son:

```text
Normal: N L R e j
SVEB:   A a J S
VEB:    V E
Fusion: F
```

La salida debe auditarse antes de mezclarla con NPPE2. MIT-BIH debe separarse
por registro/paciente en train y validación para evitar fuga de información.
