"""
==============================================================================
KAGGLE ECG ARRHYTHMIA CLASSIFIER (NPPE2 COMPETITION)
1 Modelo LightGBM + 22 Features Electrofisiológicos Auxiliares + Macro F1 Tuning
==============================================================================
Este script está listo para ejecutarse directamente en un Kaggle Notebook nuevo
o en local. No requiere instalar paquetes externos ni GPUs.
"""

import os
import glob
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 74)
print("  KAGGLE ECG ARRHYTHMIA PIPELINE (PYTHON + 22 AUXILIARY FEATURES)")
print("=" * 74)

# ------------------------------------------------------------------------------
# 1. LOCALIZACIÓN AUTOMÁTICA DEL DATASET
# ------------------------------------------------------------------------------
def locate_dataset():
    kaggle_trains = glob.glob('/kaggle/input/**/train.csv', recursive=True)
    if kaggle_trains:
        base_dir = os.path.dirname(kaggle_trains[0])
        print(f"[*] Dataset encontrado en Kaggle: {base_dir}")
        return (
            os.path.join(base_dir, 'train.csv'),
            os.path.join(base_dir, 'test.csv'),
            os.path.join(base_dir, 'sample_submission.csv')
        )
    # Rutas locales de respaldo
    return (
        'datasets/nppe2_dataset/train.csv',
        'datasets/nppe2_dataset/test.csv',
        'datasets/nppe2_dataset/sample_submission.csv'
    )

train_path, test_path, sample_path = locate_dataset()

# ------------------------------------------------------------------------------
# 2. EXTRACCIÓN DE LOS 22 FEATURES ELECTROFISIOLÓGICOS VECTORIZADOS
# ------------------------------------------------------------------------------
def extract_ecg_features(df):
    t0 = time.time()
    
    # Matriz continua de la señal (250 puntos)
    sig_cols = [f'sig_{i}' for i in range(250)]
    signals = df[sig_cols].values.astype(np.float32)
    
    pre_rr = df['pre_rr'].values.astype(np.float32)
    post_rr = df['post_rr'].values.astype(np.float32)
    rr_ratio = df['rr_ratio'].values.astype(np.float32)
    
    # 1. Línea base isoeléctrica
    baseline = (np.mean(signals[:, :16], axis=1) + np.mean(signals[:, -16:], axis=1)) / 2.0
    sig_centered = signals - baseline[:, None]
    
    # 2. Características del ritmo y pausas
    prematurity_idx = (pre_rr - post_rr) / (pre_rr + post_rr + 1e-5)
    comp_pause = (post_rr - pre_rr) / (pre_rr + 1e-5)
    comp_ratio = post_rr / (pre_rr + 1e-5)
    rr_diff = post_rr - pre_rr
    
    # 3. Complejo QRS (muestras 85 a 165)
    qrs_zone = signals[:, 85:165]
    qrs_centered = sig_centered[:, 85:165]
    
    qrs_max = np.max(qrs_zone, axis=1)
    qrs_min = np.min(qrs_zone, axis=1)
    qrs_p2p = qrs_max - qrs_min
    qrs_energy = np.mean(qrs_zone**2, axis=1)
    qrs_abs_max = np.max(np.abs(qrs_centered), axis=1)
    
    # Ancho del QRS a media altura (FWHM) -> Clave para Clase 2 vs Clase 1
    half_peak = 0.5 * qrs_abs_max[:, None]
    qrs_width = np.sum(np.abs(qrs_centered) >= half_peak, axis=1)
    
    # 4. Pendientes de despolarización (maxSlope) -> Clave para Clase 0 vs 2 y 3
    diffs = np.diff(qrs_zone, axis=1)
    max_pos_slope = np.max(diffs, axis=1)
    max_neg_slope = np.max(-diffs, axis=1)
    slope_ratio = max_pos_slope / (max_pos_slope + max_neg_slope + 1e-5)
    
    # 5. Onda T (repolarización, muestras 165 a 225)
    t_zone = signals[:, 165:225]
    t_centered = sig_centered[:, 165:225]
    t_p2p = np.max(t_zone, axis=1) - np.min(t_zone, axis=1)
    t_to_qrs_ratio = t_p2p / (qrs_p2p + 1e-5)
    
    qrs_net = np.mean(qrs_centered, axis=1)
    t_net = np.mean(t_centered, axis=1)
    stt_discordance = - (qrs_net * t_net)
    
    # 6. Onda P (actividad auricular, muestras 25 a 85)
    p_zone = sig_centered[:, 25:85]
    p_energy = np.mean(p_zone**2, axis=1)
    p_max = np.max(np.abs(p_zone), axis=1)
    p_prominence = p_max / (qrs_abs_max + 1e-5)
    
    # 7. Estadísticas globales
    vpp_global = np.max(signals, axis=1) - np.min(signals, axis=1)
    rms_global = np.sqrt(np.mean(signals**2, axis=1))
    
    features = {
        # 1-7: Ritmo y Pausas
        'pre_rr': pre_rr,
        'post_rr': post_rr,
        'rr_ratio': rr_ratio,
        'prematurity_idx': prematurity_idx,
        'comp_pause': comp_pause,
        'comp_ratio': comp_ratio,
        'rr_diff': rr_diff,
        # 8-14: Morfología QRS y Pendientes
        'qrs_width': qrs_width,
        'qrs_p2p': qrs_p2p,
        'qrs_energy': qrs_energy,
        'qrs_abs_max': qrs_abs_max,
        'max_pos_slope': max_pos_slope,
        'max_neg_slope': max_neg_slope,
        'slope_ratio': slope_ratio,
        # 15-19: Repolarización T y Onda P
        't_p2p': t_p2p,
        't_to_qrs_ratio': t_to_qrs_ratio,
        'stt_discordance': stt_discordance,
        'p_energy': p_energy,
        'p_prominence': p_prominence,
        # 20-22: Voltaje global y línea base
        'vpp_global': vpp_global,
        'rms_global': rms_global,
        'baseline': baseline
    }
    
    feat_df = pd.DataFrame(features)
    
    # Muestreo decodificado de la señal (resumido en 25 puntos clave de la onda)
    for i in range(0, 250, 10):
        feat_df[f'sig_sub_{i}'] = signals[:, i]
        
    print(f"[*] Features extraídos en {time.time() - t0:.2f} s. Dimensiones: {feat_df.shape}")
    return feat_df

# ------------------------------------------------------------------------------
# 3. CARGA DE DATASETS Y CÁLCULO DE PESOS
# ------------------------------------------------------------------------------
print(f"[*] Leyendo entrenamiento: {train_path}")
train_df = pd.read_csv(train_path)

print(f"[*] Leyendo prueba: {test_path}")
test_df = pd.read_csv(test_path)

y_train = train_df['label'].values
X_train = extract_ecg_features(train_df)
X_test = extract_ecg_features(test_df)

# Ponderación inversa para compensar desbalance de clases (Clase 3: 597 muestras)
class_counts = np.bincount(y_train)
total_samples = len(y_train)
class_weights = {c: float((total_samples / (4.0 * count))**0.5) for c, count in enumerate(class_counts)}
print(f"[*] Pesos asignados por clase: {class_weights}")

# ------------------------------------------------------------------------------
# 4. ENTRENAMIENTO DE 1 SOLO MODELO LIGHTGBM
# ------------------------------------------------------------------------------
print("\n[*] Entrenando 1 modelo LightGBM...")
params = {
    'objective': 'multiclass',
    'num_class': 4,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 1,
    'class_weight': class_weights,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}

# Validación estratificada para optimizar umbrales
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(X_train, y_train))

X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

train_data = lgb.Dataset(X_tr, label=y_tr)
val_data = lgb.Dataset(X_va, label=y_va, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=600,
    valid_sets=[train_data, val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=40, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)

# ------------------------------------------------------------------------------
# 5. VERIFICACIÓN DEL PESO GANADO POR LOS AUXILIARES (FEATURE IMPORTANCE)
# ------------------------------------------------------------------------------
importance = model.feature_importance(importance_type='gain')
feat_names = model.feature_name()
imp_df = pd.DataFrame({'feature': feat_names, 'importance': importance}).sort_values('importance', ascending=False)

print("\n" + "=" * 74)
print("  TOP 10 FEATURES CON MAYOR PESO GANADO EN EL MODELO")
print("=" * 74)
for rank, (_, row) in enumerate(imp_df.head(10).iterrows(), 1):
    print(f"  {rank:2d}. {row['feature']:<18}: {row['importance']:10.1f} gain")
print("=" * 74)

# ------------------------------------------------------------------------------
# 6. CALIBRACIÓN DE UMBRALES PARA MACRO F1
# ------------------------------------------------------------------------------
print("\n[*] Optimizando umbrales de decisión para Macro F1...")
val_probs = model.predict(X_va, num_iteration=model.best_iteration)

def loss_func(weights):
    w_probs = val_probs * weights
    preds = np.argmax(w_probs, axis=1)
    return -f1_score(y_va, preds, average='macro')

init_weights = [1.0, 1.1, 1.0, 1.2]
opt_res = minimize(loss_func, init_weights, method='Nelder-Mead', options={'maxiter': 250})
best_weights = opt_res.x

val_preds_base = np.argmax(val_probs, axis=1)
val_preds_opt = np.argmax(val_probs * best_weights, axis=1)

f1_base = f1_score(y_va, val_preds_base, average='macro')
f1_opt = f1_score(y_va, val_preds_opt, average='macro')

print(f"  Macro F1 (Argmax estándar):    {f1_base:.4f}")
print(f"  Macro F1 (Umbrales calibrados): {f1_opt:.4f}  <-- (+{f1_opt - f1_base:+.4f} ganancia)")
print(f"  Multiplicadores óptimos:       [{', '.join([f'{w:.3f}' for w in best_weights])}]")

print("\nReporte de Clasificación en Validación:")
print(classification_report(y_va, val_preds_opt, digits=4))

# ------------------------------------------------------------------------------
# 7. GENERACIÓN DEL ARCHIVO DE SUBMISIÓN (SUBMISSION.CSV)
# ------------------------------------------------------------------------------
print("[*] Generando predicciones en test.csv...")
test_probs = model.predict(X_test, num_iteration=model.best_iteration)
final_preds = np.argmax(test_probs * best_weights, axis=1)

output_path = '/kaggle/working/submission.csv' if os.path.exists('/kaggle/working') else 'submission.csv'

submission = pd.DataFrame({
    'id': test_df['id'],
    'label': final_preds
})
submission.to_csv(output_path, index=False)

print(f"\n[OK] ¡Archivo generado con éxito en: {output_path}!")
print(f"     Total de muestras: {len(submission):N0}")
print("\nDistribución de clases predichas en Test:")
print(submission['label'].value_counts().sort_index())

if os.path.exists(sample_path):
    sample_sub = pd.read_csv(sample_path)
    assert len(submission) == len(sample_sub), "Error: Conteo de filas no coincide con sample_submission."
    assert (submission['id'].values == sample_sub['id'].values).all(), "Error: Los IDs no coinciden exactamente."
    print("\n[OK] Verificación contra sample_submission.csv: 100% ALINEADO")
