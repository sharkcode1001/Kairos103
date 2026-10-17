#!/usr/bin/env python3
"""
Kairos103 Kaggle Runner
=======================
Script autónomo para ejecutar Kairos103 en Kaggle Notebooks:
1. Detecta automáticamente las rutas de train.csv, test.csv y sample_submission.csv
   en /kaggle/input/ (sin importar el nombre exacto de la subcarpeta).
2. Instala el SDK oficial de .NET 10 si no está instalado.
3. Compila Kairos103.Cli en modo Release.
4. Entrena el ensamble multi-semilla (42, 777, 20261010) con OvR en el 100% de datos.
5. Genera /kaggle/working/submission.csv con thresholds calibrados.
6. Valida la integridad del submission generado.
"""

import glob
import os
import shutil
import subprocess
import sys


def banner(title):
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


def find_file(search_roots, patterns, fallback_keywords):
    """
    Busca recursivamente archivos que coincidan exactamente con patterns,
    o en su defecto con palabras clave dentro del nombre.
    """
    candidates = []
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for current_dir, _, files in os.walk(root):
            for f in files:
                candidates.append(os.path.join(current_dir, f))

    # 1. Búsqueda exacta
    for p in candidates:
        name = os.path.basename(p).lower()
        for pat in patterns:
            if name == pat.lower():
                return p

    # 2. Búsqueda por palabras clave
    for p in candidates:
        name = os.path.basename(p).lower()
        if name.endswith(".csv"):
            if all(k in name for k in fallback_keywords):
                return p

    return None


def locate_dataset():
    """Localiza automáticamente train.csv, test.csv y sample_submission.csv."""
    search_dirs = [
        "/kaggle/input",
        "./datasets",
        "../datasets",
        "."
    ]

    print("[*] Buscando archivos del dataset en:", [d for d in search_dirs if os.path.exists(d)])

    train_path = find_file(search_dirs, ["train.csv"], ["train"])
    test_path = find_file(search_dirs, ["test.csv"], ["test"])
    sample_path = find_file(
        search_dirs,
        ["sample_submission.csv", "sample_submission.csv.csv", "sample.csv", "samplesubmission.csv"],
        ["sample"]
    )

    return train_path, test_path, sample_path


def ensure_dotnet():
    """Verifica e instala .NET 10 SDK si es necesario."""
    # Revisar si dotnet ya existe en PATH o en ~/.dotnet
    dotnet_candidates = [
        shutil.which("dotnet"),
        os.path.expanduser("~/.dotnet/dotnet"),
        "/root/.dotnet/dotnet",
        "/home/kaggle/.dotnet/dotnet"
    ]

    dotnet_bin = None
    for cand in dotnet_candidates:
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            dotnet_bin = cand
            break

    # Si no existe, instalar .NET 10 con el script oficial de Microsoft
    if not dotnet_bin:
        banner("[1/4] Instalando .NET 10 SDK (script oficial de Microsoft)...")
        install_cmd = "curl -sSL https://dot.net/v1/dotnet-install.sh | bash -s -- --channel 10.0"
        ret = subprocess.call(install_cmd, shell=True)
        if ret != 0:
            raise RuntimeError(f"Error al instalar .NET SDK (codigo {ret})")

        dotnet_bin = os.path.expanduser("~/.dotnet/dotnet")
        if not os.path.isfile(dotnet_bin):
            dotnet_bin = "/root/.dotnet/dotnet"

    # Exportar variables de entorno para que los subprocesos reconozcan dotnet
    dotnet_dir = os.path.dirname(dotnet_bin)
    os.environ["PATH"] = f"{dotnet_dir}:{os.environ.get('PATH', '')}"
    os.environ["DOTNET_ROOT"] = dotnet_dir
    os.environ["DOTNET_CLI_TELEMETRY_OPTOUT"] = "1"

    # Verificar version
    version = subprocess.check_output([dotnet_bin, "--version"]).decode().strip()
    print(f"[*] .NET SDK detectado: version {version} ({dotnet_bin})")
    return dotnet_bin


def main():
    banner("KAIROS103 - RUNNER AUTOMATICO PARA KAGGLE (.NET 10)")

    # 1. Asegurar .NET 10
    dotnet_bin = ensure_dotnet()

    # 2. Localizar archivos del dataset
    train_path, test_path, sample_path = locate_dataset()

    print("\nResultados de la busqueda de datos:")
    print(f"  Train:             {train_path}")
    print(f"  Test:              {test_path}")
    print(f"  Sample Submission: {sample_path}")

    if not train_path or not os.path.isfile(train_path):
        raise FileNotFoundError(
            "No se encontro 'train.csv' en /kaggle/input/. "
            "Asegurate de haber adjuntado el Dataset de la competicion al Notebook en Kaggle."
        )

    if not test_path or not os.path.isfile(test_path):
        raise FileNotFoundError(
            "No se encontro 'test.csv' en /kaggle/input/. "
            "Asegurate de haber adjuntado el Dataset de la competicion al Notebook en Kaggle."
        )

    # 3. Determinar ruta de salida
    if os.path.exists("/kaggle/working"):
        output_csv = "/kaggle/working/submission.csv"
        model_dir = "/kaggle/working/results"
    else:
        output_csv = "submission.csv"
        model_dir = "results"

    os.makedirs(model_dir, exist_ok=True)

    # 4. Verificar ubicacion del proyecto Kairos103
    cli_proj = "src/Kairos103.Cli/Kairos103.Cli.csproj"
    if not os.path.isfile(cli_proj):
        if os.path.isfile(f"Kairos103/{cli_proj}"):
            os.chdir("Kairos103")
        else:
            raise FileNotFoundError(f"No se encontro '{cli_proj}'. Ejecuta este script desde la raiz del repositorio.")

    # 5. Compilar en Release
    banner("[2/4] Compilando Kairos103.Cli en modo Release...")
    build_cmd = [dotnet_bin, "build", cli_proj, "-c", "Release"]
    ret = subprocess.call(build_cmd)
    if ret != 0:
        raise RuntimeError("Error durante la compilacion de Kairos103.Cli")

    # 6. Ejecutar pipeline completo
    banner("[3/4] Ejecutando Pipeline Kaggle (Ensamble 3 semillas + OvR + Full Train)...")
    cmd = [
        dotnet_bin, "run",
        "--project", cli_proj,
        "-c", "Release",
        "--no-build",
        "--",
        "kaggle-pipeline",
        "--train", train_path,
        "--test", test_path,
        "--output", output_csv,
        "--model-dir", model_dir,
        "--seeds", "42",
        "--subregions", "16",
        "--epochs", "28",
        "--cv-folds", "0"
    ]
    if sample_path:
        cmd.extend(["--sample-submission", sample_path])

    print(f"[*] Ejecutando: {' '.join(cmd)}\n")
    sys.stdout.flush()
    ret = subprocess.call(cmd)
    if ret != 0:
        raise RuntimeError(f"El pipeline termino con error (codigo {ret})")

    # 7. Validar submission generado
    banner("[4/4] Verificando submission generado...")
    if not os.path.isfile(output_csv):
        raise FileNotFoundError(f"No se genero el archivo de submission: {output_csv}")

    with open(output_csv, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    row_count = len(lines) - 1
    print(f"  Archivo:        {output_csv}")
    print(f"  Tamano:         {os.path.getsize(output_csv):,} bytes")
    print(f"  Total de filas: {row_count:,}")

    print("\nPrimeras 5 lineas de submission.csv:")
    for l in lines[:6]:
        print(f"    {l}")

    banner("PIPELINE COMPLETADO CON EXITO")
    print(f"El archivo '{output_csv}' esta listo para enviar en Kaggle.")


if __name__ == "__main__":
    main()
