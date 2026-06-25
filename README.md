# STF-KernelSHAP

Repositorio reproducible para los experimentos de explicabilidad EEG con STF-KernelSHAP en tareas de imaginacion motora (MI) y ADHD/TDAH.

El codigo reutilizable vive en `src/stf_kernelshap/` y los cuadernos en `Notebooks/` son runners del flujo experimental.

## Estructura del repositorio

- `src/stf_kernelshap/`: paquete reutilizable con datos, modelos, entrenamiento, evaluacion, XAI, reporting y visualizacion.
- `Notebooks/`: cuadernos reproducibles.
- `Models/`: pesos entrenados y journals de Optuna incluidos mediante Git LFS cuando aplica.
- `Results/`: CSVs, atribuciones, metricas de faithfulness y logs de tiempo.
- `Figures/`: figuras finales.
- `MI_TDAH_Dataset/`: carpeta local opcional para datos, ignorada por Git.
- `Temp/`: salidas de debug, ignoradas por Git.

El repositorio usa Git LFS para artefactos binarios como `.npz`, `.mat`, `.h5`, `.pkl` y `.pdf`.

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
git lfs install
```

## Datos

Los cuadernos resuelven los datos en este orden:

1. `MI_TDAH_Dataset/` en la raiz del repositorio.
2. `Data/` local heredada en la raiz del repositorio.
3. Dataset Kaggle `daprosero/mi-tdah-dataset` mediante `kagglehub`.

Layout esperado:

- `MI_TDAH_Dataset/Data/MI/`
- `MI_TDAH_Dataset/Data/TDAH/`
- o `Data/MI/` y `Data/TDAH/`

Descarga manual opcional:

```python
import kagglehub

path = kagglehub.dataset_download("daprosero/mi-tdah-dataset")
print("Path to dataset files:", path)
```

## Configuracion global de cuadernos

Todos los cuadernos empiezan con una celda bootstrap que:

1. Encuentra la raiz del repositorio.
2. Clona `https://github.com/Daprosero/STF-KernelSHAP.git` si se ejecuta desde un notebook suelto en Colab/Kaggle.
3. Instala `requirements.txt` en Colab/Kaggle.
4. Agrega `src/` a `sys.path`.
5. Llama `setup_notebook_environment(debug=DEBUG)`.

`DEBUG` controla las rutas de salida:

- `DEBUG=False`: escribe en `Results/`, `Figures/` y `Models/`.
- `DEBUG=True`: escribe en `Temp/Results/`, `Temp/Figures/` y `Temp/Models/`.

## Orden recomendado de reproduccion

1. `Notebooks/Train_models.ipynb`
2. `Notebooks/TDAH_MI_Results.ipynb`
3. `Notebooks/run_xai_models_2.ipynb`
4. `Notebooks/Computational_cost_scalability.ipynb`
5. `Notebooks/Faithfulness_summary.ipynb`
6. `Notebooks/Classification_results.ipynb`
7. `Notebooks/Topoplots.ipynb`

## Guia por cuaderno

### `Train_models.ipynb`

Entrena modelos MI y ADHD/TDAH con Optuna.

Entradas principales:

- Datos MI/TDAH desde `DATA_DIR`.
- Arquitecturas en `src/stf_kernelshap/modeling/models.py`.
- Optuna journals y pesos previos si existen.

Salidas:

- `Models/.../Optuna/*.journal`
- `Models/.../Models/*.weights.h5`

Configuraciones principales:

- `model_name`: `eegnet`, `shallowconvnet` o `tgarnet`.
- `window_name`: `2.5-5` o `0-7` para MI.
- `worker_id`: particion de sujetos MI.
- Defaults de funciones: `n_trials=20`, `epochs=100`, `batch_size=16`.
- Semilla base: `42`.

### `TDAH_MI_Results.ipynb`

Reconstruye los mejores modelos con parametros Optuna y genera metricas finales.

Salidas:

- `Results/MI_results.csv`
- `Results/TDAH_results.csv`

Configuraciones principales:

- Ventanas/sujetos MI seleccionados.
- Modelos evaluados: `eegnet`, `shallowconvnet`, `tgarnet`.
- Repeticiones y folds.
- Control de semilla mediante `seed`, `seed_gap` y fold.

### `run_xai_models_2.ipynb`

Genera atribuciones XAI para MI y ADHD/TDAH.

Salidas:

- `Results/attributions_y_test/...`
- `Results/attributions_y_pred/...`

Configuraciones principales:

- `mi_subjects_to_extract`
- `folds_to_extract`
- `xai_methods`
- `models_to_run`
- `use_y_test`
- `sample_indices`

Nota importante:

- Mantener las combinaciones modelo/metodo configuradas en el cuaderno.
- EEGNet no se usa con el set completo de metodos perturbativos/gradiente porque su firma de entrada actual no es compatible con todos los metodos del runner, especialmente `Occlusion`, `IntegratedGradients` y `GradCAM++`.

### `Computational_cost_scalability.ipynb`

Estima o reutiliza tiempos de computo por muestra explicada.

Salidas:

- `Results/xai_timing_logs.csv`
- `Figures/Computational_cost_scalability.pdf`

Defaults:

- `DEBUG = False`
- `OVERWRITE_TIMINGS = False`
- `timing_model = "tgarnet"`
- `timing_hardware = "NVIDIA T4 GPU"`
- `timing_sample_fraction = 0.05`
- `timing_min_trials = 20`
- `timing_max_trials = 100`
- `timing_random_state = 42`
- `save_attributions = False`

Comportamiento:

- Si existe `Results/xai_timing_logs.csv` y `OVERWRITE_TIMINGS=False`, el cuaderno reutiliza esos tiempos y solo regenera la figura.
- Si `OVERWRITE_TIMINGS=True`, borra el log existente y vuelve a estimar los tiempos.
- Si `DEBUG=True`, el mismo flujo escribe en `Temp/Results/` y `Temp/Figures/`.
- El benchmark debe mantener `timing_model="tgarnet"` para el set completo de metodos XAI.

Figura:

- Ordena metodos de mayor a menor tiempo medio.
- Usa etiquetas XAI rotadas 90 grados.
- No usa titulos internos ni xlabel.
- Usa ylabel `Time [s]`.
- La leyenda de color queda vertical en la parte superior derecha.

### `Faithfulness_summary.ipynb`

Resume metricas de faithfulness MoRF/ROAD.

Salidas:

- `Results/faithfulness_selected_relevance/faithfulness_auc_summary_for_paper.csv`
- Archivos auxiliares por muestra y tablas listas para figura.

Configuraciones principales:

- Ventanas MI.
- Metodos XAI.
- Modelo usado para las atribuciones.
- Porcentajes de perturbacion.
- Metrica de ranking MoRF/ROAD.

### `Classification_results.ipynb`

Genera figuras resumen de desempeno de clasificacion.

Entradas:

- `Results/MI_results.csv`
- `Results/TDAH_results.csv`

Salidas:

- Figuras en `Figures/`.

Configuraciones principales:

- Orden de modelos.
- Orden de metricas.
- Ventana MI seleccionada.
- Directorio de guardado.

### `Topoplots.ipynb`

Genera mapas topograficos, mapas STF y figuras espectrales.

Entradas:

- Atribuciones `.npz` desde `Results/attributions_*`.
- Datos MI/TDAH.
- Pesos/modelos entrenados cuando se requieren metricas de perturbacion.

Salidas:

- Figuras topograficas y de relevancia en `Figures/`.
- Resultados seleccionados de faithfulness cuando aplica.

Configuraciones principales:

- Sujetos/folds MI.
- Fold ADHD/TDAH.
- `model_name`.
- `XAI_METHODS`.
- Bandas de frecuencia.
- Segmentos temporales.
- Modo de agregacion de relevancia.

## Parametros XAI por defecto

El runner central es `compute_xai_all_xtest(...)`, llamado normalmente por `run_mi_xai_and_save(...)` o `run_tdah_xai_and_save(...)`.

Parametros compartidos:

- `X_test`: muestras a explicar.
- `X_train`, `y_train`: fondo/referencia estratificada cuando el metodo lo requiere.
- `y_pred`: clase objetivo; en los cuadernos puede ser `y_test` o prediccion del modelo.
- `method`: `KernelSHAP`, `LIME`, `Occlusion`, `IntegratedGradients`, `GradCAM++` o `STF-KernelSHAP`.
- `sample_indices`: subconjunto de ensayos.
- `label_source`: `y_pred` o `y_test`.
- `output_layer_name`: default `out_activation`.

Defaults adaptativos:

- `KernelSHAP`: `background_size=min(100, max(8, 0.05*n_samples))`, `nsamples=500`, `l1_reg="num_features(200)"`.
- `LIME`: `background_size=min(200, max(30, 0.10*n_samples))`, `num_features=200`, `num_samples=1000`.
- `Occlusion`: `window_seconds=1.0`, `stride_seconds=0.25`, `baseline_value=None`, `occ_batch_size=256`.
- `IntegratedGradients`: `baseline=None`, `steps=50`, `batch_size=1`.
- `GradCAM++`: ultima capa `Conv2D` compatible; para ShallowConvNet se usa `Conv2D_1`.
- `STF-KernelSHAP`: `nfft=512`, `nsamples=500`, `l1_reg="num_features(200)"`, `baseline_tf=None`, `silent=True`.

Estructura STF-KernelSHAP:

- MI: bandas theta `(4, 8)`, alpha `(8, 13)`, beta `(13, 30)`, gamma `(30, 40)` Hz.
- ADHD/TDAH: delta `(0.5, 4)`, theta `(4, 8)`, alpha `(8, 13)`, beta `(13, 30)`, gamma `(30, 40)` Hz.
- MI `0-7`: segmentos `(0, 2)`, `(2, 2.5)`, `(2.5, 5)`, `(5, 7)` s.
- Otras ventanas: segmento temporal completo si no se define `time_segments_sec`.
- Numero de features estructurados: `M_breve = C_breve * Q`.

## Semillas

Default general: `42`.

- `set_seed(seed=42)` fija `PYTHONHASHSEED`, `random`, NumPy y TensorFlow.
- Optuna usa `GPSampler(seed=42)`.
- XAI usa `random_state=42` por defecto.
- En STF-KernelSHAP se usa `random_state + sample_position` por muestra explicada.

## Ejemplo minimo MI

```python
from stf_kernelshap.visualization.topoplots import build_mi_data_from_npz
from stf_kernelshap.xai.runner import run_mi_xai_and_save

mi_subjects = {"2.5-5": {43: 1}}
mi_data = build_mi_data_from_npz(
    base_path_mi="MI_TDAH_Dataset/Data/MI",
    mi_subjects_to_extract=mi_subjects,
)

run_mi_xai_and_save(
    mi_data=mi_data,
    mi_subjects_to_extract=mi_subjects,
    results_dir="Results",
    models_mi_root="Models/MI",
    models_to_run=("shallowconvnet",),
    xai_methods_to_run=("STF-KernelSHAP",),
    use_y_test=True,
    sample_indices=[0, 1, 2],
    hardware="NVIDIA T4 GPU",
)
```

## Ejemplo minimo ADHD/TDAH

```python
from stf_kernelshap.visualization.topoplots import build_tdah_data_from_segmented
from stf_kernelshap.xai.runner import run_tdah_xai_and_save

tdah_data = {
    4: build_tdah_data_from_segmented(
        fold_to_extract=4,
        folds_path="MI_TDAH_Dataset/Data/TDAH/folds.pkl",
        path_adhd="MI_TDAH_Dataset/Data/TDAH/ieee/ADHD_group",
        path_control="MI_TDAH_Dataset/Data/TDAH/ieee/Control_group",
    )
}

run_tdah_xai_and_save(
    tdah_data_by_fold=tdah_data,
    folds_to_extract=[4],
    results_dir="Results",
    models_tdah_root="Models/TDAH",
    models_to_run=("tgarnet",),
    xai_methods_to_run=("STF-KernelSHAP",),
    use_y_test=True,
    sample_indices=[0, 1, 2],
    hardware="NVIDIA T4 GPU",
)
```

## Artefactos principales

- `Results/xai_timing_logs.csv`: tiempos por muestra explicada.
- `Figures/Computational_cost_scalability.pdf`: figura de costo computacional.
- `Results/MI_results.csv`: metricas MI.
- `Results/TDAH_results.csv`: metricas ADHD/TDAH.
- `Results/attributions_y_test/`: atribuciones respecto a etiqueta real.
- `Results/attributions_y_pred/`: atribuciones respecto a prediccion del modelo.

## Citacion

Si este repositorio apoya un articulo, cite el articulo y el DOI/enlace publico del dataset externo cuando este disponible.
