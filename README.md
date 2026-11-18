# Clasificación de demencia con ADNI — Proyecto Final de Machine Learning

Proyecto Final Machine Learning
Alumno: Victor Francisco Duarte Villalobos (403230).
Facultad de Ingeniería UACH
Asesora: Dra. Olanda Prieto Ordaz.

**Aplicación en línea:** https://proyecto-final-ml-adni.streamlit.app

El repositorio contiene dos partes:

1. **La aplicación web** (raíz del repositorio). Recibe los datos de una persona (edad, sexo,
   escolaridad, copias del alelo APOE4 y volúmenes cerebrales de resonancia magnética calculados con
   FreeSurfer) y muestra la predicción de tres modelos entrenados para distinguir personas
   cognitivamente normales (CN) de personas con demencia (Dementia).
2. **Los experimentos** (`experimentos/`). El notebook con el que se entrenaron, optimizaron,
   seleccionaron y evaluaron todos los modelos, e instrucciones para reproducirlo.

## Modelos de la aplicación

Se entrenaron y optimizaron modelos de varias familias (lineales, máquinas de soporte vectorial,
árboles y ensambles). La métrica principal fue la Balanced Accuracy en validación cruzada repetida
(5 folds × 5 repeticiones). Como las diferencias entre los mejores modelos fueron mínimas, se eligieron
tres de familias distintas: el mejor ensamble, el mejor modelo individual y el mejor modelo lineal.

| Modelo | Configuración | Balanced Accuracy (CV repetida) | Balanced Accuracy (Test) | ROC-AUC (Test) |
|---|---|---|---|---|
| Stacking | Regresión Logística + SVM polinomial + Gradient Boosting balanceado; metamodelo: Regresión Logística balanceada | 0.892 ± 0.025 | 0.901 | 0.960 |
| SVM polinomial | grado 2, coef0 = 1, C = 1, class_weight = balanced | 0.892 ± 0.028 | 0.892 | 0.962 |
| Regresión Logística | L2, C = 0.01, class_weight = balanced | 0.889 ± 0.021 | 0.882 | 0.943 |

Cada archivo `.joblib` contiene el Pipeline completo: imputación (MICE), escalamiento,
codificación del sexo y el modelo. Se entrenaron solo con el conjunto de entrenamiento
(1,299 participantes) y se evaluaron una vez en el conjunto de prueba (325).

## Estructura

```
app.py                          Interfaz de Streamlit (5 secciones, con diseño propio en CSS)
.streamlit/config.toml          Tema de la aplicación (colores y tipografía Montserrat)
features.py                     Ingeniería de características, carga de modelos y predicción
modelos/                        stacking.joblib, svm.joblib, logistica.joblib,
                                metadata.json (variables y rangos) y resultados.json (métricas de Test)
datos/perfiles_sinteticos.csv   Perfiles de ejemplo (sintéticos)
tests/prueba_app_vs_python.py   Compara las predicciones de la aplicación contra Python
requirements.txt                Dependencias con versión fija
experimentos/                   Notebook de los experimentos (sin salidas) e instrucciones
```

## Uso de la aplicación

En línea no requiere instalar nada: se abre https://proyecto-final-ml.streamlit.app, se elige la
sección **Predicción**, se capturan los datos (o se carga un perfil de ejemplo o un CSV con las mismas
columnas que `datos/perfiles_sinteticos.csv`) y se presiona **Clasificar**. La aplicación pide la edad
y la escolaridad en años enteros.

Ejecución local (requiere Python 3.13, la versión con la que se guardaron los modelos):

```
pip install -r requirements.txt
streamlit run app.py
```

Prueba de funcionamiento (compara las probabilidades que muestra la aplicación contra las de
Python con los mismos datos):

```
python tests/prueba_app_vs_python.py
```

Despliegue: Streamlit Community Cloud, archivo principal `app.py`, Python 3.13 (se elige en
*Advanced settings*; scikit-learn 1.6.1 no tiene versión para Python 3.14).

## Datos

Los modelos se entrenaron con datos de la Alzheimer's Disease Neuroimaging Initiative (ADNI,
adni.loni.usc.edu). El acuerdo de uso de ADNI no permite redistribuir los datos, por lo que este
repositorio no los incluye. Además, el repositorio es privado porque los archivos `svm.joblib` y
`stacking.joblib` guardan los vectores de soporte del SVM, que son filas (estandarizadas) de
participantes de entrenamiento. Los perfiles de `datos/perfiles_sinteticos.csv` no son personas reales:
se generaron al azar a partir de las medias y covarianzas de cada grupo.

## Limitaciones

Solo distingue CN de Dementia (no incluye deterioro cognitivo leve), requiere volúmenes de
FreeSurfer armonizados como en ADNI y no usa pruebas cognitivas.
