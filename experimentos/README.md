# Experimentos

`Proyecto_Final_ADNI_Victor_Duarte.ipynb` contiene todo el trabajo experimental del proyecto:
análisis exploratorio, preprocesamiento, modelos con valores por defecto (Dummy, Regresión Logística,
SGD, SVM lineal, polinomial y RBF, Random Forest, AdaBoost, Gradient Boosting y XGBoost),
optimización con `GridSearchCV`, ensambles (Voting duro, Voting suave y Stacking), selección del
modelo con validación cruzada repetida 5×5, evaluación final en prueba con *bootstrap*, curvas ROC y
Precision-Recall, SHAP, comparación con la literatura y exportación de los modelos de la aplicación.

El notebook se sube **sin salidas**, porque algunas de ellas (por ejemplo, las primeras filas del
DataFrame) muestran datos de participantes de ADNI, que no se pueden redistribuir.

## Cómo reproducir los experimentos

### 1. Obtener acceso a ADNI

Solicitar acceso en https://adni.loni.usc.edu y aceptar su acuerdo de uso de datos. Con el acceso
aprobado, descargar desde LONI IDA (*Study Data*) el paquete **ADNIMERGE2** para R (fases ADNI1, GO,
2, 3 y 4).

### 2. Generar los dos archivos CSV

El notebook lee dos archivos que no están en este repositorio:

**`ADNI_practica.csv`**: una fila por participante en su primera visita, armada con las tablas del
paquete ADNIMERGE2:

- `ADSP_PHC_DEMODX`: diagnóstico (`DX`: CN, MCI o Dementia), sexo, edad y años de escolaridad.
- `ADSP_PHC_T1_FS`: volumen intracraneal (ICV) y volúmenes y grosores de FreeSurfer armonizados con
  ComBat (columnas `_combat`): hipocampo, corteza entorrinal, amígdala, ventrículo lateral y
  ventrículo lateral inferior de cada hemisferio, grosor cortical medio de cada hemisferio y `CortexVol`.
- `APOERES`: genotipo APOE; `APOE4` es el número de alelos ε4 (0, 1 o 2).

Pasos de limpieza: excluir las visitas de participantes que no cumplieron los criterios de selección
(`VISCODE2 == 'f'`) y los diagnósticos fuera de CN, MCI y Dementia; unir el diagnóstico con la
resonancia por participante (`RID`) con la fecha más cercana y una tolerancia de 90 días; conservar
la primera visita de cada participante. Columnas que usa el notebook: `RID`, `VISCODE2`, `EXAMDATE`,
`PHC_SCANDATE`, `DX`, `Edad`, `Sexo`, `Escolaridad`, `APOE4`, `ICV` y las 13 medidas cerebrales
(los nombres están en `modelos/metadata.json`, clave `variables_originales`). Resultado: 2,833
participantes, de los cuales 1,624 son CN o Dementia.

**`APOE4_C2N.csv`**: columnas `RID` y `APOE4_C2N`, con el número de alelos ε4 determinado en plasma
(tabla `C2N_PRECIVITYAD2_PLASMA`), para recuperar el genotipo de los participantes que no lo tienen en
`APOERES`.

### 3. Ejecutar el notebook

1. Subir `ADNI_practica.csv` y `APOE4_C2N.csv` a la raíz de Google Drive (`MyDrive/`).
2. Abrir el notebook en Google Colab y ejecutar todas las celdas. Al inicio se monta Google Drive.
3. Las figuras se guardan en `MyDrive/figuras_proyecto/` (300 dpi) y los modelos en
   `MyDrive/modelos_adni/`.

### Versiones y semillas

Ejecutado en Google Colab con Python 3.13.18, scikit-learn 1.6.1, XGBoost 3.4.1, SHAP 0.54.1,
NumPy 2.4.6, pandas 2.2.3 y joblib 1.6.0 (el notebook imprime las versiones al inicio).
Semilla `random_state = 42` para la partición 80/20, las particiones de validación cruzada y los
modelos; `random_state = 0` para la validación cruzada repetida 5×5.
