"""Funciones compartidas por la aplicación y por las pruebas.

Contiene exactamente lo mismo que se usó en el notebook del proyecto:
la ingeniería de características (crear_features) y la clase
GradientBoostingBalanceado, que forma parte del modelo Stacking.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

CARPETA = Path(__file__).parent
CARPETA_MODELOS = CARPETA / 'modelos'

# Los tres modelos de la aplicación (clave del archivo -> nombre para mostrar)
MODELOS = {
    'stacking': 'Stacking',
    'svm': 'SVM polinomial',
    'logistica': 'Regresión Logística',
}


class GradientBoostingBalanceado(GradientBoostingClassifier):
    """GradientBoostingClassifier que pondera a los pacientes con pesos balanceados
    (lo mismo que hace class_weight="balanced" en los demás modelos)."""

    def fit(self, X, y, sample_weight=None):
        pesos = compute_sample_weight('balanced', y)
        return super().fit(X, y, sample_weight=pesos)


def crear_features(datos):
    """Ingeniería de características del notebook: suma izquierda + derecha,
    promedio del grosor cortical y normalización por volumen intracraneal (ICV)."""
    datos = datos.copy()
    datos['Hipocampo_total'] = datos['Left_Hippocampus'] + datos['Right_Hippocampus']
    datos['Entorrinal_total'] = datos['lh_entorhinal_volume'] + datos['rh_entorhinal_volume']
    datos['Amigdala_total'] = datos['Left_Amygdala'] + datos['Right_Amygdala']
    datos['Ventriculos_total'] = datos['Left_Lateral_Ventricle'] + datos['Right_Lateral_Ventricle']
    datos['Vent_inferior_total'] = datos['Left_Inf_Lat_Vent'] + datos['Right_Inf_Lat_Vent']
    datos['Grosor_medio'] = (datos['lh_MeanThickness_thickness'] + datos['rh_MeanThickness_thickness']) / 2
    for col in ['Hipocampo_total', 'Entorrinal_total', 'Amigdala_total',
                'Ventriculos_total', 'Vent_inferior_total', 'CortexVol']:
        datos[col + '_ICV'] = datos[col] / datos['ICV'] * 1000
    return datos


def cargar_metadata():
    with open(CARPETA_MODELOS / 'metadata.json', encoding='utf-8') as f:
        return json.load(f)


def cargar_modelos():
    """Carga los tres Pipelines entrenados. El modelo Stacking se guardó desde el notebook,
    donde GradientBoostingBalanceado estaba definida en el módulo principal (__main__);
    por eso se registra ahí antes de cargar."""
    import __main__
    __main__.GradientBoostingBalanceado = GradientBoostingBalanceado
    return {clave: joblib.load(CARPETA_MODELOS / f'{clave}.joblib') for clave in MODELOS}


def punto_de_corte(modelo):
    """Probabilidad a partir de la cual el modelo clasifica como Dementia.

    Regresión Logística y Stacking: 0.5. En el SVM la clase se decide con la distancia a la
    frontera (que incluye class_weight="balanced") y la probabilidad se calcula aparte con la
    calibración de Platt, P = 1 / (1 + exp(A * distancia + B)). Con distancia = 0 queda el
    punto de corte equivalente, que no es 0.5."""
    final = modelo.named_steps['modelo'] if hasattr(modelo, 'named_steps') else modelo
    if hasattr(final, 'probB_') and len(final.probB_) == 1:
        return float(1 / (1 + np.exp(-final.probB_[0])))
    return 0.5


def preparar_entrada(pacientes, metadata):
    """Aplica crear_features y deja las columnas en el orden del entrenamiento."""
    datos = crear_features(pacientes)
    return datos[metadata['variables_modelo']]


def predecir(modelos, pacientes, metadata):
    """Devuelve una tabla con la clase y la probabilidad de Dementia de cada modelo
    para cada paciente (una fila por paciente y modelo)."""
    X = preparar_entrada(pacientes, metadata)
    filas = []
    for clave, nombre in MODELOS.items():
        modelo = modelos[clave]
        clases = modelo.predict(X)
        probabilidades = modelo.predict_proba(X)[:, 1]
        for i in range(len(X)):
            filas.append({'paciente': i, 'modelo': nombre, 'clase': int(clases[i]),
                          'diagnostico': metadata['clases'][str(int(clases[i]))],
                          'prob_dementia': float(probabilidades[i]),
                          'corte': punto_de_corte(modelo)})
    return pd.DataFrame(filas)
