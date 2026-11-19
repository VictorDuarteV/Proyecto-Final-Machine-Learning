"""Prueba de funcionamiento: compara las predicciones de la aplicación contra las obtenidas
directamente en Python con los mismos datos de entrada.

Uso (desde la carpeta del proyecto):
    python tests/prueba_app_vs_python.py                         # perfiles sintéticos
    python tests/prueba_app_vs_python.py local/pacientes_ejemplo.csv   # pacientes reales (solo en local)

1. "Python directo": carga cada modelo con joblib y llama a predict / predict_proba.
2. "Aplicación": ejecuta app.py sin navegador (streamlit.testing), captura los mismos datos
   en el formulario, presiona Clasificar y lee las probabilidades de la tabla de resultados
   que se muestra en pantalla.

"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import features as F  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

archivo = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / 'datos' / 'perfiles_sinteticos.csv'
pacientes = pd.read_csv(archivo)
metadata = F.cargar_metadata()
modelos = F.cargar_modelos()
numericas = [c for c in metadata['variables_originales'] if c not in ('APOE4', 'Sexo')]

# 1. Python directo
X = F.crear_features(pacientes)[metadata['variables_modelo']]
directo = {nombre: (modelos[clave].predict(X), modelos[clave].predict_proba(X)[:, 1])
           for clave, nombre in F.MODELOS.items()}


def probabilidades_en_pantalla(app):
    """Lee las probabilidades de la tabla de resultados que se muestra en la interfaz."""
    for elemento in app.markdown:
        if 'tabla resultados' in elemento.value:
            return dict(re.findall(r'data-modelo="([^"]+)">([^<]+)<', elemento.value))
    return {}


# 2. Aplicación
filas, errores = [], 0
for i, paciente in pacientes.iterrows():
    app = AppTest.from_file(str(RAIZ / 'app.py'), default_timeout=120).run()
    app.radio(key='seccion').set_value('Predicción').run()
    for col in numericas:
        valor = float(paciente[col])
        app.number_input(key='in_' + col).set_value(valor)
    app.selectbox(key='in_Sexo').set_value(paciente['Sexo'])
    apoe = 'Desconocido' if pd.isna(paciente['APOE4']) else str(int(paciente['APOE4']))
    app.selectbox(key='in_APOE4').set_value(apoe)
    app.button(key='clasificar').click().run()
    assert not app.exception, app.exception
    tabla = app.session_state['prediccion']['tabla']
    mostradas = probabilidades_en_pantalla(app)
    for _, fila in tabla.iterrows():
        clase_py, prob_py = directo[fila['modelo']][0][i], directo[fila['modelo']][1][i]
        en_pantalla = mostradas.get(fila['modelo'], '')
        ok = (fila['clase'] == clase_py) and abs(fila['prob_dementia'] - prob_py) < 1e-9 \
            and en_pantalla == f'{prob_py * 100:.1f} %'
        errores += not ok
        filas.append({'Paciente': i + 1, 'Modelo': fila['modelo'],
                      'Clase (Python)': clase_py, 'Clase (app)': fila['clase'],
                      'Prob. (Python)': round(prob_py, 6), 'Prob. (app)': round(fila['prob_dementia'], 6),
                      'En pantalla': en_pantalla, 'Coincide': 'sí' if ok else 'NO'})

print(pd.DataFrame(filas).to_string(index=False))
print(f'\n{len(filas)} comparaciones, {errores} diferencias.')
sys.exit(1 if errores else 0)
