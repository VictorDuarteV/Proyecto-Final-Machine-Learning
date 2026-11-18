"""Aplicación web del Proyecto Final de Machine Learning.

Clasificación de demencia (CN vs Dementia) con datos de ADNI usando los tres
modelos seleccionados en el proyecto: Stacking, SVM polinomial y Regresión Logística.

Ejecución local:  streamlit run app.py
"""
import json
from decimal import ROUND_HALF_UP, Decimal

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import features as F

st.set_page_config(page_title='Clasificación de demencia con ADNI', layout='wide')

# ---------------------------------------------------------------- colores
FONDO, SUPERFICIE, BORDE = '#F5F3EE', '#FFFFFF', '#E2DDD3'
TEXTO, TEXTO_SEC = '#1C2B33', '#56656D'
PRIMARIO = '#0E4F5C'
COLOR_CN, COLOR_DEM = '#1F6F8B', '#C4572F'
COLOR_MODELO = {'Stacking': '#0E4F5C', 'SVM polinomial': '#7FA9B3', 'Regresión Logística': '#D49A3A'}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');
html, body, .stApp, button, input, select, textarea {font-family: 'Montserrat', sans-serif;}
.stApp * {font-variant-numeric: tabular-nums;}
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], footer {display: none !important;}
[data-testid="stMainBlockContainer"] {max-width: 1184px; padding: 0 32px 80px;}

/* Encabezado con navegación */
.st-key-encabezado {background: #FFFFFF; box-shadow: 0 0 0 100vmax #FFFFFF; clip-path: inset(0 -100vmax);
                    position: relative; margin-bottom: 16px;}
.st-key-encabezado::after {content: ""; position: absolute; left: -100vmax; right: -100vmax; bottom: 0;
                           height: 1px; background: #E2DDD3;}
.marca {height: 62px; transform: translateY(-8px); display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 17px; color: #1C2B33;}
.marca .logo {width: 30px; height: 30px; border-radius: 8px; background: #0E4F5C; color: #FFFFFF;
              display: inline-flex; align-items: center; justify-content: center; font-size: 16px;}
.st-key-encabezado [data-testid="stElementContainer"], .st-key-encabezado [data-testid="stRadio"] {width: 100%;}
.st-key-encabezado div[role="radiogroup"] {gap: 2px; justify-content: flex-end; flex-wrap: wrap; width: 100%;}
.st-key-encabezado [data-testid="stRadioOption"] {margin: 0; padding: 20px 14px 18px; cursor: pointer;
                                                 border-bottom: 2px solid transparent;}
.st-key-encabezado [data-testid="stRadioOption"] > div > div:first-child {display: none;}
.st-key-encabezado [data-testid="stRadioOption"] p {font-size: 14px; font-weight: 500; color: #56656D;}
.st-key-encabezado [data-testid="stRadioOption"][data-selected="true"] {border-bottom-color: #0E4F5C;}
.st-key-encabezado [data-testid="stRadioOption"][data-selected="true"] p {color: #0E4F5C; font-weight: 600;}
div[data-testid="stElementContainer"]:has(style) {display: none;}

/* Texto */
div[data-testid="stMarkdownContainer"] .titulo {padding: 0; font-size: 34px; font-weight: 700; letter-spacing: -0.015em; line-height: 1.2; color: #1C2B33;
         margin: 16px 0 6px;}
div[data-testid="stMarkdownContainer"] .seccion {padding: 0; font-size: 24px; font-weight: 600; line-height: 1.25; color: #1C2B33; margin: 28px 0 10px;}
div[data-testid="stMarkdownContainer"] .subseccion {padding: 0; font-size: 20px; font-weight: 600; color: #1C2B33; margin: 0 0 6px;}
div[data-testid="stMarkdownContainer"] .texto {color: #36454D; font-size: 15px; line-height: 1.6; max-width: 860px; margin: 0 0 14px;}
div[data-testid="stMarkdownContainer"] .nota {color: #56656D; font-size: 13px; line-height: 1.5; margin: 8px 0 0;}
.etiqueta {font-size: 12px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #0E4F5C;}
.fila-titulo {display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between;
              gap: 6px 24px; margin: 28px 0 14px;}
div[data-testid="stMarkdownContainer"] .fila-titulo .seccion {margin: 0;}
.fila-titulo > span {font-size: 14px; color: #56656D;}

/* Tarjetas */
.tarjeta {background: #FFFFFF; border: 1px solid #E2DDD3; border-radius: 16px; padding: 32px 36px;}
div[data-testid="stMarkdownContainer"] .tarjeta h1 {font-size: 36px; font-weight: 700; letter-spacing: -0.015em; line-height: 1.2; margin: 6px 0 0;
             padding: 0; color: #1C2B33;}
.campos {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px 32px;
         border-top: 1px solid #ECE8E0; padding-top: 22px; margin-top: 24px;}
.campo {display: flex; flex-direction: column; gap: 2px; font-weight: 500;}
.campo span:first-child {font-size: 12px; letter-spacing: 0.04em; text-transform: uppercase; color: #56656D;}
.dos-columnas {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 32px 48px;}
div[data-testid="stMarkdownContainer"] .dos-columnas .seccion {margin-top: 8px;}
.cifras {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px;}
.cifras.dos {grid-template-columns: repeat(2, minmax(0, 1fr));}
.cifra {background: #FFFFFF; border: 1px solid #E2DDD3; border-radius: 14px; padding: 20px 22px;
        display: flex; flex-direction: column; gap: 4px;}
.cifra .rotulo, .cifra .detalle {font-size: 13px; color: #56656D;}
.cifra .rotulo {font-weight: 500; display: flex; align-items: center; gap: 8px;}
.cifra .valor {font-size: 32px; font-weight: 600; line-height: 1.15; color: #1C2B33;}
.punto {width: 10px; height: 10px; border-radius: 50%; display: inline-block;}
.lista-modelos article {display: grid; grid-template-columns: 56px minmax(0, 1fr); gap: 4px 16px;
                        padding: 22px 0; border-top: 1px solid #E2DDD3;}
.lista-modelos article:last-child {border-bottom: 1px solid #E2DDD3;}
.lista-modelos .num {font-size: 14px; font-weight: 600; color: #0E4F5C; padding-top: 4px;}
div[data-testid="stMarkdownContainer"] .lista-modelos h3 {font-size: 21px; font-weight: 600; margin: 0 0 4px; padding: 0; color: #1C2B33;}
.lista-modelos .sub {font-size: 13px; color: #56656D; margin-bottom: 6px;}
div[data-testid="stMarkdownContainer"] .lista-modelos p {margin: 0; color: #36454D; max-width: 860px; line-height: 1.6;}
.tres {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px;}
div[data-testid="stMarkdownContainer"] .tres .cifra h3, div[data-testid="stMarkdownContainer"] .dos-tarjetas .cifra h3 {font-size: 19px; font-weight: 600; margin: 0; padding: 0; color: #1C2B33;}
div[data-testid="stMarkdownContainer"] .tres .cifra p, div[data-testid="stMarkdownContainer"] .dos-tarjetas .cifra p {margin: 0; font-size: 14px; color: #36454D; line-height: 1.6;}
.dos-tarjetas {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px;}
div[data-testid="stMarkdownContainer"] .lista {margin: 0; padding-left: 22px; color: #36454D; max-width: 860px; line-height: 1.6;}
.lista li {margin-bottom: 8px;}

/* Tablas */
.marco {background: #FFFFFF; border: 1px solid #E2DDD3; border-radius: 16px; overflow-x: auto;}
div[data-testid="stMarkdownContainer"] table.tabla {width: 100%; border-collapse: collapse; font-size: 15px;
                                                    margin: 0; border: 0;}
div[data-testid="stMarkdownContainer"] table.tabla th {text-align: left; padding: 14px 24px; font-size: 12px;
    font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: #56656D; background: #FAF8F4;
    border: 0; border-bottom: 1px solid #E2DDD3;}
div[data-testid="stMarkdownContainer"] table.tabla td {padding: 14px 24px; border: 0;
                                                       border-bottom: 1px solid #ECE8E0; color: #1C2B33;}
div[data-testid="stMarkdownContainer"] table.tabla tr:last-child td {border-bottom: 0;}
div[data-testid="stMarkdownContainer"] table.tabla .der {text-align: right;}
div[data-testid="stMarkdownContainer"] table.tabla .fuerte {font-weight: 600; white-space: nowrap;}
div[data-testid="stMarkdownContainer"] code {background: #ECE8E0; color: #1C2B33; padding: 1px 6px; border-radius: 4px;}
div[data-testid="stMarkdownContainer"] table.tabla .sec {color: #56656D;}
div[data-testid="stMarkdownContainer"] table.tabla .mejor {font-weight: 700; color: #0E4F5C;}
div[data-testid="stMarkdownContainer"] table.compacta td, div[data-testid="stMarkdownContainer"] table.compacta th {
    padding-left: 14px; padding-right: 14px; font-size: 14px;}
div[data-testid="stMarkdownContainer"] table.compacta th {font-size: 12px;}
.clase {display: inline-flex; align-items: center; gap: 8px; font-weight: 600;}
.clase::before {content: ""; width: 10px; height: 10px; border-radius: 50%;}
.clase.cn {color: #1A5E77;} .clase.cn::before {background: #1F6F8B;}
.clase.dem {color: #A8461F;} .clase.dem::before {background: #C4572F;}
.muestra {display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 8px;}

/* Avisos */
.aviso {display: flex; align-items: flex-start; gap: 12px; padding: 14px 18px; border-radius: 12px;
        border: 1px solid; margin: 0 0 16px; line-height: 1.5;}
.aviso svg {flex: none; margin-top: 1px;}
.aviso.ok {background: #E6F1EE; border-color: #B9D8CF; color: #1E4D41;}
.aviso.alerta {background: #FBF1E3; border-color: #EBCB9E; color: #6B4410;}
.aviso.error {background: #FBEAE4; border-color: #E9B7A4; color: #7A2E14;}

/* Gráfica de probabilidades */
.figura {background: #FFFFFF; border: 1px solid #E2DDD3; border-radius: 16px; padding: 24px 28px; margin-top: 16px;}
.figura .cabeza {display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between;
                 gap: 8px 24px; margin-bottom: 18px;}
.figura .cabeza b {font-size: 18px; font-weight: 600;}
.leyenda {display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px; color: #56656D;}
.leyenda span {display: inline-flex; align-items: center; gap: 6px;}
.leyenda i {display: inline-block; width: 12px; height: 12px; border-radius: 3px;}
.leyenda i.linea {width: 3px; height: 14px; border-radius: 1px; background: #1C2B33;}
.barras {display: flex; flex-direction: column; gap: 14px;}
.barras .fila {display: grid; grid-template-columns: 160px minmax(0, 1fr) 64px; gap: 14px; align-items: center;
               font-size: 14px;}
.barras .fila > span:first-child {font-weight: 500;}
.barras .fila > span:last-child {text-align: right;}
.pista {position: relative; height: 24px; background: #EFEBE3; border-radius: 6px;}
.pista .barra {height: 100%; border-radius: 6px;}
.pista .corte {position: absolute; top: -5px; width: 3px; height: 34px; background: #1C2B33; border-radius: 2px;
               transform: translateX(-1.5px);}
.eje {display: flex; justify-content: space-between; font-size: 12px; color: #56656D;}

/* Matrices de confusión */
.matriz-tarjeta {background: #FFFFFF; border: 1px solid #E2DDD3; border-radius: 16px; padding: 20px 22px;}
.matriz-tarjeta b {font-size: 18px; font-weight: 600; display: block; margin-bottom: 12px;}
.matriz {display: grid; grid-template-columns: 84px minmax(0, 1fr) minmax(0, 1fr); gap: 6px; font-size: 12px;}
.matriz .eti {color: #56656D; font-weight: 500; align-self: center;}
.matriz .eti.col {text-align: center;}
.celda {height: 82px; border-radius: 10px; display: flex; flex-direction: column; align-items: center;
        justify-content: center;}
.celda strong {font-size: 22px; font-weight: 600;}

/* Widgets */
.st-key-origen, [data-testid="stForm"], .st-key-grafica {background: #FFFFFF; border: 1px solid #E2DDD3 !important;
                                                          border-radius: 16px; padding: 24px 28px;}
[data-testid="stForm"] {padding: 28px;}
[data-testid="stWidgetLabel"] p {font-size: 13px; font-weight: 500; color: #36454D;}
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {display: none;}
.encabezado-tabla {font-size: 12px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
                   color: #56656D;}
.estructura {font-size: 14px; font-weight: 500; min-height: 40px; display: flex; align-items: center;}
.stButton button, .stFormSubmitButton button {font-weight: 600; min-height: 44px;}
.stFormSubmitButton button {min-height: 50px; font-size: 16px;}
.stFormSubmitButton button:hover {background: #0A3D47; border-color: #0A3D47; color: #FFFFFF;}
[data-testid="stFileUploaderDropzone"] {background: #FAF8F4; border: 1px dashed #C9C2B4; padding: 8px 8px 8px 16px;}
[data-testid="stFileUploaderDropzone"] {flex-direction: row-reverse; justify-content: space-between;}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span {font-size: 0;}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span::after {content: "Arrastra un CSV con una o más personas";
    font-size: 14px; color: #56656D;}
[data-testid="stFileUploaderDropzoneInstructions"] > div > small, [data-testid="stFileUploaderDropzoneInstructions"] svg,
[data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"] {display: none;}
[data-testid="stFileUploaderDropzone"] button {min-height: 36px;}
[data-testid="stFileUploaderDropzone"] button p {font-size: 0;}
[data-testid="stFileUploaderDropzone"] button p::after {content: "Buscar archivo"; font-size: 14px;}
[data-testid="stFileUploaderDropzoneInstructions"] {flex: 1; white-space: nowrap;}
.st-key-origen .stButton button {min-height: 40px;}
</style>
""",
    unsafe_allow_html=True,
)


def html(texto):
    """Muestra HTML. Se quitan la sangría y las líneas vacías para que Markdown no lo trate como código."""
    st.markdown('\n'.join(linea.strip() for linea in texto.splitlines() if linea.strip()), unsafe_allow_html=True)


ICONO_OK = ('<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2E6B5A" stroke-width="2.2" '
            'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/>'
            '<path d="M8 12.5l2.6 2.6L16 9.5"/></svg>')
ICONO_ALERTA = ('<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#A86B12" stroke-width="2.2" '
                'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3.5l9 16H3z"/>'
                '<path d="M12 10v4.5"/><path d="M12 17.5h.01"/></svg>')


def aviso(texto, tipo='alerta'):
    icono = ICONO_OK if tipo == 'ok' else ICONO_ALERTA
    html(f'<div class="aviso {tipo}" role="status">{icono}<span>{texto}</span></div>')


def clase_html(diagnostico):
    return f'<span class="clase {"dem" if diagnostico == "Dementia" else "cn"}">{diagnostico}</span>'


def porcentaje(p):
    """0.984 -> '98.4 %' (con espacio antes del signo, como en el reporte)."""
    return f'{p * 100:.1f} %'


def redondear3(x):
    """Redondeo a 3 decimales hacia arriba en .5 (0.8625 -> 0.863), como en el reporte."""
    return float(Decimal(repr(float(x))).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))


def entero(x):
    """Redondea a entero hacia arriba en .5 (70.5 -> 71), como se hace a mano."""
    return int(np.floor(float(x) + 0.5))


# ---------------------------------------------------------------- carga de recursos
@st.cache_resource
def obtener_modelos():
    # Se cargan una sola vez; después quedan en memoria
    return F.cargar_modelos()


@st.cache_data
def obtener_metadata():
    return F.cargar_metadata()


@st.cache_data
def obtener_resultados():
    with open(F.CARPETA_MODELOS / 'resultados.json', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def obtener_perfiles():
    return pd.read_csv(F.CARPETA / 'datos' / 'perfiles_sinteticos.csv')


modelos = obtener_modelos()
metadata = obtener_metadata()
resultados = obtener_resultados()
perfiles = obtener_perfiles()
RANGOS = metadata['rangos_train']
VARIABLES = metadata['variables_originales']

CAMPOS_VOLUMEN = [
    ('Hipocampo', 'Left_Hippocampus', 'Right_Hippocampus'),
    ('Corteza entorrinal', 'lh_entorhinal_volume', 'rh_entorhinal_volume'),
    ('Amígdala', 'Left_Amygdala', 'Right_Amygdala'),
    ('Ventrículo lateral', 'Left_Lateral_Ventricle', 'Right_Lateral_Ventricle'),
    ('Ventrículo lateral inferior', 'Left_Inf_Lat_Vent', 'Right_Inf_Lat_Vent'),
]
OPCIONES_APOE4 = ['0', '1', '2', 'Desconocido']
ENTERAS = ['Edad', 'Escolaridad']
NUMERICAS = [c for c in VARIABLES if c not in ('APOE4', 'Sexo')]
NOMBRES_CAMPO = {
    'Edad': 'Edad', 'Escolaridad': 'Escolaridad', 'ICV': 'Volumen intracraneal (ICV)',
    'CortexVol': 'Volumen de la corteza',
    'lh_MeanThickness_thickness': 'Grosor cortical medio (izquierdo)',
    'rh_MeanThickness_thickness': 'Grosor cortical medio (derecho)',
}
for _nombre, _izq, _der in CAMPOS_VOLUMEN:
    NOMBRES_CAMPO[_izq] = f'{_nombre} (izquierdo)'
    NOMBRES_CAMPO[_der] = f'{_nombre} (derecho)'


# ---------------------------------------------------------------- estado del formulario
def cargar_en_formulario(fila, origen):
    """Copia los valores de un paciente (fila de un DataFrame o diccionario) a los campos.
    Edad y escolaridad se capturan en años enteros."""
    for col in NUMERICAS:
        st.session_state['in_' + col] = entero(fila[col]) if col in ENTERAS else float(fila[col])
    st.session_state['in_Sexo'] = fila['Sexo'] if fila['Sexo'] in metadata['categorias_sexo'] else 'Femenino'
    apoe = fila['APOE4']
    st.session_state['in_APOE4'] = 'Desconocido' if pd.isna(apoe) else str(int(float(apoe)))
    st.session_state['origen'] = origen
    st.session_state.pop('prediccion', None)


if 'in_Edad' not in st.session_state:
    cargar_en_formulario(perfiles.iloc[0], perfiles['perfil'].iloc[0])


def leer_formulario():
    fila = {col: float(st.session_state['in_' + col]) for col in NUMERICAS}
    fila['Sexo'] = st.session_state['in_Sexo']
    apoe = st.session_state['in_APOE4']
    fila['APOE4'] = np.nan if apoe == 'Desconocido' else float(apoe)
    return pd.DataFrame([fila])[VARIABLES]


def validar(paciente):
    """Devuelve (errores, avisos). Los errores impiden la predicción; los avisos no."""
    errores, avisos = [], []
    for col in NUMERICAS:
        valor = paciente[col].iloc[0]
        r = RANGOS[col]
        nombre = NOMBRES_CAMPO[col]
        if pd.isna(valor) or valor <= 0:
            errores.append(f'<b>{nombre}</b> debe ser un número mayor que 0.')
        elif valor < 0.5 * r['min'] or valor > 1.5 * r['max']:
            errores.append(f'<b>{nombre}</b> = {valor:,.2f} no es un valor plausible (en los datos de '
                           f'entrenamiento va de {r["min"]:,.2f} a {r["max"]:,.2f}). Revisa las unidades.')
        elif valor < r['min'] or valor > r['max']:
            avisos.append(f'<b>{nombre}</b> = {valor:,.2f} está fuera del rango de entrenamiento '
                          f'({r["min"]:,.2f} a {r["max"]:,.2f}); la predicción es menos confiable.')
    return errores, avisos


def metricas_desde_matriz(mc):
    """Métricas de Test calculadas directamente de la matriz de confusión (valores exactos)."""
    tn, fp, fn, tp = mc['TN'], mc['FP'], mc['FN'], mc['TP']
    recall, especificidad, precision = tp / (tp + fn), tn / (tn + fp), tp / (tp + fp)
    return {
        'Balanced Acc.': (recall + especificidad) / 2,
        'Recall': recall,
        'Especificidad': especificidad,
        'Precision': precision,
        'F1': 2 * precision * recall / (precision + recall),
        'Accuracy': (tp + tn) / (tn + fp + fn + tp),
    }


# ---------------------------------------------------------------- encabezado y navegación
SECCIONES = ['Proyecto', 'Predicción', 'Características', 'Comparación', 'Interpretación']
with st.container(key='encabezado'):
    c_marca, c_nav = st.columns([1, 2.4], vertical_alignment='center')
    with c_marca:
        html('<div class="marca"><span class="logo" aria-hidden="true">D</span>Demencia · ADNI</div>')
    with c_nav:
        seccion = st.radio('Secciones', SECCIONES, horizontal=True, label_visibility='collapsed', key='seccion')


# ================================================================ 1. PROYECTO
if seccion == 'Proyecto':
    html("""
    <section class="tarjeta">
      <span class="etiqueta">Proyecto Final · Machine Learning</span>
      <h1>Clasificación de demencia a partir de resonancia magnética</h1>
      <div class="campos">
        <div class="campo"><span>Alumno</span><span>Victor Francisco Duarte Villalobos</span></div>
        <div class="campo"><span>Matrícula</span><span>403230</span></div>
        <div class="campo"><span>Asesora</span><span>Dra. Olanda Prieto Ordaz</span></div>
        <div class="campo"><span>Institución</span><span>Universidad Autónoma de Chihuahua</span></div>
        <div class="campo"><span>Programa</span><span>Maestría en Ingeniería en Computación</span></div>
        <div class="campo"><span>Materia</span><span>Machine Learning</span></div>
      </div>
    </section>
    """)
    html("""
    <div class="dos-columnas" style="margin-top: 40px">
      <div>
        <h2 class="seccion">Problema</h2>
        <p class="texto">La enfermedad de Alzheimer es la causa más frecuente de demencia. En una resonancia
        magnética se puede medir la atrofia del hipocampo, la corteza entorrinal y la amígdala, y el agrandamiento
        de los ventrículos. El problema es de clasificación binaria supervisada: distinguir a personas
        cognitivamente normales (CN) de personas con demencia (Dementia) a partir de datos demográficos, el alelo
        APOE4 y volúmenes cerebrales.</p>
      </div>
      <div>
        <h2 class="seccion">Objetivo de la aplicación</h2>
        <p class="texto">Recibir los datos de una persona, obtener la predicción de los tres modelos seleccionados
        en el proyecto y compararlas entre sí, junto con los resultados de su evaluación experimental.</p>
      </div>
    </div>
    """)
    html(f"""
    <h2 class="seccion" style="margin-top: 40px">Base de datos</h2>
    <p class="texto">Alzheimer's Disease Neuroimaging Initiative (ADNI), fases 1 a 4: una fila por participante
    en su primera visita, con volúmenes calculados con FreeSurfer y armonizados entre escáneres (ComBat). Las
    pruebas cognitivas (MMSE, CDR, ADAS) no se usaron como predictoras porque el diagnóstico se basa en ellas.</p>
    <div class="cifras">
      <div class="cifra"><span class="rotulo">Participantes</span><span class="valor">1,624</span>
        <span class="detalle">CN y Dementia, primera visita</span></div>
      <div class="cifra"><span class="rotulo"><i class="punto" style="background:{COLOR_CN}"></i>CN</span>
        <span class="valor">1,224</span><span class="detalle">75.4 % de la base</span></div>
      <div class="cifra"><span class="rotulo"><i class="punto" style="background:{COLOR_DEM}"></i>Dementia</span>
        <span class="valor">400</span><span class="detalle">24.6 % de la base</span></div>
      <div class="cifra"><span class="rotulo">Train / Test</span><span class="valor">1,299 / 325</span>
        <span class="detalle">80 % / 20 %, estratificada</span></div>
    </div>
    """)
    html("""
    <h2 class="seccion" style="margin-top: 48px">Modelos implementados</h2>
    <p class="texto" style="margin-bottom: 22px">Se entrenaron y optimizaron modelos de varias familias (lineales,
    máquinas de soporte vectorial, árboles y ensambles). La métrica principal fue la Balanced Accuracy en validación
    cruzada repetida. Como las diferencias entre los mejores modelos fueron mínimas, se eligieron tres de familias
    distintas: el mejor ensamble, el mejor modelo individual y el mejor modelo lineal.</p>
    <div class="lista-modelos">
      <article><span class="num">01</span><div>
        <h3>Stacking</h3>
        <div class="sub">Ensamble · Regresión Logística + SVM polinomial + Gradient Boosting balanceado;
        metamodelo: Regresión Logística balanceada</div>
        <p>Combina tres modelos y entrena un metamodelo que aprende cuánto confiar en cada uno. Obtuvo la mayor
        Balanced Accuracy en validación cruzada repetida.</p></div></article>
      <article><span class="num">02</span><div>
        <h3>SVM polinomial</h3>
        <div class="sub">Kernel · grado 2, coef0 = 1, C = 1, class_weight = balanced</div>
        <p>Busca la frontera con el mayor margen entre las clases; el kernel polinomial de grado 2 permite
        fronteras curvas. Fue el mejor modelo individual.</p></div></article>
      <article><span class="num">03</span><div>
        <h3>Regresión Logística</h3>
        <div class="sub">Lineal · L2, C = 0.01, class_weight = balanced</div>
        <p>Estima la probabilidad de demencia como una suma ponderada de las variables. Es el modelo más simple e
        interpretable, y su desempeño es prácticamente igual al de los otros dos.</p></div></article>
    </div>
    """)


# ================================================================ 2. PREDICCIÓN
elif seccion == 'Predicción':
    html('<h1 class="titulo">Predicción</h1>'
         '<p class="texto">Ingresa los datos de una persona o carga un ejemplo, y presiona Clasificar.</p>')

    with st.container(key='origen'):
        e1, e2 = st.columns(2, gap='large')
        with e1:
            p1, p2 = st.columns([3, 1.3], vertical_alignment='bottom')
            nombre_perfil = p1.selectbox('Perfil sintético', perfiles['perfil'])
            p2.button('Cargar perfil', type='secondary', width='stretch', on_click=cargar_en_formulario,
                      args=(perfiles[perfiles['perfil'] == nombre_perfil].iloc[0], nombre_perfil))
        with e2:
            archivo = st.file_uploader('Archivo CSV', type='csv')
            if archivo is not None:
                try:
                    datos_csv = pd.read_csv(archivo)
                    faltantes = [c for c in VARIABLES if c not in datos_csv.columns]
                    if faltantes:
                        aviso('Al archivo le faltan estas columnas: ' + ', '.join(faltantes), 'error')
                    elif datos_csv[NUMERICAS].isna().any().any():
                        aviso('El archivo tiene valores vacíos en columnas numéricas (solo APOE4 puede ir vacío).',
                              'error')
                    else:
                        f1, f2 = st.columns([1, 1.6], vertical_alignment='bottom')
                        fila_csv = int(f1.number_input('Fila', 1, len(datos_csv), 1))
                        f2.button('Cargar fila', width='stretch', on_click=cargar_en_formulario,
                                  args=(datos_csv.iloc[fila_csv - 1], f'Fila {fila_csv} de {archivo.name}'))
                except Exception:
                    aviso('No se pudo leer el archivo. Verifica que sea un CSV válido.', 'error')

    with st.form('formulario'):
        html('<h2 class="subseccion">Datos demográficos y genética</h2>')
        d1, d2, d3, d4 = st.columns(4)
        d1.number_input('Edad (años)', min_value=0, step=1, format='%d', key='in_Edad')
        d2.selectbox('Sexo', metadata['categorias_sexo'], key='in_Sexo')
        d3.number_input('Escolaridad (años)', min_value=0, step=1, format='%d', key='in_Escolaridad')
        d4.selectbox('Copias del alelo APOE4', OPCIONES_APOE4, key='in_APOE4')

        html('<h2 class="subseccion" style="margin-top: 20px">Resonancia magnética (FreeSurfer)</h2>'
             '<p class="nota" style="margin: 0 0 12px">Volúmenes en mm³ y grosor cortical en mm.</p>')
        anchos = [0.8, 1, 1]
        h1, h2, h3 = st.columns(anchos)
        h1.markdown('<span class="encabezado-tabla">Estructura</span>', unsafe_allow_html=True)
        h2.markdown('<span class="encabezado-tabla">Hemisferio izquierdo</span>', unsafe_allow_html=True)
        h3.markdown('<span class="encabezado-tabla">Hemisferio derecho</span>', unsafe_allow_html=True)
        for nombre, izq, der in CAMPOS_VOLUMEN:
            r1, r2, r3 = st.columns(anchos, vertical_alignment='center')
            r1.markdown(f'<span class="estructura">{nombre}</span>', unsafe_allow_html=True)
            r2.number_input(f'{nombre} izquierdo', min_value=0.0, step=10.0, format='%.0f', key='in_' + izq,
                            label_visibility='collapsed')
            r3.number_input(f'{nombre} derecho', min_value=0.0, step=10.0, format='%.0f', key='in_' + der,
                            label_visibility='collapsed')
        r1, r2, r3 = st.columns(anchos, vertical_alignment='center')
        r1.markdown('<span class="estructura">Grosor cortical medio (mm)</span>', unsafe_allow_html=True)
        r2.number_input('Grosor cortical medio izquierdo', min_value=0.0, step=0.01, format='%.3f',
                        key='in_lh_MeanThickness_thickness', label_visibility='collapsed')
        r3.number_input('Grosor cortical medio derecho', min_value=0.0, step=0.01, format='%.3f',
                        key='in_rh_MeanThickness_thickness', label_visibility='collapsed')
        t1, t2 = st.columns(2)
        t1.number_input('Volumen de la corteza (CortexVol)', min_value=0.0, step=1000.0, format='%.0f',
                        key='in_CortexVol')
        t2.number_input('Volumen intracraneal (ICV)', min_value=0.0, step=1000.0, format='%.0f', key='in_ICV')

        enviado = st.form_submit_button('Clasificar', type='primary', width='stretch', key='clasificar')

    if enviado:
        paciente = leer_formulario()
        errores, avisos = validar(paciente)
        if errores:
            st.session_state.pop('prediccion', None)
            for e in errores:
                aviso(e, 'error')
        else:
            st.session_state['prediccion'] = {
                'tabla': F.predecir(modelos, paciente, metadata),
                'avisos': avisos,
            }

    if 'prediccion' in st.session_state:
        pred = st.session_state['prediccion']
        tabla = pred['tabla']
        html('<h2 class="seccion" style="margin-top: 36px">Resultado de los tres modelos</h2>')
        for texto in pred['avisos']:
            aviso(texto, 'alerta')

        if tabla['clase'].nunique() == 1:
            aviso(f'Los tres modelos coinciden: <strong>{tabla["diagnostico"].iloc[0]}</strong>.', 'ok')
        else:
            a_favor = tabla[tabla['clase'] == 1]['modelo'].tolist()
            en_contra = tabla[tabla['clase'] == 0]['modelo'].tolist()
            uno = len(a_favor) == 1
            aviso(f'Los modelos no coinciden: {" y ".join(f"<strong>{m}</strong>" for m in a_favor)} '
                  f'{"clasifica" if uno else "clasifican"} como Dementia; {" y ".join(en_contra)}, como CN. '
                  'Es un caso cercano a la frontera de decisión.', 'alerta')

        filas = ''.join(
            f'<tr><td class="fuerte">{f["modelo"]}</td><td>{clase_html(f["diagnostico"])}</td>'
            f'<td class="der prob" data-modelo="{f["modelo"]}">{porcentaje(f["prob_dementia"])}</td>'
            f'<td class="der sec">{porcentaje(f["corte"])}</td></tr>'
            for _, f in tabla.iterrows())
        html('<div class="marco"><table class="tabla resultados"><thead><tr><th>Modelo</th><th>Clase predicha</th>'
             '<th class="der">Probabilidad de demencia</th><th class="der">Punto de corte</th></tr></thead>'
             f'<tbody>{filas}</tbody></table></div>')

        barras = ''
        for _, f in tabla.iterrows():
            color = COLOR_DEM if f['clase'] == 1 else COLOR_CN
            barras += (f'<div class="fila"><span>{f["modelo"]}</span><div class="pista">'
                       f'<div class="barra" style="width:{f["prob_dementia"] * 100:.2f}%;background:{color}"></div>'
                       f'<div class="corte" style="left:{f["corte"] * 100:.2f}%"></div></div>'
                       f'<span>{f["prob_dementia"] * 100:.1f} %</span></div>')
        corte_svm = tabla[tabla['modelo'] == 'SVM polinomial']['corte'].iloc[0]
        html(f"""
        <figure class="figura" style="margin-left:0;margin-right:0">
          <figcaption class="cabeza"><b>Probabilidad de demencia por modelo</b>
            <span class="leyenda"><span><i style="background:{COLOR_CN}"></i>CN</span>
            <span><i style="background:{COLOR_DEM}"></i>Dementia</span>
            <span><i class="linea"></i>Punto de corte</span></span></figcaption>
          <div class="barras">{barras}
            <div class="fila"><span></span><div class="eje"><span>0 %</span><span>25 %</span><span>50 %</span>
            <span>75 %</span><span>100 %</span></div><span></span></div>
          </div>
          <p class="nota" style="margin-top: 18px">El SVM clasifica como Dementia a partir de
          {corte_svm * 100:.1f} %; el Stacking y la Regresión Logística, a partir de 50 %.</p>
        </figure>
        """)


# ================================================================ 3. CARACTERÍSTICAS
elif seccion == 'Características':
    html("""
    <h1 class="titulo">Características (feature engineering)</h1>
    <p class="texto" style="margin-bottom: 28px">Antes de llegar a los modelos, los 18 datos capturados se
    transforman en 11 características. Es el mismo procedimiento que se usó en el notebook para entrenar
    (función <code>crear_features</code>).</p>
    <div class="tres">
      <div class="cifra"><span class="etiqueta">01</span><h3>Suma de hemisferios</h3>
        <p>Los volúmenes izquierdo y derecho de cada estructura están muy correlacionados. Sumarlos deja una sola
        variable por estructura.</p></div>
      <div class="cifra"><span class="etiqueta">02</span><h3>Promedio del grosor cortical</h3>
        <p>El grosor no es un volumen, así que no se suma: se promedian los dos hemisferios.</p></div>
      <div class="cifra"><span class="etiqueta">03</span><h3>Normalización por ICV</h3>
        <p>El tamaño del cerebro depende del tamaño de la cabeza. Dividir entre el volumen intracraneal (× 1,000)
        permite comparar a personas con cabezas de distinto tamaño. El ICV no entra como característica.</p></div>
    </div>
    """)
    icv = '(izq + der) / ICV × 1,000'
    transformaciones = [
        ('Edad', 'Edad', 'Sin cambio'), ('Escolaridad', 'Escolaridad', 'Sin cambio'),
        ('APOE4', 'Copias del alelo APOE4', 'Sin cambio'), ('Sexo', 'Sexo', 'Sin cambio'),
        ('Grosor cortical medio', 'Grosor izq., grosor der.', '(izq + der) / 2'),
        ('Hipocampo / ICV', 'Hipocampo izq. y der., ICV', icv),
        ('Entorrinal / ICV', 'Corteza entorrinal izq. y der., ICV', icv),
        ('Amígdala / ICV', 'Amígdala izq. y der., ICV', icv),
        ('Ventrículos laterales / ICV', 'Ventrículo lateral izq. y der., ICV', icv),
        ('Ventrículo lat. inferior / ICV', 'Ventrículo lateral inferior izq. y der., ICV', icv),
        ('Corteza / ICV', 'CortexVol, ICV', 'CortexVol / ICV × 1,000'),
    ]
    filas = ''.join(f'<tr><td class="sec">{i}</td><td class="fuerte">{c}</td><td>{d}</td>'
                    f'<td class="{"sec" if calc == "Sin cambio" else ""}">{calc}</td></tr>'
                    for i, (c, d, calc) in enumerate(transformaciones, 1))
    html('<h2 class="seccion" style="margin-top: 36px">De los datos capturados a las características</h2>'
         '<div class="marco"><table class="tabla compacta"><thead><tr><th>#</th><th>Característica</th>'
         f'<th>Datos que usa</th><th>Cálculo</th></tr></thead><tbody>{filas}</tbody></table></div>')

    entrada = F.preparar_entrada(leer_formulario(), metadata).iloc[0]
    unidades = {'Edad': 'años', 'Escolaridad': 'años', 'APOE4': 'copias', 'Sexo': '—', 'Grosor_medio': 'mm'}
    filas = ''
    for col, valor in entrada.items():
        if isinstance(valor, str):
            texto = valor
        elif pd.isna(valor):
            texto = 'Desconocido'
        elif col in ('Edad', 'Escolaridad', 'APOE4'):
            texto = f'{valor:.0f}'
        else:
            texto = f'{valor:,.3f}'
        filas += (f'<tr><td class="fuerte">{metadata["etiquetas_variables"][col]}</td><td class="der">{texto}</td>'
                  f'<td class="sec">{unidades.get(col, "mm³ por cada 1,000 mm³ de ICV")}</td></tr>')
    origen = st.session_state.get('origen', '')
    html(f'<div class="fila-titulo"><h2 class="seccion">Características que reciben los modelos</h2>'
         f'<span>Datos cargados: {origen}</span></div>'
         '<div class="marco"><table class="tabla compacta"><thead><tr><th>Característica</th>'
         f'<th class="der">Valor</th><th>Unidad</th></tr></thead><tbody>{filas}</tbody></table></div>')


# ================================================================ 4. COMPARACIÓN
elif seccion == 'Comparación':
    info_modelos = list(resultados['modelos'].values())
    html('<h1 class="titulo">Comparación de modelos</h1>'
         '<p class="texto">Resultados del notebook del proyecto. Los modelos se eligieron con validación cruzada '
         'en Train; el conjunto de prueba (Test) se usó una sola vez, al final.</p>')

    # Métricas de Test: las de la matriz se calculan exactas; ROC-AUC viene del notebook
    nombres_metricas = ['Balanced Acc.', 'Recall', 'Especificidad', 'Precision', 'F1', 'ROC-AUC', 'Accuracy']
    tabla_test = pd.DataFrame(
        [{**metricas_desde_matriz(i['matriz_confusion']), 'ROC-AUC': i['test']['ROC-AUC']} for i in info_modelos],
        index=[i['nombre'] for i in info_modelos])[nombres_metricas]
    redondeada = tabla_test.map(redondear3)
    filas = ''
    for modelo, fila in redondeada.iterrows():
        celdas = ''.join(f'<td class="der{" mejor" if v == redondeada[m].max() else ""}">{v:.3f}</td>'
                         for m, v in fila.items())
        filas += (f'<tr><td class="fuerte"><i class="muestra" style="background:{COLOR_MODELO[modelo]}"></i>'
                  f'{modelo}</td>{celdas}</tr>')
    encabezados = ''.join(f'<th class="der">{m}</th>' for m in nombres_metricas)
    html('<div class="fila-titulo"><h2 class="seccion">Resultados en Test</h2>'
         '<span>325 pacientes: 245 CN y 80 Dementia (20 % de la base)</span></div>'
         f'<div class="marco"><table class="tabla compacta"><thead><tr><th>Modelo</th>{encabezados}</tr></thead>'
         f'<tbody>{filas}</tbody></table></div>'
         '<p class="nota">En negritas, el valor más alto de cada métrica.</p>')

    with st.container(key='grafica'):
        leyenda = ''.join(f'<span><i style="background:{c}"></i>{m}</span>' for m, c in COLOR_MODELO.items())
        html(f'<div class="figura" style="border:0;padding:0;margin:0"><div class="cabeza" style="margin:0">'
             f'<b>Métricas en Test por modelo</b><span class="leyenda">{leyenda}</span></div></div>')
        largo = tabla_test.reset_index(names='Modelo').melt('Modelo', var_name='Métrica', value_name='Valor')
        orden_modelos = list(COLOR_MODELO)
        barras = alt.Chart(largo).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X('Métrica:N', sort=nombres_metricas, title=None,
                    axis=alt.Axis(labelAngle=0, labelFontSize=13, labelColor='#36454D', ticks=False,
                                  domainColor='#C9C2B4', labelPadding=10)),
            xOffset=alt.XOffset('Modelo:N', sort=orden_modelos),
            y=alt.Y('Valor:Q', scale=alt.Scale(domain=[0.7, 1.0], clamp=True), title=None,
                    axis=alt.Axis(values=[0.7, 0.8, 0.9, 1.0], format='.2f', labelColor=TEXTO_SEC,
                                  labelFontSize=12, gridColor=BORDE, gridDash=[4, 4], domain=False, ticks=False)),
            color=alt.Color('Modelo:N', sort=orden_modelos, legend=None,
                            scale=alt.Scale(domain=orden_modelos, range=list(COLOR_MODELO.values()))),
            tooltip=['Modelo', 'Métrica', alt.Tooltip('Valor:Q', format='.3f')],
        ).properties(height=280, background=SUPERFICIE).configure(font='Montserrat').configure_view(strokeWidth=0)
        st.altair_chart(barras, width='stretch', theme=None)
        html('<p class="nota" style="margin:0">Eje vertical de 0.70 a 1.00.</p>')

    def color_celda(p):
        """Escala de #EEF4F4 (0 %) a #0E4F5C (100 %) según el porcentaje de la fila."""
        a, b = np.array([0xEE, 0xF4, 0xF4]), np.array([0x0E, 0x4F, 0x5C])
        r, g, bl = (a + (b - a) * p).round().astype(int)
        return f'#{r:02X}{g:02X}{bl:02X}', '#FFFFFF' if p > 0.5 else TEXTO

    tarjetas = ''
    for info in info_modelos:
        mc = info['matriz_confusion']
        celdas = ''
        for etiqueta, a, b in [('Real CN', mc['TN'], mc['FP']), ('Real Dementia', mc['FN'], mc['TP'])]:
            celdas += f'<span class="eti">{etiqueta}</span>'
            for v in (a, b):
                fondo, texto = color_celda(v / (a + b))
                celdas += (f'<div class="celda" style="background:{fondo};color:{texto}"><strong>{v}</strong>'
                           f'<span>{v / (a + b) * 100:.1f} %</span></div>')
        tarjetas += (f'<div class="matriz-tarjeta"><b>{info["nombre"]}</b><div class="matriz"><span></span>'
                     '<span class="eti col">Pred. CN</span><span class="eti col">Pred. Dementia</span>'
                     f'{celdas}</div></div>')
    html('<div class="fila-titulo" style="margin-top: 40px"><h2 class="seccion">Matrices de confusión en Test</h2>'
         '<span>El color indica el porcentaje de cada fila (clase real)</span></div>'
         f'<div class="tres">{tarjetas}</div>')

    mejor_ba = max(i['cv_repetida']['media'] for i in info_modelos)
    mejor_rec = max(i['cv_repetida']['recall'] for i in info_modelos)
    config_corta = {'Stacking': 'Logística + SVM polinomial + Gradient Boosting; metamodelo: Regresión Logística '
                                'balanceada',
                    'SVM polinomial': 'grado 2, coef0 = 1, C = 1, class_weight = balanced',
                    'Regresión Logística': 'L2, C = 0.01, class_weight = balanced'}
    filas = ''.join(
        f'<tr><td class="fuerte">{i["nombre"]}</td>'
        f'<td class="der{" mejor" if i["cv_repetida"]["media"] == mejor_ba else ""}">{i["cv_repetida"]["media"]:.4f}</td>'
        f'<td class="der">{i["cv_repetida"]["desv"]:.4f}</td>'
        f'<td class="der{" mejor" if i["cv_repetida"]["recall"] == mejor_rec else ""}">{i["cv_repetida"]["recall"]:.4f}</td>'
        f'<td class="sec">{config_corta[i["nombre"]]}</td></tr>' for i in info_modelos)
    html("""
    <h2 class="seccion" style="margin-top: 40px">Validación cruzada repetida en Train</h2>
    <p class="texto">Criterio con el que se eligieron los modelos. Train (1,299 pacientes) se dividió en 5 folds
    estratificados, y el proceso se repitió 5 veces con particiones distintas: 25 evaluaciones por modelo. La tabla
    muestra el promedio y la desviación estándar de esas 25.</p>
    """)
    html('<div class="marco"><table class="tabla compacta"><thead><tr><th>Modelo</th>'
         '<th class="der">Balanced Acc. media</th><th class="der">Desv. estándar</th>'
         f'<th class="der">Recall media</th><th>Configuración</th></tr></thead><tbody>{filas}</tbody></table></div>')
    html('<p class="texto" style="margin-top: 14px">En validación cruzada los tres modelos están prácticamente '
         'empatados: la diferencia entre el Stacking y la Regresión Logística (0.003) es mucho menor que la '
         'variación entre particiones (≈ 0.025). En Test el Stacking queda 0.019 arriba de la Regresión Logística '
         '(IC 95 %: 0.004 a 0.037).</p>')


# ================================================================ 5. INTERPRETACIÓN
else:
    html(f"""
    <h1 class="titulo">Interpretación y limitaciones</h1>
    <h2 class="seccion">Clase predicha</h2>
    <div class="dos-tarjetas">
      <div class="cifra">{clase_html('CN').replace('>CN<', '>CN (cognitivamente normal)<')}
        <p>Las medidas de la persona se parecen a las de los participantes de ADNI sin deterioro cognitivo.</p></div>
      <div class="cifra">{clase_html('Dementia')}
        <p>Las medidas se parecen a las de los participantes con demencia: menor volumen de hipocampo, corteza
        entorrinal y amígdala, y ventrículos más grandes.</p></div>
    </div>
    <h2 class="seccion" style="margin-top: 40px">Probabilidades</h2>
    <ul class="lista">
      <li>Cada modelo entrega un valor de 0 % a 100 % y clasifica como Dementia cuando supera su punto de corte:
      50 % en el Stacking y la Regresión Logística, 25.8 % en el SVM.</li>
      <li>El corte del SVM es distinto porque su probabilidad se calcula aparte (calibración de Platt) y no incluye
      el peso extra que se le dio a la clase Dementia (class_weight = balanced). Por eso queda cerca de la
      proporción real de demencia en los datos (24.6 %).</li>
      <li>Valores cercanos al corte indican un caso limítrofe, donde es más probable que los modelos no
      coincidan.</li>
    </ul>
    <h2 class="seccion" style="margin-top: 40px">Confiabilidad del Stacking en Test</h2>
    <div class="cifras dos">
      <div class="cifra"><span class="rotulo">Demencia detectada (recall)</span><span class="valor">70 de 80</span>
        <span class="detalle">87.5 %</span></div>
      <div class="cifra"><span class="rotulo">CN bien clasificados (especificidad)</span>
        <span class="valor">227 de 245</span><span class="detalle">92.7 %</span></div>
    </div>
    <p class="texto" style="margin-top: 14px">Aproximadamente 1 de cada 8 casos de demencia no se detecta y 1 de
    cada 14 personas sanas recibe una falsa alarma.</p>
    <h2 class="seccion" style="margin-top: 40px">Limitaciones</h2>
    <ol class="lista">
      <li>Solo distingue CN de Dementia; no incluye el deterioro cognitivo leve (MCI).</li>
      <li>Los volúmenes deben venir de FreeSurfer y estar armonizados como en ADNI.</li>
      <li>Los participantes de ADNI no representan a la población general (Estados Unidos y Canadá, escolaridad
      alta).</li>
      <li>Usa datos de una sola visita, sin pruebas cognitivas ni otros biomarcadores.</li>
      <li>Con valores fuera del rango de entrenamiento la predicción es menos confiable.</li>
    </ol>
    """)
