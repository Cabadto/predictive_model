
"""
# Aplicación Avanzada de Predicción de Tiempo de Recuperación Post-Cirugía

**Experto en Algoritmos de Redes Neuronales y Modelos Híbridos**

Esta aplicación utiliza modelos avanzados de deep learning (CNN y modelos híbridos) para predecir y clasificar el tiempo de recuperación de pacientes después de una cirugía, con validación estadística robusta y generación de reportes completos.
"""

# =========================
#        IMPORTS
# =========================
import os
import io
import base64
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import label_binarize

from fpdf import FPDF

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    matthews_corrcoef, classification_report, auc, precision_score, recall_score, f1_score, cohen_kappa_score, balanced_accuracy_score
)

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout,
    BatchNormalization, Input, concatenate, Attention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from PIL import Image
from statsmodels.stats.contingency_tables import mcnemar
# =========================
#  RUTAS LOCALES
# =========================
DATA_PATH = "./data"
MODEL_PATH = "./modelos"

# Mostrar contenido para verificación
st.write("Ruta de datos:", DATA_PATH)
st.write("Archivos en data:", os.listdir(DATA_PATH))

def traducir(key):
    lenguaje = st.session_state.get("lang", "es")
    return traducciones.get(key, {}).get(lenguaje, key)

archivos_requeridos = {
    'ADMISSIONS': 'ADMISSIONS.csv',
    'PATIENTS': 'PATIENTS.csv',
    'PROCEDURES': 'PROCEDURES_ICD.csv',
    'DIAGNOSES': 'DIAGNOSES_ICD.csv'
}

dfs = {}
for name, filename in archivos_requeridos.items():
    filepath = os.path.join(DATA_PATH, filename)
    try:
        dfs[name] = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
        st.write(f"✅ {filename} cargado correctamente (UTF-8)")
    except UnicodeDecodeError:
        try:
            dfs[name] = pd.read_csv(filepath, encoding='latin1', low_memory=False)
            st.write(f"✅ {filename} cargado correctamente (Latin-1)")
        except Exception as e:
            st.error(f"❌ Error al cargar {filename}: {str(e)}")
            dfs[name] = None
    except Exception as e:
        st.error(f"❌ Error al cargar {filename}: {str(e)}")
        dfs[name] = None

# =========================
#  UTILIDADES DE PREPROCESO
# =========================
def map_icd_a_cirugia(icd_code):
    """Mapea rangos de ICD9 a macro-tipos de cirugía (aprox)."""
    try:
        codigo = float(str(icd_code).split('.')[0])  # soporta "9955", "45.13", etc.
        if 30 <= codigo <= 34: return 'Cardiovascular'
        elif 35 <= codigo <= 39: return 'Torácica'
        elif 40 <= codigo <= 41: return 'Hematolinfática'
        elif 42 <= codigo <= 54: return 'Digestiva'
        elif 76 <= codigo <= 84: return 'Ortopédica'
        else: return 'Otros'
    except Exception:
        return 'Otros'

def seleccion_segura(df, candidates, default=None):
    """Devuelve la primera columna que exista en df de la lista candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return default

def asegurar_serie_temporal(df, seed=42, length=60):
    if df.empty:
        st.warning("DataFrame vacío: no se pueden generar series temporales.")
        df['Serie_Temporal'] = []
        return df

    if 'Serie_Temporal' in df.columns:
        try:
            sample = df['Serie_Temporal'].iloc[0]
            _ = len(sample)
            return df
        except Exception:
            pass  # Si no cumple, regeneramos.

    rng = np.random.default_rng(seed)
    L = length

    edad = df.get('EDAD', pd.Series(0, index=df.index)).fillna(df['EDAD'].median() if 'EDAD' in df else 0).to_numpy()
    comorb = df.get('COMORBILIDADES', pd.Series(0, index=df.index)).fillna(0).to_numpy()
    rec = df.get('TIEMPO_RECUPERACION', pd.Series(0, index=df.index)).fillna(df['TIEMPO_RECUPERACION'].median() if 'TIEMPO_RECUPERACION' in df else 0).to_numpy()

    def normalizar_seguro(arr):
        if len(arr) == 0:
            return np.array([])
        min_val, max_val = np.nanmin(arr), np.nanmax(arr)
        if max_val - min_val == 0:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    edad_n, com_n, rec_n = normalizar_seguro(edad), normalizar_seguro(comorb), normalizar_seguro(rec)

    series = []
    t = np.linspace(0, 2*np.pi, L)
    for i in range(len(df)):
        base = 0.4*edad_n[i] + 0.3*com_n[i] + 0.3*rec_n[i]
        wave = 0.3*np.sin(t*(1 + 0.5*com_n[i])) + 0.2*np.cos(t*(1 + rec_n[i]))
        noise = rng.normal(0, 0.05 + 0.1*com_n[i], size=L)
        s = base + wave + noise
        s_min, s_max = s.min(), s.max()
        s = np.zeros_like(s) if s_max - s_min == 0 else (s - s_min) / (s_max - s_min)
        series.append(s.astype(float).tolist())

    df = df.copy()
    df['Serie_Temporal'] = series
    return df

# =========================
#  PREPROCESAMIENTO PRINCIPAL
# =========================
@st.cache_data(show_spinner=False)
def cargar_y_preprocesar_datos():
    # Validar archivos esenciales
    global dfs
    if dfs['ADMISSIONS'] is None or dfs['ADMISSIONS'].empty or \
    dfs['PATIENTS'] is None or dfs['PATIENTS'].empty:
      st.error("Error: faltan ADMISSIONS.csv o PATIENTS.csv")
      return None

    # Copiar dataframes
    adm = dfs['ADMISSIONS'].copy()
    pat = dfs['PATIENTS'].copy()
    pro = dfs['PROCEDURES'].copy() if dfs['PROCEDURES'] is not None else None
    dia = dfs['DIAGNOSES'].copy() if dfs['DIAGNOSES'] is not None else None

    # <-- Añade esto -->
    st.write("Filas en ADMISSIONS:", len(adm))
    st.write("Filas en PATIENTS:", len(pat))
    if pro is not None:
        st.write("Filas en PROCEDURES:", len(pro))
    if dia is not None:
        st.write("Filas en DIAGNOSES:", len(dia))
    # Merge ADMISSIONS + PATIENTS
    df = pd.merge(adm, pat, on='SUBJECT_ID', how='left', suffixes=('_AD','_PT'))
    st.write("Filas después de merge ADMISSIONS + PATIENTS:", len(df))
    # Merge con PROCEDURES
    if pro is not None:
        df = pd.merge(df, pro[['SUBJECT_ID','HADM_ID','ICD9_CODE','SEQ_NUM','ROW_ID']],
                      on=['SUBJECT_ID','HADM_ID'], how='left')
    st.write("Filas después de merge con PROCEDURES:", len(df))
    # Convertir fechas
    for col in ['ADMITTIME','DISCHTIME','DOB']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Filtrar fechas no plausibles
    fecha_min = pd.Timestamp('1900-01-01')
    fecha_max = pd.Timestamp('2500-01-01')  # suficiente para tus datos
    for col in ['ADMITTIME','DISCHTIME','DOB']:
        if col in df.columns:
            df = df[df[col].between(fecha_min, fecha_max)]
            st.write(f"Filas después de filtrar {col}:", len(df))

    # Evitar recuperaciones negativas
    if 'DISCHTIME' in df.columns and 'ADMITTIME' in df.columns:
        df = df[df['DISCHTIME'] >= df['ADMITTIME']]

    # Cálculo EDAD seguro usando años
    if {'ADMITTIME','DOB'}.issubset(df.columns):
        df['ADMIT_YEAR'] = df['ADMITTIME'].dt.year
        df['DOB_YEAR'] = df['DOB'].dt.year
        df['EDAD'] = (df['ADMIT_YEAR'] - df['DOB_YEAR']).clip(0,120).astype('Int64')
        # <-- Añade esto -->
        st.write("Edad mínima y máxima:", df['EDAD'].min(), df['EDAD'].max())
        st.write("Filas con EDAD válida:", df['EDAD'].notna().sum())
    else:
        df['EDAD'] = pd.Series([pd.NA]*len(df), dtype='Int64')

    # Cálculo TIEMPO_RECUPERACION seguro en días
    if {'DISCHTIME','ADMITTIME'}.issubset(df.columns):
        df['TIEMPO_RECUPERACION'] = ((df['DISCHTIME'] - df['ADMITTIME']).dt.days).clip(0)
        st.write("Tiempo de recuperación mínima y máxima:", df['TIEMPO_RECUPERACION'].min(), df['TIEMPO_RECUPERACION'].max())
    else:
        df['TIEMPO_RECUPERACION'] = pd.Series([np.nan]*len(df), dtype='float')

    # Tipo de cirugía
    if 'ICD9_CODE' in df.columns:
        df['TIPO_CIRUGIA'] = df['ICD9_CODE'].apply(map_icd_a_cirugia)
    else:
        df['TIPO_CIRUGIA'] = 'Otros'

    # Comorbilidades
    if dia is not None and {'SUBJECT_ID','HADM_ID'}.issubset(dia.columns):
        comorb = dia.groupby(['SUBJECT_ID','HADM_ID']).size().reset_index(name='COMORBILIDADES')
        df = pd.merge(df, comorb, on=['SUBJECT_ID','HADM_ID'], how='left')
    else:
        df['COMORBILIDADES'] = np.nan

    # Clase de recuperación
    bins = [-0.1, 7, 14, 30, np.inf]
    labels = ['Rápida', 'Media', 'Lenta', 'Muy Lenta']
    df['CLASE_RECUPERACION'] = pd.cut(df['TIEMPO_RECUPERACION'], bins=bins, labels=labels)

    # Serie temporal sintética
    df = asegurar_serie_temporal(df, seed=42, length=60)

    # Sexo
    if 'GENDER' in df.columns:
        df['GENDER'] = df['GENDER'].astype(str).fillna('Unknown')
    else:
        df['GENDER'] = 'Unknown'

    # Reset index final
    df = df.reset_index(drop=True)

    # Debug corto
    print("Columnas disponibles:", df.columns.tolist())
    print(df.head())

    return df

# Carga global (para toda la app)
# --- Carga de archivos ---

df = cargar_y_preprocesar_datos()
if df is None or df.empty:
    st.warning("No se pudieron procesar los datos. Verifica los archivos subidos.")
    st.stop()

st.success("Datos cargados y procesados correctamente")
st.dataframe(df.head())
def evaluar_modelo(model, X_test, y_test, name):
    """
    Evalúa un modelo multi-clase, generando:
    - Matriz de confusión (mapa de calor)
    - Reporte de clasificación
    - Curva ROC y AUC (multi-clase)
    """
    numero_clases = y_test.shape[1] if len(y_test.shape) > 1 else len(np.unique(y_test))

    # Predicciones
    y_pred_prob = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)

    # Convertir y_test a clases enteras si es one-hot
    y_test_classes = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test

    # Matriz de confusión
    cm = confusion_matrix(y_test_classes, y_pred_classes)

    # Reporte de clasificación
    report = classification_report(y_test_classes, y_pred_classes, output_dict=True)

    # MCC
    mcc = matthews_corrcoef(y_test_classes, y_pred_classes)

    estadigrafos = calcular_estadigrafos(y_test_classes, y_pred_classes)

    # Curva ROC / AUC (multi-clase)
    try:
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test_classes, classes=range(numero_clases))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(numero_clases):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Promedio macro
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(numero_clases)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(numero_clases):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= numero_clases
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    except Exception as e:
        print("Error calculando ROC:", e)
        fpr = tpr = roc_auc = None

    # Accuracy
    accuracy = np.mean(y_pred_classes == y_test_classes)

    # Gráficos
    plt.figure(figsize=(15, 5))

    # Matriz de confusión
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {name}')
    plt.xlabel('Predicho')
    plt.ylabel('Real')

    # Curva ROC
    plt.subplot(1, 2, 2)
    if fpr is not None:
        plt.plot(fpr["macro"], tpr["macro"], color='darkorange',
                 lw=2, label=f'Macro ROC curve (area = {roc_auc["macro"]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Curva ROC - {name}')
        plt.legend(loc="lower right")

    # Guardar figura en memoria
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()

    return {
        'name': name,
        'model': model,
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_auc': roc_auc,
        'evaluation_image': img,
        'accuracy': accuracy,
        'mcc': mcc,
        'evaluation_image': img,
        'precision': estadigrafos['Precision (macro)'],
        'recall': estadigrafos['Recall (macro)'],
        'f1': estadigrafos['F1-score (macro)'],
        'kappa': estadigrafos['Kappa de Cohen'],
    }

# =========================
#    MODELOS DE DEEP LEARNING
# =========================
def crear_modelo_cnn(forma_entrada, numero_clases, learning_rate=0.001):
    modelo = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=forma_entrada, padding='same'),
        BatchNormalization(),
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(numero_clases, activation='softmax')
    ])
    modelo.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return modelo

def crear_modelo_hibrido(forma_tabular, forma_serie_temporal, numero_clases):
    entrada_tabular = Input(shape=(forma_tabular,))
    x = Dense(64, activation='relu')(entrada_tabular)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    ts_input = Input(shape=forma_serie_temporal)
    y = Conv1D(64, 3, activation='relu', padding='same')(ts_input)
    y = MaxPooling1D(2)(y)
    y = Conv1D(128, 3, activation='relu', padding='same')(y)
    y = MaxPooling1D(2)(y)
    y = Flatten()(y)

    combinado = concatenate([x, y])
    z = Dense(128, activation='relu')(combinado)
    z = Dropout(0.5)(z)
    salida = Dense(numero_clases, activation='softmax')(z)

    modelo = Model(inputs=[entrada_tabular, ts_input], outputs=salida)
    modelo.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return modelo

def crear_modelo_cnn_lstm(forma_serie_temporal, numero_clases):
    modelo = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=forma_serie_temporal, padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(numero_clases, activation='softmax')
    ])
    modelo.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return modelo

def crear_modelo_hibrido_con_atencion(tabular_shape, forma_serie_temporal, numero_clases):
    entrada_tabular = Input(shape=(tabular_shape,))
    x = Dense(64, activation='relu')(entrada_tabular)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    ts_input = Input(shape=forma_serie_temporal)
    y = Conv1D(64, 3, activation='relu', padding='same')(ts_input)
    y = MaxPooling1D(2)(y)
    y = Conv1D(128, 3, activation='relu', padding='same')(y)
    y = MaxPooling1D(2)(y)

    attn = Attention()([y, y])
    y = concatenate([y, attn])
    y = Flatten()(y)

    combinado = concatenate([x, y])
    z = Dense(128, activation='relu')(combinado)
    z = Dropout(0.5)(z)
    output = Dense(numero_clases, activation='softmax')(z)

    modelo = Model(inputs=[entrada_tabular, ts_input], outputs=output)
    modelo.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return modelo

# =========================
#     PREP. DATOS MODELOS
# =========================
def preparar_datos_para_cnn(input_df):
    dfm = input_df.copy()

    # Target
    y = dfm['CLASE_RECUPERACION'].astype(str)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    # Caso extremo: si todo es 'nan', LabelEncoder falla -> filtramos NaN
    valid_mask = ~pd.isna(dfm['CLASE_RECUPERACION'])
    dfm = dfm[valid_mask].reset_index(drop=True)
    y = dfm['CLASE_RECUPERACION'].astype(str)
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # Features tabulares
    numeric_features = []
    for col in ['EDAD', 'COMORBILIDADES', 'TIEMPO_RECUPERACION']:
        if col in dfm.columns:
            numeric_features.append(col)
    # Si por algún motivo faltaran, crea al menos una
    if not numeric_features:
        dfm['EDAD'] = dfm.get('EDAD', pd.Series([0]*len(dfm)))
        numeric_features = ['EDAD']

    categorical_features = []
    for col in ['GENDER', 'ICD9_CODE', 'TIPO_CIRUGIA']:
        if col in dfm.columns:
            categorical_features.append(col)

    # Matriz tabular
    X_tabular = dfm[numeric_features + categorical_features].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )

    X_tabular_processed = preprocessor.fit_transform(X_tabular)

    # Series temporales
    # Garantizado por asegurar_serie_temporal
    X_time_series = np.array(dfm['Serie_Temporal'].tolist())
    if X_time_series.ndim == 2:
        X_time_series = X_time_series[..., np.newaxis]  # (n, L, 1)

    # Split
    X_train_tab, X_test_tab, X_train_ts, X_test_ts, y_train, y_test = train_test_split(
        X_tabular_processed, X_time_series, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
    )

    return (X_train_tab, X_test_tab, X_train_ts, X_test_ts, y_train, y_test, preprocessor, le)

# =========================
#   ENTRENAMIENTO/EVALUACIÓN
# =========================
from sklearn.model_selection import KFold
from scipy.stats import f_oneway, friedmanchisquare

import scikit_posthocs as sp

def entrenar_y_evaluar_modelos():

    # Preparar datos
    X_train_tab, X_test_tab, X_train_ts, X_test_ts, y_train, y_test, preprocessor, le = preparar_datos_para_cnn(df)
    st.header("Entrenamiento de Modelos CNN e Híbridos")

    forma_serie_temporal = (X_train_ts.shape[1], X_train_ts.shape[2])
    numero_clases = y_train.shape[1] if len(y_train.shape) > 1 else len(np.unique(y_train))
    tabular_shape = X_train_tab.shape[1]

    # =====================
    # Carpeta de modelos local
    # =====================
    MODEL_DIR = "./modelos"
    os.makedirs(MODEL_DIR, exist_ok=True)

    configuracion_modelos = {
        'CNN_1D_Simple': {'builder': lambda: crear_modelo_cnn(forma_serie_temporal, numero_clases), 'data': 'ts_only'},
        'CNN_1D_Profunda': {'builder': lambda: crear_modelo_cnn(forma_serie_temporal, numero_clases), 'data': 'ts_only'},
        'CNN_LSTM': {'builder': lambda: crear_modelo_cnn_lstm(forma_serie_temporal, numero_clases), 'data': 'ts_only'},
        'Hibrido_CNN_MLP': {'builder': lambda: crear_modelo_hibrido(tabular_shape, forma_serie_temporal, numero_clases), 'data': 'both'},
        'Hibrido_CNN_Attention': {'builder': lambda: crear_modelo_hibrido_con_atencion(tabular_shape, forma_serie_temporal, numero_clases), 'data': 'both'}
    }

    results = []
    trained_models = {}
    history_logs = {}
    eval_results = {}
    reentrenar = st.session_state.get("reentrenar", False)

    for model_name, config in configuracion_modelos.items():
        st.write(f"## {model_name}")
        model_path = os.path.join(MODEL_DIR, f"{model_name}.h5")

        if os.path.exists(model_path) and not reentrenar:
            st.write(f"🔄 Cargando {model_name} desde local...")
            model = tf.keras.models.load_model(model_path)
            training_time = 0
        else:
            st.write(f"🚀 Entrenando {model_name}...")
            model = config['builder']()

            callbacks = [
                EarlyStopping(patience=8, monitor='val_loss', restore_best_weights=True),
                ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')
            ]

            start_time = time.time()
            if config['data'] == 'ts_only':
                history = model.fit(
                    X_train_ts, y_train,
                    validation_data=(X_test_ts, y_test),
                    epochs=30, batch_size=64, verbose=0, callbacks=callbacks
                )
            else:
                history = model.fit(
                    [X_train_tab, X_train_ts], y_train,
                    validation_data=([X_test_tab, X_test_ts], y_test),
                    epochs=30, batch_size=64, verbose=0, callbacks=callbacks
                )
            training_time = time.time() - start_time
            history_logs[model_name] = history.history
            model.save(model_path)
            st.success(f"💾 Modelo guardado en: {model_path}")

        # Evaluación normal
        X_eval = X_test_ts if config['data'] == 'ts_only' else [X_test_tab, X_test_ts]
        eval_result = evaluar_modelo(model, X_eval, y_test, model_name)

        # Predicciones
        if config['data'] == 'ts_only':
            y_pred = model.predict(X_test_ts, verbose=0)
        else:
            y_pred = model.predict([X_test_tab, X_test_ts], verbose=0)

        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        mcc_value = matthews_corrcoef(y_true_classes, y_pred_classes)

       
        table = confusion_matrix(y_true_classes, y_pred_classes)

        result = mcnemar(table)
        st.write(f"McNemar test: statistic={result.statistic}, p-value={result.pvalue}")

        results.append({
            'Modelo': model_name,
            'Accuracy': eval_result['accuracy'],
            'AUC': eval_result['roc_auc']['macro'] if eval_result['roc_auc'] else 0,
            'Matriz_Confusion': eval_result['confusion_matrix'],
            'Reporte_Clasificacion': eval_result['classification_report'],
            'Tiempo_Entrenamiento': training_time,
            'MCC': mcc_value,
            'Precision': eval_result['precision'],
            'Recall': eval_result['recall'],
            'F1': eval_result['f1'],
            'Kappa': eval_result['kappa'],
            
        })

        trained_models[model_name] = model
        eval_results[model_name] = eval_result

        st.success(f"{model_name} listo en {training_time:.2f} s")
        st.image(eval_result['evaluation_image'], caption=f"Matriz de Confusión + Curva ROC de {model_name}")


    results_df = pd.DataFrame(results).sort_values(by='AUC', ascending=False).reset_index(drop=True)



    return results_df, trained_models, history_logs, eval_results, le, y_test


# =========================
#   RESULTADOS Y ESTADÍSTICA
# =========================
def theil_u(y_true, y_pred):
    """Calcula el coeficiente de Theil."""
    return np.sqrt(np.sum((y_true - y_pred)**2) / np.sum(y_true**2))

def diebold_mariano(y_true, y_pred1, y_pred2):
    """Prueba de Diebold-Mariano para comparar dos modelos."""
    d = (y_true - y_pred1)**2 - (y_true - y_pred2)**2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    DM_stat = mean_d / np.sqrt(var_d / len(d))
    from scipy.stats import t
    p_value = 2 * (1 - t.cdf(np.abs(DM_stat), df=len(d)-1))
    return DM_stat, p_value


def mostrar_resultados_estadisticas(results, trained_models, le, y_test):
    st.header("📊 Comparación de Modelos y Estadísticas")

    # Crear DataFrame de resultados
    results_df = pd.DataFrame(results)
    results_df_no_cm = results_df.drop(columns=['Matriz_Confusion'], errors='ignore')
    st.subheader("Tabla de Resultados")
    st.dataframe(results_df_no_cm.sort_values(by='AUC', ascending=False))

     # Barra de AUC
    fig = px.bar(results_df_no_cm, x='Modelo', y='AUC', title='Comparación de AUC entre Modelos')
    st.plotly_chart(fig, use_container_width=True)
    # -------------------------
    # 6. Estadísticos descriptivos
    # -------------------------
    st.subheader("Estadísticos Descriptivos de Métricas")
    st.write(results_df_no_cm[['Accuracy','AUC','F1','MCC']].describe())

    # -------------------------
    # 7. Gráficos de métricas
    # -------------------------
    st.subheader("Distribución de Métricas entre Modelos")
    for metrica in ['Accuracy','AUC','F1','MCC']:
        fig = px.box(results_df_no_cm, y=metrica, points="all", title=f"Distribución de {metrica}")
        st.plotly_chart(fig)

    # -------------------------
    # 11. Matriz de Confusión del Mejor Modelo
    # -------------------------
    st.header("Matrices de Confusión")
    model_names = results_df['Modelo'].tolist()
    selected_model = st.selectbox("Seleccione modelo", model_names)

    selected_result_row = results_df[results_df['Modelo'] == selected_model].iloc[0]
    cm = selected_result_row.get('Matriz_Confusion', None)
    selected_result = st.session_state['eval_results'][selected_model]

    classes = le.classes_
    if cm is not None:
        # Mostrar heatmap interactivo
        fig_cm = px.imshow(cm, labels=dict(x="Predicho", y="Real", color="Cantidad"),
                           x=classes, y=classes, title=f"Matriz de Confusión - {selected_model}",
                           text_auto=True)
        st.plotly_chart(fig_cm, use_container_width=True)

        # Estadísticos derivados
        st.subheader(f"Estadísticos de la Matriz de Confusión - {selected_model}")
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        st.write("TP:", TP, "FP:", FP, "FN:", FN)

    # -------------------------
    # 12. Validación estadística
    # -------------------------
    st.subheader("Validaciones Estadísticas - Clasificación")
    y_true_classes = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(selected_result['y_pred'], axis=1)
    mcc_value = matthews_corrcoef(y_true_classes, y_pred_classes)
    st.write(f"Coeficiente de Matthews: {mcc_value:.3f}")

    # McNemar test
    from statsmodels.stats.contingency_tables import mcnemar
    table = confusion_matrix(y_true_classes, y_pred_classes)
    result = mcnemar(table)
    st.write(f"McNemar test: statistic={result.statistic}, p-value={result.pvalue:.3f}")

    # -------------------------
    # Curvas ROC y AUC
    # -------------------------
    st.header("Curvas ROC")
    X_train_tab, X_test_tab, X_train_ts, X_test_ts, _, _, _, _ = preparar_datos_para_cnn(df)
    model = trained_models[selected_model]

    if selected_model.startswith('Hibrido'):
        y_pred = model.predict([X_test_tab, X_test_ts], verbose=0)
    else:
        y_pred = model.predict(X_test_ts, verbose=0)

    fig_roc = go.Figure()
    for i in range(y_test.shape[1]):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_pred[:, i])
        auc_i = roc_auc_score(y_test[:, i], y_pred[:, i])
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                     name=f"{le.classes_[i]} (AUC={auc_i:.2f})"))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Aleatorio',
                                 line=dict(dash='dash')))
    fig_roc.update_layout(title=f"Curva ROC Multiclase - {selected_model}",
                          xaxis_title="FPR", yaxis_title="TPR")
    st.plotly_chart(fig_roc, use_container_width=True)

    # -------------------------
    # 13. Series Temporales (Theil y Diebold-Mariano)
    # -------------------------
    st.header("Validación de Series Temporales")
    ts_models = [name for name in model_names if not name.startswith('Hibrido')]
    if len(ts_models) >= 2:
        st.subheader("Comparación entre modelos de series temporales")
        model1 = st.selectbox("Modelo 1 TS", ts_models, index=0)
        model2 = st.selectbox("Modelo 2 TS", ts_models, index=1)

        y_pred1 = trained_models[model1].predict(X_test_ts, verbose=0)
        y_pred2 = trained_models[model2].predict(X_test_ts, verbose=0)
        y_true_classes = np.argmax(y_test, axis=1)
        y_pred1_classes = np.argmax(y_pred1, axis=1)
        y_pred2_classes = np.argmax(y_pred2, axis=1)

        # Coeficiente de Theil
        theil1 = theil_u(y_true_classes, y_pred1_classes)
        theil2 = theil_u(y_true_classes, y_pred2_classes)
        st.write(f"Theil U - {model1}: {theil1:.3f}, {model2}: {theil2:.3f}")

        # Prueba Diebold-Mariano
        DM_stat, p_value = diebold_mariano(y_true_classes, y_pred1_classes, y_pred2_classes)
        st.write(f"Diebold-Mariano: DM stat={DM_stat:.3f}, p-value={p_value:.3f}")
        if p_value < 0.05:
            st.success("Diferencia significativa entre los modelos TS (p < 0.05).")
        else:
            st.info("No hay evidencia de diferencia significativa entre los modelos TS (p ≥ 0.05).")

    return results_df


def calcular_estadigrafos(y_true, y_pred):
    stats = {
        "Accuracy": np.mean(y_true == y_pred),
        "Precision (macro)": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "Recall (macro)": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "F1-score (macro)": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "Kappa de Cohen": cohen_kappa_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
    }
    return stats

# =========================
#    REPORTE EN PDF
# =========================
def crear_pdf(results_df, trained_models, eval_results, save_path, lang="es"):
    from fpdf import FPDF
    import matplotlib.pyplot as plt
    import numpy as np

    # Seleccionar el mejor modelo según AUC
    best_row = results_df.sort_values(by='AUC', ascending=False).iloc[0]
    best_model_name = best_row['Modelo']
    best_eval = eval_results[best_model_name]

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Traducciones según idioma
    texto = traducciones.get(lang, traducciones["es"])

    # -----------------
    # Título y Resumen
    # -----------------
    pdf.set_font("Arial", "B", 16)
    pdf.multi_cell(0, 10, texto["app_title"], align="C")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, texto.get("report_summary_header", "Resumen Ejecutivo"), ln=True)
    pdf.set_font("Arial", "", 12)
    resumen = (f"{texto.get('report_summary_text', 'El mejor modelo fue')} {best_model_name} "
               f"con AUC={best_row['AUC']:.3f}, Accuracy={best_row['Accuracy']:.3f} "
               f"y MCC={best_row.get('MCC', 0):.3f}.\n"
               f"{texto.get('report_summary_followup', 'A continuación se presenta la tabla comparativa y detalles del modelo.')}")
    pdf.multi_cell(0, 6, resumen)
    pdf.ln(5)

    # -----------------
    # Tabla Comparativa
    # -----------------
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, texto.get("report_table_header", "Comparación de Modelos"), ln=True)
    pdf.set_font("Arial", "B", 10)

    cols = ['Modelo', 'Accuracy', 'AUC', 'MCC', 'Tiempo_Entrenamiento']
    table_data = results_df.copy()
    if 'MCC' not in table_data.columns:
        table_data['MCC'] = 0

    col_widths = [50, 25, 25, 25, 30]
    for i, col in enumerate(cols):
        pdf.cell(col_widths[i], 8, col, border=1, align='C')
    pdf.ln()

    pdf.set_font("Arial", "", 10)
    for _, row in table_data.iterrows():
        pdf.cell(col_widths[0], 8, str(row['Modelo']), border=1)
        pdf.cell(col_widths[1], 8, f"{row['Accuracy']:.3f}", border=1, align='C')
        pdf.cell(col_widths[2], 8, f"{row['AUC']:.3f}", border=1, align='C')
        pdf.cell(col_widths[3], 8, f"{row['MCC']:.3f}", border=1, align='C')
        pdf.cell(col_widths[4], 8, f"{row['Tiempo_Entrenamiento']:.2f}", border=1, align='C')
        pdf.ln()

    pdf.ln(5)

    # -----------------
    # Matriz de Confusión del Mejor Modelo
    # -----------------
    cm = best_eval['confusion_matrix']
    classes = best_eval.get('label_classes', list(range(cm.shape[0])))

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i_cm in range(cm.shape[0]):
        for j_cm in range(cm.shape[1]):
            ax.text(j_cm, i_cm, int(cm[i_cm, j_cm]), ha="center", va="center", color="black")

    ax.set_xlabel(texto.get("predicted_label", "Predicho"))
    ax.set_ylabel(texto.get("true_label", "Real"))
    fig.tight_layout()
    img_path = f"/tmp/{best_model_name}_cm.png"
    fig.savefig(img_path)
    plt.close(fig)

    pdf.image(img_path, w=120)

    # Guardar PDF
    pdf.output(save_path)




# =========================
#  CONFIGURACIÓN MULTILENGUAJE
# =========================
traducciones = {
    "es": {
        "app_title": "Predicción Avanzada de Tiempo de Recuperación Post-Cirugía",
        "app_description": """Esta aplicación utiliza modelos avanzados de deep learning (CNN y modelos híbridos) para
predecir y clasificar el tiempo de recuperación de pacientes después de una cirugía,
con validación estadística robusta y generación de reportes completos.""",
        "menu": "Menú",
        "eda": "EDA y Preprocesamiento",
        "training": "Entrenamiento de Modelos",
        "results": "Resultados y Estadísticas",
        "report": "Generar Reporte PDF",
        "about": "Acerca de",
        "eda_header": "Análisis Exploratorio de Datos",
        "eda_pie_title": "Distribución de Tiempos de Recuperación",
        "no_classes": "No hay clases de recuperación definidas (TIEMPO_RECUPERACION con NaN).",
        "no_numeric": "No hay suficientes columnas numéricas para el scatter.",
        "scatter_relation": "Relación entre Variables",
        "var_x": "Variable X",
        "var_y": "Variable Y",
        "report_summary_header": "Resumen Ejecutivo",
        "report_summary_text": "El mejor modelo fue",
        "report_summary_followup": "A continuación se presenta la tabla comparativa y detalles del modelo.",
        "report_table_header": "Comparación de Modelos",
        "predicted_label": "Predicho",
        "true_label": "Real",
        "color_by": "Color por",
        "time_series": "Series Temporales de Ejemplo",
        "no_time_series": "No hay series temporales disponibles.",
        "start_training": "Iniciar Entrenamiento de Modelos",
        "training_done": "Entrenamiento completado.",
        "train_results": "Resultados del Entrenamiento",
        "first_train_warning": "Primero entrena los modelos en la sección 'Entrenamiento de Modelos'.",
        "report_warning": "Completa el entrenamiento y resultados antes de generar el reporte.",
        "about_header": "Acerca de esta Aplicación",
        "about_text": """**Aplicación Avanzada de Predicción de Tiempo de Recuperación Post-Cirugía**

Esta aplicación permite evaluar y comparar diferentes modelos de redes neuronales para predecir la recuperación post-operatoria.

**Modelos incluidos:**
- **CNN_1D_Simple:** Red neuronal convolucional simple, ideal para series temporales cortas.
- **CNN_1D_Profunda:** CNN más profunda, captura patrones más complejos en datos temporales.
- **CNN_LSTM:** Combina convoluciones 1D con capas LSTM para capturar dependencias temporales a largo plazo.
- **Hibrido_CNN_MLP:** Combina CNN para series temporales y MLP para datos tabulares, integrando información de diferentes fuentes.
- **Hibrido_CNN_Attention:** Similar al anterior, con mecanismo de atención que enfatiza las características más relevantes de la serie temporal.

**Pruebas y métricas:**
- **AUC (Área Bajo la Curva ROC):** Evalúa la capacidad de discriminación de los modelos.
- **Accuracy:** Precisión global de clasificación.
- **MCC (Matthews Correlation Coefficient):** Mide la calidad de predicción considerando desbalance de clases.
- **McNemar:** Prueba estadística para comparar la diferencia de predicción entre pares de modelos.
- **Coeficiente de Theil (Theil U):** Mide la precisión de los pronósticos de series temporales.
- **Diebold-Mariano:** Compara pares de modelos en términos de error de predicción en series temporales.

**Visualizaciones interactivas:**
- Curvas ROC multiclase
- Barras comparativas de AUC
- Matrices de confusión dinámicas

**Reporte PDF automático:** resumen ejecutivo con el mejor modelo, tabla comparativa y gráficos.

**Claves para PDF:**
- report_summary_header: "Resumen Ejecutivo"
- report_summary_text: "El mejor modelo fue"
- report_summary_followup: "A continuación se presenta la tabla comparativa y detalles del modelo."
- report_table_header: "Comparación de Modelos"
- predicted_label: "Predicho"
- true_label: "Real"

**Advertencia:** Esta aplicación es solo para fines educativos o de soporte; no sustituye la evaluación ni el juicio clínico."""
    },

    "en": {
        "app_title": "Advanced Post-Surgery Recovery Time Prediction",
        "app_description": """This application uses advanced deep learning models (CNN and hybrid models) to
predict and classify patients' recovery time after surgery,
with robust statistical validation and complete report generation.""",
        "menu": "Menu",
        "eda": "EDA and Preprocessing",
        "training": "Model Training",
        "results": "Results and Statistics",
        "report": "Generate PDF Report",
        "about": "About",
        "eda_header": "Exploratory Data Analysis",
        "eda_pie_title": "Recovery Time Distribution",
        "no_classes": "No recovery classes defined (TIEMPO_RECUPERACION with NaN).",
        "no_numeric": "Not enough numeric columns for scatter plot.",
        "scatter_relation": "Variable Relationship",
        "var_x": "Variable X",
        "var_y": "Variable Y",
        "report_summary_header": "Executive Summary",
        "report_summary_text": "The best model was",
        "report_summary_followup": "Below is the comparison table and model details.",
        "report_table_header": "Models Comparison",
        "predicted_label": "Predicted",
        "true_label": "Actual",
        "color_by": "Color by",
        "time_series": "Example Time Series",
        "no_time_series": "No time series available.",
        "start_training": "Start Model Training",
        "training_done": "Training completed.",
        "train_results": "Training Results",
        "first_train_warning": "First, train the models in the 'Model Training' section.",
        "report_warning": "Complete training and results before generating the report.",
        "about_header": "About this Application",
        "about_text": """**Advanced Post-Surgery Recovery Time Prediction Application**

This application allows evaluating and comparing different neural network models to predict post-operative recovery.

**Included Models:**
- **CNN_1D_Simple:** Simple convolutional neural network, ideal for short time series.
- **CNN_1D_Deep:** Deeper CNN that captures more complex patterns in temporal data.
- **CNN_LSTM:** Combines 1D convolutions with LSTM layers to capture long-term temporal dependencies.
- **Hybrid_CNN_MLP:** Combines CNN for time series and MLP for tabular data, integrating information from multiple sources.
- **Hybrid_CNN_Attention:** Similar to the previous hybrid model, with an attention mechanism emphasizing the most relevant temporal features.

**Tests and Metrics:**
- **AUC (Area Under ROC Curve):** Measures the model's discriminative ability.
- **Accuracy:** Overall classification precision.
- **MCC (Matthews Correlation Coefficient):** Evaluates prediction quality considering class imbalance.
- **McNemar:** Statistical test comparing prediction differences between model pairs.
- **Theil's U Coefficient:** Measures forecasting accuracy for time series.
- **Diebold-Mariano:** Compares model pairs in terms of prediction errors on time series.

**Interactive Visualizations:**
- Multiclass ROC curves
- AUC comparison bars
- Dynamic confusion matrices

**Automatic PDF Report:** Executive summary with the best model, comparison table, and charts.

**PDF Keys:**
- report_summary_header: "Executive Summary"
- report_summary_text: "The best model was"
- report_summary_followup: "Below is the comparison table and model details."
- report_table_header: "Model Comparison"
- predicted_label: "Predicted"
- true_label: "Actual"

**Warning:** This application is for educational or support purposes only and does not replace clinical judgment."""
    },

    "pt": {
        "app_title": "Previsão Avançada de Tempo de Recuperação Pós-Cirurgia",
        "app_description": """Este aplicativo utiliza modelos avançados de deep learning (CNN e modelos híbridos) para
prever e classificar o tempo de recuperação de pacientes após uma cirurgia,
com validação estatística robusta e geração completa de relatórios.""",
        "menu": "Menu",
        "eda": "EDA e Pré-processamento",
        "training": "Treinamento de Modelos",
        "results": "Resultados e Estatísticas",
        "report": "Gerar Relatório PDF",
        "about": "Sobre",
        "eda_header": "Análise Exploratória de Dados",
        "eda_pie_title": "Distribuição do Tempo de Recuperação",
        "no_classes": "Nenhuma classe de recuperação definida (TIEMPO_RECUPERACION com NaN).",
        "no_numeric": "Não há colunas numéricas suficientes para o gráfico de dispersão.",
        "scatter_relation": "Relação entre Variáveis",
        "var_x": "Variável X",
        "var_y": "Variável Y",
        "report_summary_header": "Resumo Executivo",
        "report_summary_text": "O melhor modelo foi",
        "report_summary_followup": "A seguir está a tabela comparativa e detalhes do modelo.",
        "report_table_header": "Comparação de Modelos",
        "predicted_label": "Previsto",
        "true_label": "Real",
        "color_by": "Cor por",
        "time_series": "Séries Temporais de Exemplo",
        "no_time_series": "Nenhuma série temporal disponível.",
        "start_training": "Iniciar Treinamento de Modelos",
        "training_done": "Treinamento concluído.",
        "train_results": "Resultados do Treinamento",
        "first_train_warning": "Primeiro, treine os modelos na seção 'Treinamento de Modelos'.",
        "report_warning": "Complete o treinamento e os resultados antes de gerar o relatório.",
        "about_header": "Sobre esta Aplicação",
        "about_text": """**Aplicação Avançada de Previsão do Tempo de Recuperação Pós-Cirurgia**

Esta aplicação permite avaliar e comparar diferentes modelos de redes neurais para prever a recuperação pós-operatória.

**Modelos incluídos:**
- **CNN_1D_Simple:** Rede neural convolucional simples, ideal para séries temporais curtas.
- **CNN_1D_Profunda:** CNN mais profunda, captura padrões mais complexos em dados temporais.
- **CNN_LSTM:** Combina convoluções 1D com camadas LSTM para capturar dependências temporais de longo prazo.
- **Híbrido_CNN_MLP:** Combina CNN para séries temporais e MLP para dados tabulares, integrando informações de múltiplas fontes.
- **Híbrido_CNN_Attention:** Semelhante ao anterior, com mecanismo de atenção que destaca as características mais relevantes da série temporal.

**Testes e Métricas:**
- **AUC (Área sob a Curva ROC):** Avalia a capacidade de discriminação do modelo.
- **Accuracy:** Precisão geral da classificação.
- **MCC (Matthews Correlation Coefficient):** Avalia a qualidade da previsão considerando desequilíbrio de classes.
- **McNemar:** Teste estatístico para comparar diferenças de previsão entre pares de modelos.
- **Coeficiente de Theil (Theil U):** Mede a precisão das previsões em séries temporais.
- **Diebold-Mariano:** Compara pares de modelos em termos de erro de previsão em séries temporais.

**Visualizações Interativas:**
- Curvas ROC multiclasse
- Barras comparativas de AUC
- Matrizes de confusão dinâmicas

**Relatório PDF Automático:** Resumo executivo com o melhor modelo, tabela comparativa e gráficos.

**Chaves para PDF:**
- report_summary_header: "Resumo Executivo"
- report_summary_text: "O melhor modelo foi"
- report_summary_followup: "Abaixo está a tabela comparativa e detalhes do modelo."
- report_table_header: "Comparação de Modelos"
- predicted_label: "Previsto"
- true_label: "Real"

**Aviso:** Esta aplicação é apenas para fins educativos ou de suporte e não substitui o julgamento clínico."""
    }
}

def traducir(key):
    lang = st.session_state.get("lang", "es")
    return traducciones.get(lang, traducciones["es"]).get(key, key)


# =========================
#       INTERFAZ STREAMLIT
# =========================
def main():
    # Selección de idioma
    if "lang" not in st.session_state:
        st.session_state.lang = "es"

    lenguaje = st.sidebar.selectbox(
        "🌐 Idioma / Language / Idioma",
        options=["es", "en", "pt"],
        format_func=lambda x: {"es": "Español", "en": "English", "pt": "Português"}[x]
    )
    st.session_state.lang = lenguaje

    st.title(traducir("app_title"))
    st.write(traducir("app_description"))

    menu = st.sidebar.selectbox(traducir("menu"), [
        traducir("eda"),
        traducir("training"),
        traducir("results"),
        traducir("report"),
        traducir("about")
    ])

    if menu == traducir("eda"):
        st.header(traducir("eda_header"))

        # ======================
        # 📊 Estadísticos descriptivos
        # ======================
        #numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        numeric_cols = [c for c in ['EDAD','COMORBILIDADES','TIEMPO_RECUPERACION'] if c in df.columns]
        if numeric_cols:
            st.subheader("📈 Estadísticos descriptivos")
            col = st.selectbox("Selecciona una variable numérica", numeric_cols)

            if col:
                media = df[col].mean()
                mediana = df[col].median()
                moda = df[col].mode()[0] if not df[col].mode().empty else None
                varianza = df[col].var()
                desviacion = df[col].std()

                st.write(f"**Media:** {media:.2f}")
                st.write(f"**Mediana:** {mediana:.2f}")
                st.write(f"**Moda:** {moda}")
                st.write(f"**Varianza:** {varianza:.2f}")
                st.write(f"**Desviación estándar:** {desviacion:.2f}")

                # Histograma
                fig, ax = plt.subplots()
                sns.histplot(df[col].dropna(), kde=True, ax=ax)
                ax.set_title(f"Distribución de {col}")
                st.pyplot(fig)

                # Boxplot
                fig2, ax2 = plt.subplots()
                sns.boxplot(x=df[col], ax=ax2)
                ax2.set_title(f"Boxplot de {col}")
                st.pyplot(fig2)

        # ======================
        # 🍰 Gráfico de torta (ya lo tenías)
        # ======================
        if df['CLASE_RECUPERACION'].notna().any():
            fig = px.pie(df.dropna(subset=['CLASE_RECUPERACION']),
                        names='CLASE_RECUPERACION',
                        title=traducir("eda_pie_title"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(traducir("no_classes"))

        # ======================
        # 🔹 Scatter dinámico (ya lo tenías)
        # ======================
        numeric_candidates_x = [c for c in ['EDAD','COMORBILIDADES','TIEMPO_RECUPERACION'] if c in df.columns]
        numeric_candidates_y = [c for c in ['TIEMPO_RECUPERACION','EDAD','COMORBILIDADES'] if c in df.columns]
        color_candidates = ['None'] + [c for c in ['CLASE_RECUPERACION','TIPO_CIRUGIA','GENDER'] if c in df.columns]

        if not numeric_candidates_x or not numeric_candidates_y:
            st.warning(traducir("no_numeric"))
        else:
            st.subheader(traducir("scatter_relation"))
            x_axis = st.selectbox(traducir("var_x"), numeric_candidates_x, index=0)
            y_axis = st.selectbox(traducir("var_y"), numeric_candidates_y, index=0)
            color_by = st.selectbox(traducir("color_by"), color_candidates, index=0)

            dfx = df.dropna(subset=[x_axis, y_axis]).copy()
            if color_by == 'None':
                fig = px.scatter(dfx, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
            else:
                fig = px.scatter(dfx, x=x_axis, y=y_axis, color=color_by,
                                title=f"{y_axis} vs {x_axis} (color: {color_by})")
            st.plotly_chart(fig, use_container_width=True)

        # ======================
        # 📈 Serie temporal (ya lo tenías)
        # ======================
        st.subheader(traducir("time_series"))
        if 'Serie_Temporal' in df.columns and len(df) > 0:
            sample_patients = st.slider("Número de pacientes a mostrar", 1, 10, 3)
            fig_ts = go.Figure()
            subset = df.head(sample_patients)
            for i, (_, row) in enumerate(subset.iterrows(), start=1):
                fig_ts.add_trace(go.Scatter(
                    y=row['Serie_Temporal'],
                    name=f"Fila {i} - {row.get('CLASE_RECUPERACION','NA')}"
                ))
            fig_ts.update_layout(title=traducir("time_series"),
                                xaxis_title="Tiempo", yaxis_title="Valor Normalizado")
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info(traducir("no_time_series"))


    elif menu == traducir("training"):
        if st.button(traducir("start_training")):
            with st.spinner("Entrenando modelos..."):
                # Entrenamiento y evaluación
                results, trained_models, history_logs, eval_results, le, y_test = entrenar_y_evaluar_modelos()
                
                # Guardar en session_state
                st.session_state['results'] = results
                st.session_state['trained_models'] = trained_models
                st.session_state['history_logs'] = history_logs
                st.session_state['eval_results'] = eval_results
                st.session_state['le'] = le
                st.session_state['y_test'] = y_test
                
                st.success(traducir("training_done"))

        # Mostrar tabla resumida de resultados
        if 'results' in st.session_state:
            st.header(traducir("train_results"))
            results_df = pd.DataFrame(st.session_state['results'])
            if 'Matriz_Confusion' in results_df.columns:
                results_df_no_cm = results_df.drop(columns=['Matriz_Confusion'])
            else:
                results_df_no_cm = results_df.copy()
            st.dataframe(results_df_no_cm.sort_values(by='AUC', ascending=False))
            st.subheader("📊 Estadística descriptiva de variables numéricas")
            st.dataframe(df[['EDAD', 'COMORBILIDADES', 'TIEMPO_RECUPERACION']].describe().T)

    elif menu == traducir("results"):
        if all(k in st.session_state for k in ['results', 'trained_models', 'le', 'eval_results', 'y_test']):
            results_df = mostrar_resultados_estadisticas(
                st.session_state['results'],
                st.session_state['trained_models'],
                st.session_state['le'],
                st.session_state['y_test']
            )
            st.session_state['results_df'] = results_df

        else:
            st.warning(traducir("first_train_warning"))

    elif menu == traducir("report"):
      if 'results_df' in st.session_state and 'trained_models' in st.session_state and 'eval_results' in st.session_state:
          pdf_path = "/tmp/reporte_modelos.pdf"
          crear_pdf(
              results_df=st.session_state['results_df'],
              trained_models=st.session_state['trained_models'],
              eval_results=st.session_state['eval_results'],
              save_path=pdf_path,
              lang=st.session_state.lang
          )

          with open(pdf_path, "rb") as f:
              pdf_bytes = f.read()
              st.download_button(
                  label="📥 Descargar Reporte PDF",
                  data=pdf_bytes,
                  file_name="reporte_modelos.pdf",
                  mime="application/pdf"
              )
      else:
          st.warning(traducir("report_warning"))

    elif menu == traducir("about"):
          st.header(traducir("about_header"))
          st.write(traducir("about_text"))

# =========================
#          MAIN
# =========================
if __name__ == "__main__":
    main()



