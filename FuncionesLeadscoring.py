#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import pandas as pd
import cloudpickle

# Resolver rutas relativas de forma dinamica
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', '04_Modelos')
if not os.path.exists(MODEL_DIR):
    MODEL_DIR = BASE_DIR

# Lista de categorias principales (no raras) detectadas durante el entrenamiento del modelo.
# Las demas se agrupan en 'OTROS' para mantener coherencia matematica con el OHE.
CATEGORIAS_VALIDAS = {
    'ocupacion': ['Student', 'Unemployed', 'Working Professional'],
    'ambito': [
        'Banking, Investment And Insurance', 'Business Administration', 'Finance Management',
        'Healthcare Management', 'Human Resource Management', 'IT Projects Management',
        'International Business', 'Marketing Management', 'Media and Advertising',
        'Operations Management', 'Select', 'Supply Chain Management', 'Travel and Tourism'
    ],
    'descarga_lm': ['No', 'Yes'],
    'ult_actividad': [
        'Chat Conversation', 'Converted to Lead', 'Email Link Clicked', 'Email Opened',
        'Page Visited on Website', 'SMS Sent'
    ]
}

def calidad_datos(df):
    """
    Funcion original de limpieza y preparacion del dataset.
    Nota: Se mantiene para compatibilidad con la deserializacion de sklearn/cloudpickle.
    """
    temp = df.copy()
    
    def imputar_moda(variable):
        return(variable.fillna(variable.mode()[0]))
    
    var_imputar_moda = ['ocupacion','ambito']
    temp[var_imputar_moda] = temp[var_imputar_moda].apply(imputar_moda)
    
    var_imputar_valor = ['descarga_lm','ult_actividad']
    valor = 'DESCONOCIDO'
    temp[var_imputar_valor] = temp[var_imputar_valor].fillna(valor)
    
    var_imputar_mediana = ['paginas_vistas_visita','score_actividad','score_perfil','tiempo_en_site_total']
    
    def imputar_mediana(variable):
        if pd.api.types.is_integer_dtype(variable):
            return(variable.fillna(int(variable.median())))
        else:
            return(variable.fillna(variable.median()))
    
    temp[var_imputar_mediana] = temp[var_imputar_mediana].apply(imputar_mediana)
    
    def agrupar_cat_raras(variable, criterio = 0.02):
        frecuencias = variable.value_counts(normalize=True)
        temp_list = [cada for cada in frecuencias.loc[frecuencias < criterio].index.values]
        temp2 = np.where(variable.isin(temp_list),'OTROS',variable)
        return(temp2)
    
    var_agrupar_cat_raras = ['ocupacion','ambito','descarga_lm','ult_actividad']
    for variable in var_agrupar_cat_raras:
        temp[variable] = agrupar_cat_raras(temp[variable], criterio = 0.02)
    
    temp['paginas_vistas_visita'] = temp['paginas_vistas_visita'].clip(0, 20)
    
    return(temp)

def mapear_raros_a_otros(df):
    """
    Pre-mapea categorias que no son comunes en el entrenamiento hacia 'OTROS'.
    Esto es crucial para registros individuales (1 sola fila) o lotes pequenos
    donde la frecuencia dinamica (value_counts) fallaria en agruparlas.
    """
    temp = df.copy()
    for col, validas in CATEGORIAS_VALIDAS.items():
        if col in temp.columns:
            temp[col] = temp[col].apply(lambda x: x if (pd.isna(x) or str(x) in validas) else 'OTROS')
    return temp

def cargar_modelo():
    """
    Carga el pipeline de ejecucion entrenado.
    """
    # Intentar buscar en el mismo directorio (para despliegue autocontenido)
    ruta_pipe = os.path.join(BASE_DIR, 'pipe_ejecucion.pickle')
    if not os.path.exists(ruta_pipe):
        # Fallback al directorio original del modelo
        ruta_pipe = os.path.join(MODEL_DIR, 'pipe_ejecucion.pickle')
        
    if not os.path.exists(ruta_pipe):
        raise FileNotFoundError(f"No se encontro el archivo del modelo en: {ruta_pipe}")
    
    with open(ruta_pipe, 'rb') as f:
        pipe = cloudpickle.load(f)
    return pipe

def predecir(df):
    """
    Prepara los datos pre-mapeando categorias raras y ejecuta la prediccion del scoring.
    """
    # 1. Asegurar que las variables estan en el orden y nombre correcto
    variables_finales = ['ambito', 'descarga_lm', 'ocupacion', 'paginas_vistas_visita',
                         'score_actividad', 'score_perfil', 'tiempo_en_site_total', 'ult_actividad']
    
    df_prepared = df[variables_finales].copy()
    
    # 2. Pre-mapear raros a OTROS
    df_prepared = mapear_raros_a_otros(df_prepared)
    
    # 3. Cargar el pipeline y predecir probabilidades
    pipe = cargar_modelo()
    probabilidades = pipe.predict_proba(df_prepared)[:, 1]
    
    return probabilidades
