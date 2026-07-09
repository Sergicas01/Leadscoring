import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Asegurar que el directorio de despliegue esta en el PYTHONPATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Importar funciones de leadscoring
try:
    import FuncionesLeadscoring as fls
    funcs_loaded = True
except Exception as e:
    funcs_loaded = False
    funcs_error = str(e)

# CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="Sistema de Priorización de Leads",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ESTILOS CSS PERSONALIZADOS (Alineado con un diseño corporativo moderno y sobrio)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
    }
    
    /* Cabecera corporativa */
    .corporate-header {
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 20px;
        margin-bottom: 30px;
    }
    .corporate-title {
        font-size: 28px;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.03em;
        margin: 0;
    }
    .corporate-subtitle {
        font-size: 14px;
        color: #475569;
        margin-top: 4px;
    }
    
    /* Panel de formulario */
    .form-panel {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
    }
    .panel-section-title {
        font-size: 15px;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 16px;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 8px;
    }
    
    /* Tarjeta de resultado ejecutivo */
    .executive-card {
        border-radius: 6px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 4px solid #cbd5e1;
    }
    .badge-high {
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        display: inline-block;
    }
    .badge-medium {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fef08a;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        display: inline-block;
    }
    .badge-low {
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #e2e8f0;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# RESOLUCIÓN DE RUTA DE DATOS POR DEFECTO
def get_default_data_path():
    # Intentar buscar en el mismo directorio (para despliegue autocontenido)
    path_local = os.path.join(BASE_DIR, 'validacion.csv')
    if os.path.exists(path_local):
        return path_local
    # Fallback al directorio original de datos
    path_orig = os.path.join(BASE_DIR, '..', '02_Datos', '02_Validacion', 'validacion.csv')
    if os.path.exists(path_orig):
        return path_orig
    return None

# CABECERA PRINCIPAL
st.markdown("""
<div class="corporate-header">
    <h1 class="corporate-title">Sistema de Priorización y Calificación de Leads</h1>
    <p class="corporate-subtitle">Plataforma de análisis comercial para la clasificación automatizada de registros y optimización del embudo de conversión.</p>
</div>
""", unsafe_allow_html=True)

if not funcs_loaded:
    st.error(f"Error al inicializar el módulo de ejecución: {funcs_error}")
    st.stop()

# Cargar dataset de validación de fondo
default_path = get_default_data_path()
raw_df = None
if default_path:
    try:
        raw_df = pd.read_csv(default_path, sep=',')
    except Exception:
        pass

# CREACIÓN DE PESTAÑAS (Pestaña individual y Pestaña de análisis histórico EDA)
tab1, tab2 = st.tabs(["Calificación de Lead Individual", "Análisis de Factores de Conversión (EDA)"])

with tab1:
    # FORMULARIO DE ENTRADA
    st.markdown('<div class="form-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-section-title">Parámetros del Registro Comercial (Lead)</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        ocupacion = st.selectbox(
            "Situación Profesional:",
            options=['Unemployed', 'Working Professional', 'Student', 'Housewife', 'Businessman', 'Other'],
            format_func=lambda x: {
                'Unemployed': 'Desempleado',
                'Working Professional': 'Profesional en Activo',
                'Student': 'Estudiante',
                'Housewife': 'Ama de Casa',
                'Businessman': 'Empresario',
                'Other': 'Otro'
            }.get(x, x),
            index=1,
            help="Ocupación actual declarada por el contacto."
        )
        
        ambito = st.selectbox(
            "Área de Especialización:",
            options=[
                'Select', 'Supply Chain Management', 'Healthcare Management', 'Operations Management', 
                'Human Resource Management', 'Retail Management', 'Banking, Investment And Insurance', 
                'Finance Management', 'Business Administration', 'Marketing Management', 
                'Media and Advertising', 'Rural and Agribusiness', 'Hospitality Management', 
                'Travel and Tourism', 'International Business', 'IT Projects Management', 
                'Services Excellence', 'E-COMMERCE', 'E-Business'
            ],
            format_func=lambda x: {
                'Select': 'No Especificado',
                'Supply Chain Management': 'Gestión de Cadena de Suministro',
                'Healthcare Management': 'Gestión Sanitaria',
                'Operations Management': 'Gestión de Operaciones',
                'Human Resource Management': 'Recursos Humanos',
                'Retail Management': 'Gestión de Retail',
                'Banking, Investment And Insurance': 'Banca, Inversión y Seguros',
                'Finance Management': 'Gestión Financiera',
                'Business Administration': 'Administración de Empresas',
                'Marketing Management': 'Gestión de Marketing',
                'Media and Advertising': 'Medios y Publicidad',
                'Rural and Agribusiness': 'Agroindustria',
                'Hospitality Management': 'Gestión de Hostelería',
                'Travel and Tourism': 'Turismo',
                'International Business': 'Negocios Internacionales',
                'IT Projects Management': 'Gestión de Proyectos IT',
                'Services Excellence': 'Excelencia en Servicios',
                'E-COMMERCE': 'Comercio Electrónico',
                'E-Business': 'Negocios Digitales'
            }.get(x, x),
            index=9,
            help="Sector profesional del contacto."
        )

    with col2:
        descarga_lm = st.selectbox(
            "Descarga de Contenido Directo:",
            options=['No', 'Yes'],
            format_func=lambda x: {'No': 'No', 'Yes': 'Sí'}.get(x, x),
            index=0,
            help="Indica si el contacto ha descargado recursos informativos (Lead Magnets)."
        )
        
        ult_actividad = st.selectbox(
            "Último Canal de Contacto:",
            options=[
                'Chat Conversation', 'Email Opened', 'Email Link Clicked', 'Unreachable', 
                'SMS Sent', 'Email Bounced', 'Converted to Lead', 'Unsubscribed', 
                'Page Visited on Website', 'Form Submitted on Website', 'Had a Phone Conversation', 
                'Email Received', 'Approached upfront'
            ],
            format_func=lambda x: {
                'Chat Conversation': 'Conversación por Chat',
                'Email Opened': 'Correo Electrónico Abierto',
                'Email Link Clicked': 'Enlace en Correo Clickeado',
                'Unreachable': 'No Localizable',
                'SMS Sent': 'Mensaje SMS Enviado',
                'Email Bounced': 'Correo Rebotado',
                'Converted to Lead': 'Convertido en Lead',
                'Unsubscribed': 'Baja de Suscripción',
                'Page Visited on Website': 'Visita de Página Web',
                'Form Submitted on Website': 'Formulario de Web Enviado',
                'Had a Phone Conversation': 'Conversación Telefónica',
                'Email Received': 'Correo Recibido',
                'Approached upfront': 'Contacto Directo Proactivo'
            }.get(x, x),
            index=1,
            help="Último punto de contacto digital registrado."
        )

    with col3:
        tiempo_en_site_total = st.slider(
            "Tiempo de Permanencia Web (segundos):",
            min_value=0,
            max_value=2500,
            value=262,
            step=10,
            help="Tiempo acumulado en el sitio web corporativo."
        )
        
        paginas_vistas_visita = st.slider(
            "Páginas Vistas por Sesión:",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.5
        )

    st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)
    col_scores_1, col_scores_2 = st.columns(2)
    with col_scores_1:
        score_actividad = st.slider(
            "Índice de Actividad Comercial:",
            min_value=9.0,
            max_value=18.0,
            value=14.0,
            step=1.0,
            help="Calificación interna del volumen de actividad comercial registrado."
        )
    with col_scores_2:
        score_perfil = st.slider(
            "Índice de Ajuste de Perfil (Fit Score):",
            min_value=11.0,
            max_value=20.0,
            value=16.0,
            step=1.0,
            help="Grado de concordancia del contacto con el Buyer Persona ideal."
        )

    st.markdown("---")
    calcular_btn = st.button("Evaluar Lead", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # LÓGICA DE PREDICCIÓN Y RESULTADOS
    if calcular_btn or 'individual_result' in st.session_state:
        if calcular_btn:
            lead_data = pd.DataFrame([{
                'ambito': ambito,
                'descarga_lm': descarga_lm,
                'ocupacion': ocupacion,
                'paginas_vistas_visita': paginas_vistas_visita,
                'score_actividad': score_actividad,
                'score_perfil': score_perfil,
                'tiempo_en_site_total': tiempo_en_site_total,
                'ult_actividad': ult_actividad
            }], index=[0])
            
            try:
                prob = fls.predecir(lead_data)[0]
                st.session_state['individual_result'] = prob
                st.session_state['last_inputs'] = (ocupacion, tiempo_en_site_total, score_actividad, descarga_lm, score_perfil)
            except Exception as ex:
                st.error(f"Error durante el cálculo del scoring: {ex}")
                st.stop()
                
        prob = st.session_state['individual_result']
        ocupacion_last, tiempo_last, score_act_last, descarga_last, score_perfil_last = st.session_state['last_inputs']
        
        # Determinar el estado comercial
        if prob >= 0.60:
            status_label = "Prioridad Alta"
            badge_html = f'<div class="badge-high">{status_label}</div>'
            color = "#166534"  # verde oscuro corporativo
            bg_color = "#f0fdf4"
            border_color = "#bbf7d0"
            desc = "Este contacto presenta una elevada probabilidad de conversión comercial. Se aconseja una llamada directa en un plazo menor a 4 horas para avanzar en el proceso de venta."
        elif prob >= 0.30:
            status_label = "Prioridad Media"
            badge_html = f'<div class="badge-medium">{status_label}</div>'
            color = "#92400e"  # dorado/marron corporativo
            bg_color = "#fffbeb"
            border_color = "#fef08a"
            desc = "El lead muestra un interés moderado. Se recomienda el seguimiento por correo electrónico con contenidos dirigidos o propuesta de sesión informativa."
        else:
            status_label = "Prioridad Baja"
            badge_html = f'<div class="badge-low">{status_label}</div>'
            color = "#475569"  # gris corporativo
            bg_color = "#f8fafc"
            border_color = "#cbd5e1"
            desc = "La probabilidad de conversión es reducida en la fase actual. Se sugiere derivar a flujos automáticos de nutrición periódica y boletines."

        st.markdown('<div class="panel-section-title" style="margin-top: 15px;">Informe Ejecutivo de Calificación</div>', unsafe_allow_html=True)
        col_g, col_details = st.columns([1, 1.3])
        
        with col_g:
            # Gráfico Gauge sobrio y corporativo
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number={'suffix': '%', 'font': {'size': 36, 'color': '#0f172a'}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b", 'tickfont': {'color': '#64748b', 'size': 11}},
                    'bar': {'color': color, 'thickness': 0.8},
                    'bgcolor': "#f1f5f9",
                    'borderwidth': 1,
                    'bordercolor': "#cbd5e1",
                    'steps': [
                        {'range': [0, 30], 'color': '#f8fafc'},
                        {'range': [30, 60], 'color': '#fffbeb'},
                        {'range': [60, 100], 'color': '#f0fdf4'}
                    ],
                }
            ))
            fig_g.update_layout(
                height=200,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'family': 'Inter'}
            )
            st.plotly_chart(fig_g, use_container_width=True)
            
        with col_details:
            st.markdown(f"""
            <div class="executive-card" style="background-color: {bg_color}; border-left-color: {color}; border-top: 1px solid {border_color}; border-bottom: 1px solid {border_color}; border-right: 1px solid {border_color};">
                <div style="margin-bottom: 10px;">{badge_html}</div>
                <p style="margin: 0; font-size: 14px; font-weight: 500; color: #1e293b; line-height: 1.6;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Siguientes pasos tácticos comerciales
            st.markdown('<div style="font-size: 13px; font-weight: 600; color: #0f172a; margin-bottom: 8px;">Recomendaciones de Actuación Comercial:</div>', unsafe_allow_html=True)
            actions = []
            if ocupacion_last == 'Working Professional':
                actions.append("Contacto enfocado en valor comercial/ROI y optimización de flujos corporativos.")
            if tiempo_last > 500:
                actions.append(f"Interacción web prolongada ({tiempo_last} segundos). Resolver dudas técnicas sobre el producto.")
            if score_act_last >= 16:
                actions.append("Actividad digital reciente elevada. Prioridad de contacto inmediata.")
            if descarga_last == 'Yes':
                actions.append("El contacto ha mostrado interés específico en recursos web; enviar contenido complementario.")
            if score_perfil_last < 13:
                actions.append("Perfil por debajo del Fit Score óptimo. Validar presupuesto de forma temprana.")
                
            if not actions:
                actions.append("Proceder con el protocolo estándar de nutrición y maduración de la cuenta.")
                
            for action in actions:
                st.markdown(f'<div style="font-size: 13px; color: #475569; margin-bottom: 4px; padding-left: 8px; border-left: 2px solid #e2e8f0;">• {action}</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="panel-section-title">Análisis Histórico de Conversión</div>', unsafe_allow_html=True)
    if raw_df is not None:
        st.write("Visualización agregada de patrones de comportamiento e interacción correspondientes a la base de validación del proyecto.")
        
        # Preparación de datos para el análisis (aplicando los filtros de contactabilidad corporativos)
        df_eda = raw_df.copy()
        df_eda = df_eda.drop_duplicates()
        if 'no_llamar' in df_eda.columns and 'no_enviar_email' in df_eda.columns:
            df_eda = df_eda.loc[
                (df_eda.no_llamar != 'OTROS') & 
                (df_eda.no_enviar_email != 'Yes') & 
                (df_eda.ult_actividad != 'Email Bounced')
            ]
        df_eda['compra_label'] = df_eda['compra'].map({1: 'Compró', 0: 'No Compró'})
        
        col_graph1, col_graph2 = st.columns(2)
        
        with col_graph1:
            st.markdown('<div style="font-size: 14px; font-weight: 600; color: #0f172a; margin-bottom: 10px;">Tasa de Conversión por Ocupación</div>', unsafe_allow_html=True)
            # Agrupar y calcular la tasa de conversión y el tamaño de muestra
            oc_conv = df_eda.groupby('ocupacion')['compra'].agg(['mean', 'count']).reset_index()
            # Filtrar ocupaciones con menos de 10 leads para asegurar significancia estadística
            oc_conv = oc_conv[oc_conv['count'] > 10]
            oc_conv['Tasa de Conversión (%)'] = round(oc_conv['mean'] * 100, 1)
            oc_conv = oc_conv.sort_values('Tasa de Conversión (%)', ascending=False)
            
            fig1 = px.bar(
                oc_conv,
                x='ocupacion',
                y='Tasa de Conversión (%)',
                color='Tasa de Conversión (%)',
                color_continuous_scale='Blues',
                labels={'ocupacion': 'Ocupación'},
                text='Tasa de Conversión (%)'
            )
            fig1.update_layout(plot_bgcolor='#ffffff', height=350, margin=dict(t=10, b=10, l=10, r=10), font={'family': 'Inter'})
            fig1.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
            st.plotly_chart(fig1, use_container_width=True)
            st.caption("Nota: Se han excluido de este gráfico las ocupaciones con menos de 10 registros en el dataset (como Empresarios u Amas de Casa) para garantizar la relevancia estadística de las tasas mostradas.")
            
        with col_graph2:
            st.markdown('<div style="font-size: 14px; font-weight: 600; color: #0f172a; margin-bottom: 10px;">Tiempo de Permanencia Web vs Tasa de Compra</div>', unsafe_allow_html=True)
            df_eda['rango_tiempo'] = pd.cut(
                df_eda['tiempo_en_site_total'], 
                bins=[0, 100, 300, 600, 1200, 2500], 
                labels=['0-100s', '100-300s', '300-600s', '600-1200s', '>1200s']
            )
            time_conv = df_eda.groupby('rango_tiempo')['compra'].mean().reset_index()
            time_conv['Tasa de Conversión (%)'] = round(time_conv['compra'] * 100, 1)
            
            fig2 = px.line(
                time_conv,
                x='rango_tiempo',
                y='Tasa de Conversión (%)',
                markers=True,
                labels={'rango_tiempo': 'Rango de Tiempo'},
                color_discrete_sequence=['#1e3a8a']
            )
            fig2.update_layout(plot_bgcolor='#ffffff', height=350, margin=dict(t=10, b=10, l=10, r=10), font={'family': 'Inter'})
            fig2.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
            st.plotly_chart(fig2, use_container_width=True)
            
        col_graph3, col_graph4 = st.columns(2)
        
        with col_graph3:
            st.markdown('<div style="font-size: 14px; font-weight: 600; color: #0f172a; margin-bottom: 10px;">Distribución de Actividad Comercial según Conversión</div>', unsafe_allow_html=True)
            fig3 = px.box(
                df_eda,
                x='compra_label',
                y='score_actividad',
                color='compra_label',
                color_discrete_map={'Compró': '#166534', 'No Compró': '#64748b'},
                labels={'compra_label': 'Conversión', 'score_actividad': 'Índice de Actividad'}
            )
            fig3.update_layout(plot_bgcolor='#ffffff', height=350, margin=dict(t=10, b=10, l=10, r=10), showlegend=False, font={'family': 'Inter'})
            fig3.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
            st.plotly_chart(fig3, use_container_width=True)
            
        with col_graph4:
            st.markdown('<div style="font-size: 14px; font-weight: 600; color: #0f172a; margin-bottom: 10px;">Conversión por Canal de Actividad Reciente</div>', unsafe_allow_html=True)
            act_conv = df_eda.groupby('ult_actividad')['compra'].agg(['mean', 'count']).reset_index()
            # Filtrar solo actividades representativas (>20 registros)
            act_conv = act_conv[act_conv['count'] > 20]
            act_conv['Tasa de Conversión (%)'] = round(act_conv['mean'] * 100, 1)
            act_conv = act_conv.sort_values('Tasa de Conversión (%)', ascending=True)
            
            fig4 = px.bar(
                act_conv,
                y='ult_actividad',
                x='Tasa de Conversión (%)',
                orientation='h',
                color='Tasa de Conversión (%)',
                color_continuous_scale='Blues',
                labels={'ult_actividad': 'Última Actividad'}
            )
            fig4.update_layout(plot_bgcolor='#ffffff', height=350, margin=dict(t=10, b=10, l=10, r=10), font={'family': 'Inter'})
            fig4.update_xaxes(showgrid=True, gridcolor='#f1f5f9')
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No se encontró el fichero de validación 'validacion.csv' en los directorios definidos para generar los análisis descriptivos.")
