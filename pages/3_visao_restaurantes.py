# Libraries
from haversine import haversine
import plotly.express as px
import plotly.graph_objects as go

# bibliotecas necessárias
import folium
import numpy as np
import pandas as pd
import datetime
import streamlit as st
from PIL import Image

from streamlit_folium import folium_static

st.set_page_config(page_title='Visão Restaurantes', page_icon='🍽', layout='wide')

# Import dataset
df = pd.read_csv( 'dataset/train.csv' )

df1 = df.copy()

# 1. STANDARDIZE MISSING VALUES (Replace 'NaN ' string with proper np.nan)
df1 = df1.replace('NaN ', np.nan) # This replaces the string 'NaN ' across the entire DataFrame

# 2. CLEANING: Drop all rows where any of the critical columns have a missing value (NaN).
df1 = df1.dropna(subset=[
    'Delivery_person_Age',
    'Road_traffic_density',
    'City',
    'Festival',
    'multiple_deliveries'
])

# 3. CONVERSION: Convert Data Types Safely

# Convert Delivery_person_Age (your problematic line 33)
df1['Delivery_person_Age'] = df1['Delivery_person_Age'].astype(int)

# Convert Delivery_person_Ratings
df1['Delivery_person_Ratings'] = df1['Delivery_person_Ratings'].astype(float)

# Convert Order_Date
df1['Order_Date'] = pd.to_datetime( df1['Order_Date'], format='%d-%m-%Y' )

# Convert multiple_deliveries
df1['multiple_deliveries'] = df1['multiple_deliveries'].astype(int)

## 5. Removendo os espacos dentro de strings/texto/object
#df1 = df1.reset_index( drop=True )
#for i in range( len( df1 ) ):
#  df1.loc[i, 'ID'] = df1.loc[i, 'ID'].strip()


# 6. Removendo os espacos dentro de strings/texto/object

df1.loc[:, 'ID'] = df1.loc[:, 'ID'].str.strip()
df1.loc[:, 'Road_traffic_density'] = df1.loc[:, 'Road_traffic_density'].str.strip()
df1.loc[:, 'Type_of_order'] = df1.loc[:, 'Type_of_order'].str.strip()
df1.loc[:, 'Type_of_vehicle'] = df1.loc[:, 'Type_of_vehicle'].str.strip()
df1.loc[:, 'City'] = df1.loc[:, 'City'].str.strip()
df1.loc[:, 'Festival'] = df1.loc[:, 'Festival'].str.strip()

# 7. Limpando a coluna de time taken
df1['Time_taken(min)'] = df1['Time_taken(min)'].apply( lambda x: x.split( '(min) ')[1] )
df1['Time_taken(min)']  = df1['Time_taken(min)'].astype( int )

# =======================================
# Barra Lateral
# =======================================
st.header( 'Marketplace - Visão Restaurantes' )

#image_path = r'C:\Users\lucas\Documents\repos\stock.png'
image = Image.open( 'stock.png' )
st.sidebar.image( image, width=120 )

st.sidebar.markdown( '# Cury Company' )
st.sidebar.markdown( '## Fastest Delivery in Town' )
st.sidebar.markdown( """---""" )

st.sidebar.markdown( '## Selecione uma data limite' )

date_slider = st.sidebar.slider( 
    'Até qual valor?',
    value=datetime.datetime( 2022, 4, 13 ),
    min_value=datetime.datetime(2022, 2, 11 ),
    max_value=datetime.datetime( 2022, 4, 6 ),
    format='DD-MM-YYYY' )

st.sidebar.markdown( """---""" )


traffic_options = st.sidebar.multiselect( 
    'Quais as condições do trânsito',
    ['Low', 'Medium', 'High', 'Jam'], 
    default=['Low', 'Medium', 'High', 'Jam'] )

st.sidebar.markdown( """---""" )
st.sidebar.markdown( '### Powered by Comunidade DS' )

# Filtro de data
linhas_selecionadas = df1['Order_Date'] <  date_slider 
df1 = df1.loc[linhas_selecionadas, :]

# Filtro de transito
linhas_selecionadas = df1['Road_traffic_density'].isin( traffic_options )
df1 = df1.loc[linhas_selecionadas, :]

# =======================================
# Layout no Streamlit
# =======================================
tab1, tab2, tab3 = st.tabs( ['Visão Gerencial', '_', '_'] )

with tab1:
    with st.container():
        st.title( 'Overall Metrics' )
        
        col1, col2, col3, col4, col5, col6 = st.columns( 6 )
        with col1:
            delivery_unique = len( df1.loc[:, 'Delivery_person_ID'].unique() ) 
            col1.metric('Entregadores únicos', delivery_unique )
            
        with col2:
            cols = ['Delivery_location_latitude', 'Delivery_location_longitude', 'Restaurant_latitude', 'Restaurant_longitude'] 
            df1['distance'] = df1.loc[:, cols].apply(
                lambda x: haversine(
                # First argument (Point 1): Restaurant coordinates as a tuple
                (x['Restaurant_latitude'], x['Restaurant_longitude']),
                # Second argument (Point 2): Delivery coordinates as a tuple
                (x['Delivery_location_latitude'], x['Delivery_location_longitude'])),axis=1)

            avg_distance = np.round(df1['distance'].mean(), 2)
            col2.metric( 'A distancia media das entregas', avg_distance )
            
        with col3:
            df_aux = (df1.loc[:, ['Time_taken(min)', 'Festival']]
                      .groupby( 'Festival')
                      .agg({'Time_taken(min)': ['mean', 'std']} ) )
            
            df_aux.columns = ['avg_time', 'std_time']
            df_aux = df_aux.reset_index()
            # 1. Filter the data for 'Yes'
            filtered_series = df_aux.loc[df_aux['Festival'] == 'Yes', 'avg_time']
        
            # 2. Check if the Series is empty
            if not filtered_series.empty:
                # If it's NOT empty, extract the single value and round it.
                avg_time_with_festival = np.round(filtered_series.item(), 2)
                # Format the output value to ensure it's displayed nicely
                metric_value = f'{avg_time_with_festival} min'
            else:
                # If it IS empty (size 0), assign a fallback value.
                avg_time_with_festival = 0.0
                metric_value = 'N/A' # Display 'N/A' in the Streamlit metric
        
            # 3. Pass the safely determined value to the Streamlit metric
            col3.metric('Tempo Médio de Entrega c/ Festival', metric_value )
            
        with col4:
            df_aux = (df1.loc[:, ['Time_taken(min)', 'Festival']]
                      .groupby( 'Festival')
                      .agg({'Time_taken(min)': ['mean', 'std']} ) )
            
            df_aux.columns = ['avg_time', 'std_time']
            df_aux = df_aux.reset_index()
            # 1. Filter the data for 'Yes' in the 'std_time' column
            filtered_series_std = df_aux.loc[df_aux['Festival'] == 'Yes', 'std_time']
            
            # 2. Check if the Series is NOT empty
            if not filtered_series_std.empty:
                # If data exists, extract the single value and round it.
                std_time_with_festival = np.round(filtered_series_std.item(), 2)
                # Format the output value
                metric_value = f'{std_time_with_festival} min'
            else:
                # If the data is missing (empty Series), assign a fallback value.
                std_time_with_festival = 0.0
                metric_value = 'N/A' # Display 'N/A' in the Streamlit metric
            
            # 3. Pass the safely determined value to the Streamlit metric
            col4.metric('Desvio Padrão Médio de Entrega c/ Festival', metric_value )
            
        with col5:
            df_aux = (df1.loc[:, ['Time_taken(min)', 'Festival']]
                      .groupby( 'Festival')
                      .agg({'Time_taken(min)': ['mean', 'std']} ) )
            
            df_aux.columns = ['avg_time', 'std_time']
            df_aux = df_aux.reset_index()
            # 1. Filter the data for the desired condition (e.g., 'Yes' or 'No')
            filtered_series = df_aux.loc[df_aux['Festival'] == 'Yes', 'avg_time']
            
            # 2. Check if the Series is NOT empty before trying to extract the number
            if not filtered_series.empty:
                # If data exists, extract the single value and round it.
                avg_time_with_festival = np.round(filtered_series.item(), 2)
                # Format the final value for the metric
                metric_value = f'{avg_time_with_festival} min'
            else:
                # If the data is missing (empty Series), assign a safe fallback value.
                metric_value = 'N/A' 
            
            # 3. Pass the safely determined value to the Streamlit metric
            col5.metric('Tempo Médio de Entrega c/ Festival', metric_value )
            
        with col6:
            df_aux = (df1.loc[:, ['Time_taken(min)', 'Festival']]
                      .groupby( 'Festival')
                      .agg({'Time_taken(min)': ['mean', 'std']} ) )
            df_aux.columns = ['avg_time', 'std_time']
            df_aux = df_aux.reset_index()
            # 1. Filter the data for the desired condition (likely 'Yes' for "c/ Festival")
            filtered_series_std = df_aux.loc[df_aux['Festival'] == 'Yes', 'std_time']
            
            # 2. Check if the Series is NOT empty
            if not filtered_series_std.empty:
                # If data exists, extract the single value and round it.
                std_time_with_festival = np.round(filtered_series_std.item(), 2)
                # Format the final value for the metric
                metric_value = f'{std_time_with_festival} min'
            else:
                # If the data is missing (empty Series), assign a safe fallback value.
                metric_value = 'N/A' # Display 'N/A' in the Streamlit metric
            
            # 3. Pass the safely determined value to the Streamlit metric (Line 216)
            col6.metric('Desvio Padrão Médio de Entrega c/ Festival', metric_value )
    
    with st.container():
        st.markdown( """---""" )
        col1, col2 = st.columns( 2 )
        
        with col1:
            df_aux = df1.loc[:, ['City', 'Time_taken(min)']].groupby( 'City' ).agg( {'Time_taken(min)': ['mean', 'std']}) 
            df_aux.columns = ['avg_time', 'std_time']
            df_aux = df_aux.reset_index()
                
            fig = go.Figure()
            fig.add_trace(go.Bar( name='Control', x=df_aux['City'], y=df_aux['avg_time'], error_y=dict(type='data', array=df_aux['std_time']))) 
            fig.update_layout(barmode='group')

            st.plotly_chart( fig )
            
        with col2:
            df_aux = (df1.loc[:, ['City', 'Time_taken(min)', 'Type_of_order']]
                      .groupby(['City', 'Type_of_order'] )
                      .agg({'Time_taken(min)': ['mean', 'std']} ) )
            df_aux.columns = ['avg_time', 'std_time']
            df_aux = df_aux.reset_index()
        
            st.dataframe( df_aux )
        
    with st.container():
        st.markdown( """---""" )
        st.title( 'Distribuição de Tempo' )

        col1, col2 = st.columns( 2 )
        with col1:
            cols = ['Delivery_location_latitude', 'Delivery_location_longitude', 'Restaurant_latitude', 'Restaurant_longitude'] 
            df1['distance'] = df1.loc[:, cols].apply(lambda x: 
                                        haversine( (x['Restaurant_latitude'], x['Restaurant_longitude']),
                                                   (x['Delivery_location_latitude'], x['Delivery_location_longitude']) ), axis=1 )
        
            avg_distance = df1.loc[:, ['City', 'distance']].groupby( 'City' ).mean().reset_index()
            fig = go.Figure( data=[ go.Pie( labels=avg_distance['City'], values=avg_distance['distance'], pull=[0, 0.1, 0])]) 
            st.plotly_chart( fig )
            
        with col2:
            # Data Aggregation
            df_aux = ( df1.loc[:, ['City', 'Time_taken(min)', 'Road_traffic_density']]
                          .groupby(['City', 'Road_traffic_density'] )
                          .agg({'Time_taken(min)': ['mean', 'std']} ) )
            df_aux.columns = ['avg_time', 'std_time']
            df_aux = df_aux.reset_index()
        
            # --- Check for Empty DataFrame before Plotting ---
            if not df_aux.empty:
                # Calculate midpoint safely (only if data exists)
                midpoint_value = np.average(df_aux['std_time'].dropna() ) 
                
                fig = px.sunburst(
                    df_aux, 
                    path=['City', 'Road_traffic_density'], 
                    values='avg_time', 
                    color='std_time', 
                    color_continuous_scale='RdBu',
                    color_continuous_midpoint=midpoint_value
                )
                st.plotly_chart( fig )
            else:
                # Display a warning if there is no data
                st.warning("Não há dados disponíveis para criar o Gráfico Sunburst.")
        
        






















