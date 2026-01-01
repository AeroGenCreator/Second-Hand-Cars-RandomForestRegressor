# Libraries And Modules
import streamlit as st
import pandas as pd
import json
import joblib
from plotly import express as px
import duckdb
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

st.set_page_config(layout='wide',page_icon='🚙',page_title='Estimate Car Prices Model')
MODEL_PATH = 'RandomForestRegressor.pkl'
TRANSFORRMER_PATH = 'PowerTransformer.pkl'
SCALER_PATH = 'StandardScaler.pkl'
ENCODER_PATH = 'TargetEncoder.pkl' 
DATA_PATH = 'performance_data.csv'
UNIQUE_VALUES_PATH = 'unique_values_columns.json'
METADATA_PATH = 'metadata.json'

@st.cache_data
def load_model(path=MODEL_PATH):
    model = joblib.load(filename=path)
    return model
MODEL = load_model()

@st.cache_data
def load_tools(path_tranformer=TRANSFORRMER_PATH,path_scaler=SCALER_PATH,path_encoder=ENCODER_PATH):
    transformer = joblib.load(path_tranformer)
    scaler = joblib.load(path_scaler)
    encoder = joblib.load(path_encoder)
    return transformer, encoder, scaler

@st.cache_data
def load_performance_data(path=DATA_PATH):
    data = pd.read_csv(path)
    return data

@st.cache_data
def load_columns_values(path=UNIQUE_VALUES_PATH):
    with open(UNIQUE_VALUES_PATH,'r',encoding='utf-8') as f:
        data = json.load(f)
        return data

@st.cache_data
def load_metadata(path=METADATA_PATH):
    with open(METADATA_PATH,'r',encoding='utf-8') as f:
        data = json.load(f)
        return data

def model_behavior(data):

    st.subheader(':material/network_intel_node: Distribution Behavior For This Model')
    # melt de data
    query = duckdb.sql("""
    SELECT
        stage,
        values
    FROM
        data
    UNPIVOT(
        values
        FOR stage
        IN (Price, Predictions)
    );
    """).df()
    
    fig = px.histogram(data_frame=query,marginal='box',color='stage')
    st.plotly_chart(fig)
    mae = np.round(mean_absolute_error(data['Price'],data['Predictions']))
    rmse = np.round(np.sqrt(mean_squared_error(data['Price'],data['Predictions'])))
    r2 = r2_score(data['Price'],data['Predictions'])

    st.divider()
    with st.expander('Check Scores For This Model'):
        st.dataframe(pd.DataFrame(data=[['€ '+str(mae),'€ '+str(rmse),str(round(r2*100,2))+'%']],columns=['MAE','RMSE','R2'],index=['Scores']))

def estimate_price(data,transformer,encoder,scaler,logarithm,left_tail,categories,numeric,order,metadata):
    
    # Tranformamos los datos:
    X = pd.DataFrame(data=data,columns=order).copy()
    X[logarithm] = np.log1p(X[logarithm])
    X[[left_tail]] = transformer.transform(X[[left_tail]])
    X[categories] = encoder.transform(X[categories])
    X[numeric] = scaler.transform(X[numeric])

    # Obtenemos la prediccion por arbol para posteriormente obtner los percentiles 5% y 95% para el I.C.
    tree_prediction = np.array([tree.predict(X) for tree in MODEL.estimators_])
    tree_prediction_real = np.expm1(tree_prediction)
    lower_bound = np.percentile(tree_prediction_real, 5)
    upper_bound = np.percentile(tree_prediction_real, 95)

    # Predicción base en Euros (revertimos el logaritmo)
    prediction_eur = np.expm1(MODEL.predict(X))[0] # Accedemos al primer elemento del array.
    
    # Convertir a pesos de 2016
    price_mxn = prediction_eur * metadata['euro_to_mxn']

    return {
        'eur': prediction_eur,
        'mxn': price_mxn,
        'lower': lower_bound,
        'upper': upper_bound
    }

def new_entry(unique_values):
    entry =  st.form(key='my_form')
    with entry:
        col1, col2 = st.columns(2)
        with col1:
            VehicleType = st.selectbox(options=unique_values['VehicleType'],accept_new_options=False,index=0,label='VehicleType')
            Gearbox = st.selectbox(options=unique_values['Gearbox'],accept_new_options=False,index=0,label='Gearbox')
            Model = st.selectbox(options=unique_values['Model'],accept_new_options=False,index=0,label='Model')
            FuelType = st.selectbox(options=unique_values['FuelType'],accept_new_options=False,index=0,label='FuelType')
            Brand = st.selectbox(options=unique_values['Brand'],accept_new_options=False,index=0,label='Brand')
        with col2:
            NotRepaired = st.selectbox(options=unique_values['NotRepaired'],accept_new_options=False,index=0,label='NotRepaired')
            Power = st.number_input(min_value=0,max_value=20000,label='Power')
            Mileage = st.number_input(min_value=5000,max_value=150000,label='Mileage')
            Age = st.number_input(min_value=6,max_value=115,label='Age')

    submitted = entry.form_submit_button(label='Submit',key='submit')
    if submitted:
        return {
            'VehicleType':[VehicleType],
            'Gearbox':[Gearbox],
            'Model':[Model],
            'FuelType':[FuelType],
            'Brand':[Brand],
            'NotRepaired':[NotRepaired],
            'Power':[Power],
            'Mileage':[Mileage],
            'Age':[Age]
        }

# --- Interfaz
st.title(':material/car_tag: Estimate Second Hand Car Prices - Regression Model ')
st.divider()

metadata = load_metadata()
unique_values = load_columns_values()
transformer, encoder, scaler = load_tools()
data = load_performance_data()

col1, col2 = st.columns([1.6,2.4])

with col1:
    st.subheader(':material/assignment_add: Type the car data you want to estimate')
    form = new_entry(unique_values=unique_values)
    if isinstance(form,dict):
        result = estimate_price(
            data=form,
            transformer=transformer,
            encoder=encoder,
            scaler=scaler,
            logarithm=metadata['logarithm_columns'],
            left_tail=metadata['power_transformer_columns'],
            categories=metadata['categoric_columns'],
            numeric=metadata['numeric_columns'],
            order=metadata['columns_order'],
            metadata=metadata
            )

        st.subheader(':material/payments: Market Valuation (Mexico 2025)')
        
        # Mostrar métricas principales
        m1, m2 = st.columns(2)
        m1.metric(label="Estimated Price (MXN)", value=f"${result['mxn']:,.2f}")
        m2.metric(label="Original Price (EUR)", value=f"€{result['eur']:,.2f}")
        
        # Mostrar Intervalo de Confianza con mejor formato
        st.info(f"""
        **Confidence Interval (90%)** Range: `$ {result['lower']*metadata['euro_to_mxn']:,.2f}` — `$ {result['upper']*metadata['euro_to_mxn']:,.2f}`  
        *This range reflects price variations based on car condition and market volatility.*
        """)
        
        # Opcional: Una barra visual para el rango
        st.write("Current Value Position:")
        st.progress(min((result['eur']/result['upper']),1.0), text="Target Price")

with col2:
    model_behavior(data=data)
    with st.expander('About This Model'):
        st.markdown("""
    # 📘 Model Appendix: The Intelligence Behind the Price

    ### 🏗️ Architecture & Methodology
    The core of this application is a **Random Forest Regressor** ensemble consisting of **200 decision trees**. Unlike a static pricing table, this model captures complex non-linear relationships between a vehicle's features to determine its current market value.

    

    #### Data Inference Pipeline
    To ensure high-fidelity predictions, every query undergoes a specific mathematical transformation pipeline before reaching the model:
    1.  **Target Encoding:** Categorical variables (Brand, Model, etc.) are transformed into numerical values that preserve label information without increasing dimensionality.
    2.  **Logarithmic Scaling:** We apply `log1p` to `Power` and `Age` to stabilize variance and mitigate the impact of extreme outliers.
    3.  **Power Transformation:** The **Mileage** feature is processed via `PowerTransformer` to normalize its distribution, which is critical for accurate regression.

    ---

    ### 📊 Performance & Benchmarking
    After evaluating several state-of-the-art algorithms, **Random Forest** was selected for its superior balance between precision and reliability.

    | Model | R² Score (Accuracy) | Key Insight |
    | :--- | :--- | :--- |
    | **Random Forest Regressor** | **0.86** | **Winner: Highest precision and robust generalization.** |
    | Gradient Boosting (XGB/LGBM) | 0.84 - 0.85 | Exceptional speed but slightly less stable on outliers. |
    | Linear Regression (Base) | 0.49 | Baseline; struggled with complex market interactions. |

    

    > **Key Achievement:** The model maintains a **Mean Absolute Error (MAE) of €970**. This ensures that users receive a valuation within a fair market margin, fulfilling the "Rusty Bargain" promise of transparency.

    ---

    ### 🔍 Technical Specifications (Metadata)
    For technical auditors and recruiters, here is the underlying environment configuration:

    * **Data Source:** eBay Kleinanzeigen (Germany).
    * **Snapshot Date:** April 2016.
    * **Currency Logic:** Trained in **EUR (€)**; converted to **MXN** using a fixed historical rate ($21.13$).
    * **Temporal Gap:** A 9-year difference exists between the training data and 2025. Technical depreciation (Age/Mileage) is prioritized over monetary inflation to reflect real-world vehicle devaluation.
    * **Feature Set:** * *Numerical:* Power (HP), Mileage (km), Age (Years).
        * *Categorical:* Vehicle Type, Gearbox, Model, Fuel Type, Brand, and Repair Status.


    ---

    ### 💡 Interpreting the Confidence Interval (C.I.)
    We don't just provide a single number; we provide a **90% Confidence Range**.
    * **Narrow Range:** Indicates high model certainty due to an abundance of similar historical data.
    * **Wide Range:** Suggests the price may fluctuate significantly based on the vehicle's physical condition or market rarity—factors that require a human expert's touch.

    ---
    Developed as part of my Data Science Portfolio: :material/public: [website](https://fringe-edge-3f8.notion.site/Andres-Lopez-M-Data-Scientist-Portafolio-2ca8851a844d80d0b61bd0c7b20a65f2?pvs=74), :material/bookmark_stacks: [Git-Hub](https://github.com/AeroGenCreator)
                    
    """)