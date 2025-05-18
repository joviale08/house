import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np
from datetime import datetime
import io
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

@st.cache_data
def load_ml_model():
    with open("best_models.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_dl_model():
    model = load_model("ann_model.keras")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_dataset():
    df = pd.read_csv("House-Data.csv")
    df.drop(columns=[col for col in df.columns if col.lower() in ['id']], inplace=True)
    
    # Gestion des outliers
    for col in df.select_dtypes(include=['number']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    
    # Conversion de la date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df.drop(columns=['date'], inplace=True)
    
    df.dropna(inplace=True)
    return df

def preprocess_uploaded_df(df):
    df.drop(columns=[col for col in df.columns if col.lower() in ['id']], inplace=True, errors='ignore')
    
    # Gestion des outliers
    for col in df.select_dtypes(include=['number']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    
    # Conversion de la date si présente
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df.drop(columns=['date'], inplace=True)
    
    df.dropna(inplace=True)
    return df

def preprocess_for_ann(input_df, original_df, scaler):
    # Copier le DataFrame pour éviter de modifier l'original
    df_processed = input_df.copy()
    
    # Supprimer la colonne 'price' si elle existe
    if 'price' in df_processed.columns:
        df_processed = df_processed.drop(columns=['price'])
    
    # Convertir 'zipcode' en variables dummy
    cat_cols = ['zipcode']
    df_processed = pd.get_dummies(df_processed, columns=cat_cols, drop_first=True)
    
    # S'assurer que les colonnes correspondent à celles utilisées lors de l'entraînement
    # Récupérer les colonnes d'entraînement à partir du dataset original
    X = original_df.drop('price', axis=1)
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    training_columns = X.columns
    
    # Ajouter les colonnes manquantes avec des valeurs 0
    for col in training_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0
    
    # Réorganiser les colonnes dans le même ordre que lors de l'entraînement
    df_processed = df_processed[training_columns]
    
    # Normaliser les données avec le scaler
    X_scaled = scaler.transform(df_processed)
    
    return X_scaled

def build_prediction_form(df, key_prefix=""):
    st.subheader("📝 Formulaire de prédiction")

    # Initialiser session_state pour stocker les valeurs du formulaire
    if f'form_values_{key_prefix}' not in st.session_state:
        st.session_state[f'form_values_{key_prefix}'] = {}

    input_data = st.session_state[f'form_values_{key_prefix}'].copy()

    with st.container():
        # Section 1 : Informations temporelles
        with st.expander("📅 Informations temporelles", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                default_date = datetime.now().date()
                if 'year' in input_data:
                    default_date = datetime(int(input_data['year']), int(input_data['month']), int(input_data['day']))
                selected_date = st.date_input("Date de l'annonce", value=default_date, 
                                              key=f"{key_prefix}date_input", 
                                              help="Date de mise en vente")
                input_data['year'] = selected_date.year
                input_data['month'] = selected_date.month
                input_data['day'] = selected_date.day

        # Section 2 : Caractéristiques du bien
        st.markdown("#### 🏠 Caractéristiques du bien")
        key_features = [col for col in df.columns if any(k in col.lower() for k in ['bed', 'bath', 'floor'])]
        excluded_cols = ['sqft_living15', 'sqft_lot15', 'long', 'lat']
        other_features = [col for col in df.columns 
                         if col.lower() not in ['price', 'year', 'month', 'day', 'yr_built', 'yr_renovated'] 
                         and col not in key_features and col not in excluded_cols]

        with st.container():
            st.markdown("**Caractéristiques principales**")
            col1, col2, col3 = st.columns(3)
            for i, col in enumerate(key_features):
                with [col1, col2, col3][i % 3]:
                    min_val = 1 if col in ['bedrooms', 'bathrooms'] else 0
                    default_val = int(input_data.get(col, min_val))
                    val = st.number_input(f"{col}", min_value=min_val, max_value=20, step=1, format="%d", 
                                         value=default_val, key=f"{key_prefix}input_{col}", 
                                         help=f"Nombre de {col.lower()} (max 20)")
                    if col in ['bedrooms', 'bathrooms'] and val < 1:
                        st.error(f"❌ Le champ {col} doit être supérieur ou égal à 1. Veuillez corriger.")
                    if val > 10:
                        st.warning(f"⚠️ {col} semble élevé ({val}). Vérifiez la valeur.")
                    input_data[col] = val

        with st.expander("Autres caractéristiques", expanded=False):
            col1, col2 = st.columns(2)
            for i, col in enumerate(other_features):
                with col1 if i % 2 == 0 else col2:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        is_integer = pd.api.types.is_integer_dtype(df[col]) or any(k in col.lower() for k in ['bed', 'bath', 'floor', 'waterfront', 'view', 'condition', 'grade'])
                        default_val = input_data.get(col, df[col].median())
                        if is_integer:
                            default_val = int(default_val)
                            val = st.number_input(f"{col}", min_value=0, max_value=int(df[col].max()), step=1, format="%d",
                                                 value=default_val, key=f"{key_prefix}input_{col}", 
                                                 help=f"Valeur médiane : {df[col].median():.1f}")
                        else:
                            default_val = float(default_val)
                            val = st.number_input(f"{col}", min_value=float(df[col].min()), max_value=float(df[col].max()),
                                                 value=default_val, key=f"{key_prefix}input_{col}", 
                                                 help=f"Valeur médiane : {df[col].median():.1f}")
                        input_data[col] = val
                    else:
                        options = df[col].dropna().unique().tolist()
                        default_val = input_data.get(col, options[0])
                        val = st.selectbox(f"{col}", options, index=options.index(default_val) if default_val in options else 0, 
                                           key=f"{key_prefix}input_{col}", help="Choisissez une option")
                        input_data[col] = val

        # Section 3 : Infos de construction
        with st.expander("🛠️ Infos de construction", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                default_yr_built = int(input_data.get('yr_built', df['yr_built'].median()))
                input_data["yr_built"] = st.number_input("Année de construction", min_value=1800, max_value=2025, step=1, 
                                                        format="%d", value=default_yr_built, 
                                                        key=f"{key_prefix}input_yr_built",
                                                        help="Année de construction de la maison")
            with col2:
                default_renov = input_data.get('yr_renovated', 0)
                renov_option = st.radio("Rénovée ?", ("Non", "Oui"), 
                                       index=0 if default_renov == 0 else 1, 
                                       key=f"{key_prefix}renov_option")
                input_data["yr_renovated"] = 0 if renov_option == "Non" else st.number_input(
                    "Année de rénovation", min_value=1900, max_value=2025, step=1, format="%d",
                    value=default_renov if default_renov > 0 else 1900, 
                    key=f"{key_prefix}input_yr_renovated",
                    help="Année de la dernière rénovation"
                )

    # Pré-remplir les colonnes exclues avec leurs médianes
    for col in excluded_cols:
        if col in df.columns:
            input_data[col] = float(df[col].median()) if pd.api.types.is_numeric_dtype(df[col]) else df[col].mode()[0]

    st.session_state[f'form_values_{key_prefix}'] = input_data
    return pd.DataFrame([input_data])

def apply_filters(df):
    st.sidebar.header("🎛️ Filtres")

    # Initialiser les filtres dans session_state
    if 'filters' not in st.session_state:
        st.session_state.filters = {}

    filtered_df = df.copy()

    # Bouton de réinitialisation
    if st.sidebar.button("🔄 Réinitialiser les filtres", help="Remet tous les filtres à leurs valeurs par défaut"):
        st.session_state.filters = {}
        st.rerun()

    # Filtre : Prix
    st.sidebar.markdown("### 💰 Prix")
    min_price, max_price = float(df['price'].min()), float(df['price'].max())
    default_price = st.session_state.filters.get('price', (min_price, max_price))
    price_range = st.sidebar.slider("Fourchette de prix (€)", min_price, max_price, default_price, 
                                   step=1000.0, key="filter_price", help="Filtrer par prix de la propriété")
    st.session_state.filters['price'] = price_range
    filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & (filtered_df['price'] <= price_range[1])]

    # Filtre : Nombre de chambres
    st.sidebar.markdown("### 🏠 Caractéristiques")
    if 'bedrooms' in df.columns:
        min_bed, max_bed = int(df['bedrooms'].min()), int(df['bedrooms'].max())
        default_bed = st.session_state.filters.get('bedrooms', (min_bed, max_bed))
        bed_range = st.sidebar.slider("Nombre de chambres", min_bed, max_bed, default_bed, 
                                     step=1, key="filter_bedrooms", help="Filtrer par nombre de chambres")
        st.session_state.filters['bedrooms'] = bed_range
        filtered_df = filtered_df[(filtered_df['bedrooms'] >= bed_range[0]) & (filtered_df['bedrooms'] <= bed_range[1])]

    # Filtre : Nombre de salles de bain
    if 'bathrooms' in df.columns:
        min_bath, max_bath = float(df['bathrooms'].min()), float(df['bathrooms'].max())
        default_bath = st.session_state.filters.get('bathrooms', (min_bath, max_bath))
        bath_range = st.sidebar.slider("Nombre de salles de bain", min_bath, max_bath, default_bath, 
                                      step=0.25, key="filter_bathrooms", help="Filtrer par nombre de salles de bain")
        st.session_state.filters['bathrooms'] = bath_range
        filtered_df = filtered_df[(filtered_df['bathrooms'] >= bath_range[0]) & (filtered_df['bathrooms'] <= bath_range[1])]

    # Filtre : Année de construction
    st.sidebar.markdown("### 🗓️ Année")
    min_year, max_year = int(df['yr_built'].min()), int(df['yr_built'].max())
    default_year = st.session_state.filters.get('yr_built', (min_year, max_year))
    year_range = st.sidebar.slider("Année de construction", min_year, max_year, default_year, 
                                  step=1, key="filter_yr_built", help="Filtrer par année de construction")
    st.session_state.filters['yr_built'] = year_range
    filtered_df = filtered_df[(filtered_df['yr_built'] >= year_range[0]) & (filtered_df['yr_built'] <= year_range[1])]

    # Filtre : Zipcode
    st.sidebar.markdown("### 📍 Localisation")
    if 'zipcode' in df.columns:
        zipcodes = df['zipcode'].astype(str).unique().tolist()
        default_zip = st.session_state.filters.get('zipcode', zipcodes)
        selected_zip = st.sidebar.multiselect("Code postal", zipcodes, default=default_zip, 
                                             key="filter_zipcode", help="Filtrer par code postal")
        st.session_state.filters['zipcode'] = selected_zip
        if selected_zip:
            filtered_df = filtered_df[filtered_df['zipcode'].astype(str).isin(selected_zip)]

    return filtered_df

def main():
    st.set_page_config(page_title="🏠 Estimation immobilière", layout="wide")
    
    st.markdown("""
        <style>
            .main {background-color: #F5F5F5;}
            .stButton>button {background-color: #1565C0; color: white; border-radius: 5px;}
            .stButton>button:hover {background-color: #de2e0b; color: white;}
            .stTextInput>div>input, .stNumberInput>div>input, .stSelectbox>div>select {
                border: 1px solid #1565C0; border-radius: 5px;
            }
            h1, h2, h3 {color: #212121;}
            .sidebar .sidebar-content {background-color: #E8F0FE;}
            .prediction-box {background-color: #2E7D32; color: white; padding: 20px; border-radius: 10px; text-align: center;}
            .highlight {background-color: #1565C0; color: white; padding: 10px; border-radius: 5px;}
            .animated-button {
                background-color: #1565C0;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
            @keyframes rebound {
                0% { transform: translateY(0); }
                40% { transform: translateY(-10px); }
                60% { transform: translateY(0); }
                80% { transform: translateY(-5px); }
                100% { transform: translateY(0); }
            }
            .stFileUploader {
                background-color: #1565C0;
                color: white;
                padding: 2px;
                border-radius: 5px;
                animation: rebound 1.5s infinite;
            }
            .stDownloadButton button {
                background-color: #1565C0;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                cursor: pointer;
                animation: pulse 2s infinite;
                transition: background-color 0.3s, transform 0.2s;
            }
            .stDownloadButton button:hover {
                background-color: #de2e0b;
                transform: scale(1.05);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='color: #1565C0;'>🔍 Application d'Estimation Immobilière</h1>", unsafe_allow_html=True)

    df = load_dataset()
    ml_model = load_ml_model()
    try:
        ann_model, scaler = load_dl_model()
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle ANN : {e}")
        ann_model, scaler = None, None

    # Appliquer les filtres
    filtered_df = apply_filters(df)

    # Initialiser l’historique des prédictions
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []

    tab1, tab2, tab3, tab5, tab4 = st.tabs(["📁 Données", "📊 Statistiques", "🔮 Prédiction (ML)", "🧠 Deep_prédiction (DL)", "📜 Historique"])

    with tab1:
        st.subheader("📁 Données Nettoyées")
        st.write(f"Résultats : ")
        uploaded_file = st.file_uploader("Charger un fichier CSV", type="csv", key="file_uploader")
        if uploaded_file is not None:
            uploaded_df = pd.read_csv(uploaded_file)
            filtered_df = preprocess_uploaded_df(uploaded_df)
            st.success("Fichier CSV chargé et prétraité avec succès !")
    
            st.rerun()
        st.dataframe(filtered_df, use_container_width=True)

    with tab2:
        st.subheader("📊 Statistiques Descriptives")
        st.markdown(
            f"<div class='highlight'>Total : {len(filtered_df)} propriétés | Prix moyen : {filtered_df['price'].mean():,.2f} €</div>", 
            unsafe_allow_html=True
        )
        st.write(f"Résultats : ")
        with st.expander("Voir les statistiques détaillées", expanded=False):
            st.dataframe(filtered_df.describe().round(2), use_container_width=True)

        st.markdown("### 📈 Visualisations")
        num_cols = filtered_df.select_dtypes(include='number').columns.tolist()
        col_selected = st.selectbox("Choisir une variable", num_cols, key="hist_select", 
                                    help="Sélectionnez une variable pour visualiser sa distribution")
        st.markdown("*Histogramme : montre la répartition des valeurs.*")
        fig = px.histogram(filtered_df, x=col_selected, title=f"Distribution de {col_selected}", 
                          color_discrete_sequence=["#1565C0"], nbins=30)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("*Boxplot : montre la médiane, les quartiles et les valeurs aberrantes.*")
        fig_box = px.box(filtered_df, y=col_selected, title=f"Boxplot de {col_selected}", 
                         color_discrete_sequence=["#1565C0"])
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("### 📉 Relation entre variables")
        col_x = st.selectbox("Variable X", num_cols, key="scatter_x", help="Choisissez la variable pour l’axe X")
        col_y = st.selectbox("Variable Y", num_cols, index=num_cols.index('price') if 'price' in num_cols else 0, 
                             key="scatter_y", help="Choisissez la variable pour l’axe Y")
        fig_scatter = px.scatter(filtered_df, x=col_x, y=col_y, title=f"{col_x} vs {col_y}", 
                                 color_discrete_sequence=["#1565C0"])
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        input_df = build_prediction_form(df, key_prefix="ml_")
        bedrooms_valid = input_df.get('bedrooms', [0])[0] >= 1
        bathrooms_valid = input_df.get('bathrooms', [0])[0] >= 1
        can_predict = bedrooms_valid and bathrooms_valid
        
        if st.button("🔮 Prédire la valeur ", disabled=not can_predict):
            with st.spinner("Calcul de l’estimation..."):
                try:
                    prediction = ml_model.predict(input_df)
                    st.markdown(
                        f"<div class='prediction-box'>"
                        f"<h3>🏷️ Estimation du bien (ML) : {prediction[0]:,.2f} €</h3>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    st.session_state.predictions.append({
                        'input': input_df.to_dict('records')[0],
                        'price': prediction[0],
                        'model': 'Machine Learning'
                    })
                except Exception as e:
                    st.error(f"❌ Erreur de prédiction (ML) : {e}")

    with tab5:
        st.subheader("🧠 Prédiction avec Deep Learning (ANN)")
        if ann_model is None or scaler is None:
            st.error("Le modèle ANN n'a pas pu être chargé. Veuillez vérifier les fichiers 'ann_model.keras' et 'scaler.pkl'.")
        else:
            input_df = build_prediction_form(df, key_prefix="dl_")
            bedrooms_valid = input_df.get('bedrooms', [0])[0] >= 1
            bathrooms_valid = input_df.get('bathrooms', [0])[0] >= 1
            can_predict = bedrooms_valid and bathrooms_valid
            
            if st.button("🧠 Prédire la valeur ", disabled=not can_predict):
                with st.spinner("Calcul de l’estimation avec le modèle ANN..."):
                    try:
                        # Préparer les données pour l'ANN
                        X_scaled = preprocess_for_ann(input_df, df, scaler)
                        # Faire la prédiction
                        prediction = ann_model.predict(X_scaled, verbose=0)
                        st.markdown(
                            f"<div class='prediction-box'>"
                            f"<h3>🏷️ Estimation du bien (DL) : {prediction[0][0]:,.2f} €</h3>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        st.session_state.predictions.append({
                            'input': input_df.to_dict('records')[0],
                            'price': float(prediction[0][0]),
                            'model': 'Deep Learning'
                        })
                    except Exception as e:
                        st.error(f"❌ Erreur de prédiction (DL) : {e}")

    with tab4:
        st.subheader("📜 Historique des prédictions")
        if st.session_state.predictions:
            pred_df = pd.DataFrame([{
                'Modèle': p['model'],
                'Prix estimé': p['price'],
                **p['input']
            } for p in st.session_state.predictions])
            st.dataframe(pred_df, use_container_width=True)
            if st.button("🗑️ Effacer l’historique", help="Supprime toutes les prédictions sauvegardées"):
                st.session_state.predictions = []
                st.rerun()
            # Bouton pour exporter l'historique
            csv_buffer = io.StringIO()
            pred_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            st.download_button(
                label="📥 Exporter l'historique en CSV",
                data=csv_data,
                file_name="historique_predictions.csv",
                mime="text/csv",
                help="Téléchargez l'historique des prédictions sous forme de fichier CSV"
            )
        else:
            st.info("Aucune prédiction enregistrée pour le moment.")

if __name__ == "__main__":
    main()