# app.py
from flask import Flask, render_template, request, jsonify
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import torch
import os
import joblib
import numpy as np
import pandas as pd
from difflib import get_close_matches
import requests_cache
from retry_requests import retry
import openmeteo_requests

# create a Flask app
app = Flask(__name__)

# Elimina la variable de entorno DEBUG_MODE
os.environ.pop("DEBUG_MODE", None)

# Load the documents
loader = DirectoryLoader("data/", glob="*.pdf", loader_cls=PyPDFLoader, use_multithreading=True)
documents = loader.load()

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# Check if CUDA is available, else use CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Creating the Embeddings and Vector Store
embeddings = HuggingFaceEmbeddings(
    model_name="recobo/agriculture-bert-uncased",
    model_kwargs={"device": device},
)

vector_store = FAISS.from_documents(text_chunks, embeddings)

# Load the model
llm = Ollama(model="llama3.2")

# Load the memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    memory=memory,
)

# Render the template
@app.route("/")
def index():
    return render_template("index.html")

# Posting the user query
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["user_input"]
    result = chain({"question": user_input, "chat_history": []})
    return {"answer": result["answer"]}

# Search for city and get coordinates
@app.route("/search_city", methods=["POST"])
def search_city():
    city_name = request.form["city_name"]
    coordinates = find_city_coordinates(city_name)
    if coordinates is None:
        return jsonify({"error": "Ciudad no encontrada"}), 404
    temperature, humidity = get_climate_data(coordinates)
    if temperature is None or humidity is None:
        return jsonify({"error": "No se pudo obtener datos climáticos."}), 500
    prediction = predict_crop(temperature, humidity)
    return jsonify({"prediction": prediction, "coordinates": coordinates})

def find_city_coordinates(city_name):
    # Cargar los datos de la ciudad
    city_data = load_city_data("model/World_Cities_Location_table.csv")
    
    # Buscar coincidencias cercanas para el nombre de la ciudad
    close_matches = get_close_matches(city_name, city_data.iloc[:, 1], n=1, cutoff=0.6)

    if close_matches:
        # Obtener la primera coincidencia más cercana
        matched_city = close_matches[0]
        # Obtener la latitud y longitud de la ciudad encontrada
        matched_row = city_data[city_data.iloc[:, 1] == matched_city].iloc[0]
        latitude = matched_row.iloc[2]
        longitude = matched_row.iloc[3]
        return latitude, longitude
    else:
        # Si no se encuentra la ciudad, devuelve las coordenadas de Tijuana por defecto
        return 32.5333333, -117.0166702

def load_city_data(file_path):
    # Cargar el archivo CSV con el delimitador correcto
    df = pd.read_csv(file_path, sep=';', header=None)
    return df

def get_climate_data(coordinates):
    # Configuración de la sesión con caché y reintentos
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Parámetros para la solicitud
    url = "https://climate-api.open-meteo.com/v1/climate"
    params = {
        "latitude": coordinates[0],  # Coordenadas de la ciudad
        "longitude": coordinates[1],
        "start_date": "2020-01-01",
        "end_date": "2024-06-09",
        "models": ["CMCC_CM2_VHR4", "FGOALS_f3_H", "HiRAM_SIT_HR", "MRI_AGCM3_2_S", "EC_Earth3P_HR", "MPI_ESM1_2_XR", "NICAM16_8S"],
        "daily": ["temperature_2m_mean", "relative_humidity_2m_mean"],  # Variables que quieres obtener
    }

    try:
        # Hacer la solicitud a la API
        responses = openmeteo.weather_api(url, params=params)

        # Verificar que se recibieron respuestas
        if not responses:
            raise ValueError("No se recibió respuesta de la API.")

        # Procesar el primer resultado (puedes agregar un bucle si hay múltiples respuestas)
        response = responses[0]
        daily = response.Daily()

        # Obtener los valores de temperatura y humedad
        daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
        daily_relative_humidity_2m_mean = daily.Variables(1).ValuesAsNumpy()

        # Crear un rango de fechas utilizando los datos de la API
        daily_data = {
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            )
        }

        # Asignar las variables de temperatura y humedad
        daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
        daily_data["relative_humidity_2m_mean"] = daily_relative_humidity_2m_mean

        # Crear un DataFrame de Pandas con los datos obtenidos
        daily_dataframe = pd.DataFrame(data=daily_data)
        daily_dataframe['date'] = daily_dataframe['date'].dt.strftime('%Y-%m-%d')

        # Filtrar el último mes en los datos
        daily_dataframe['date'] = pd.to_datetime(daily_dataframe['date'])
        last_date = daily_dataframe['date'].max()
        first_day_last_month = (last_date - pd.DateOffset(months=1)).replace(day=1)

        # Filtrar el último mes
        last_month_data = daily_dataframe[(daily_dataframe['date'] >= first_day_last_month) & (daily_dataframe['date'] <= last_date)]

        # Calcular las medias del último mes
        mean_temperature_last_month = last_month_data['temperature_2m_mean'].mean()
        mean_humidity_last_month = last_month_data['relative_humidity_2m_mean'].mean()

        return mean_temperature_last_month, mean_humidity_last_month

    except Exception as e:
        print(f"Error al obtener los datos climáticos: {e}")
        return None, None

def predict_crop(temperature, humidity):
    # Cargar el modelo de predicción
    scaler = joblib.load('model/knn/scaler.pkl')
    knn = joblib.load('model/knn/knn_model.pkl')

    # Crear un array para los nuevos datos de entrada (temperatura, humedad)
    X_new = np.array([[temperature, humidity]])

    # Preprocesar los nuevos datos escalándolos
    X_new_scaled = scaler.transform(X_new)

    # Hacer la predicción
    y_new_pred = knn.predict(X_new_scaled)

    # Devolver la predicción del cultivo
    return y_new_pred[0]

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
