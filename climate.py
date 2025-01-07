import requests

def get_weather_data(lat, lon):
    api_key = "YOUR_API_KEY"  # Inserta tu clave de API
    url = f"http://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=current,minutely,hourly,alerts&units=metric&appid={api_key}"
    
    response = requests.get(url)
    data = response.json()
    
    # Aquí deberías extraer los datos climáticos necesarios para tu predicción
    # Por ejemplo, puedes extraer promedios de temperatura, precipitaciones, etc.
    
    # Ejemplo básico de retorno:
    monthly_avg_temp = sum([day["temp"]["day"] for day in data["daily"]]) / len(data["daily"])
    return {"avg_temp": monthly_avg_temp}
