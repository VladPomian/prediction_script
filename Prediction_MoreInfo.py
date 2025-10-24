from datetime import datetime
import requests
import pandas as pd
from prophet import Prophet
import xml.etree.ElementTree as ET
import base64

# API-ключ
api_key = "0j8sbNJdINmIyT1dPVmLTxCcPgXLhNxAAfPYahfk"

start_date = "2012-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

# Функция для получения данных из API
def fetch_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"Ошибка запроса API ({url}): {e}"

# Функция для преобразования sourceLocation (например, "S2E6", "N12W99", "S999E999")
def parse_source_location(location):
    if location:
        lat_dir = location[0]
        lon_dir = location[location.find('E') + 1] if 'E' in location else location[location.find('W') + 1]
        
        lat_end = location.find('E') if 'E' in location else location.find('W')
        
        lat = int(location[1:lat_end])
        lon = int(location[lat_end + 1:])

        if lat_dir == 'S':
            lat = -lat
        if lon_dir == 'W':
            lon = -lon

        return lat, lon
    return None, None

# Функция для преобразования classType в числовую интенсивность
def convert_class_type_to_int(class_type):
    if class_type:
        if class_type.startswith('X'):
            return float(class_type[1:]) * 10
        elif class_type.startswith('M'):
            return float(class_type[1:])
        elif class_type.startswith('C'):
            return float(class_type[1:]) / 10
    return 0

# Функция для преобразования времени в объект datetime
def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%MZ").date()
    except ValueError as e:
        return f"Ошибка парсинга даты ({date_str}): {e}"

# Обработка данных
status = "Failure"
try:
    # Получаем данные из API
    cme_url = f'https://api.nasa.gov/DONKI/CMEAnalysis?startDate={start_date}&endDate={end_date}&mostAccurateOnly=true&api_key={api_key}'
    flr_url = f'https://api.nasa.gov/DONKI/FLR?startDate={start_date}&endDate={end_date}&api_key={api_key}'
    gst_url = f'https://api.nasa.gov/DONKI/GST?startDate={start_date}&endDate={end_date}&api_key={api_key}'

    cme_data = fetch_data(cme_url)
    flr_data = fetch_data(flr_url)
    gst_data = fetch_data(gst_url)

    # Проверка ошибок при получении API-данных
    if isinstance(cme_data, str) or isinstance(flr_data, str) or isinstance(gst_data, str):
        raise Exception(f"Ошибка API: {cme_data} | {flr_data} | {gst_data}")

    # Подготовка данных
    cme_df = pd.DataFrame([
        { 
            "ds": datetime.strptime(cme['time21_5'], "%Y-%m-%dT%H:%MZ").date(),
            "y": cme.get('speed', 0)
        } 
        for cme in cme_data if 'time21_5' in cme and 'speed' in cme
    ]).dropna()

    flr_df = pd.DataFrame([
        { 
            "ds": datetime.strptime(flare['peakTime'], "%Y-%m-%dT%H:%MZ").date(),
            "y": convert_class_type_to_int(flare.get('classType', '')),  
            "sourceLocation_lat": parse_source_location(flare.get('sourceLocation', ''))[0],  
            "sourceLocation_lon": parse_source_location(flare.get('sourceLocation', ''))[1]  
        } 
        for flare in flr_data if 'peakTime' in flare and 'classType' in flare
    ]).dropna()

    flr_df["ds"] = pd.to_datetime(flr_df["ds"], errors='coerce')
    flr_df = flr_df[flr_df["ds"].notna()]  
    flr_df["sourceLocation_lat"] = pd.to_numeric(flr_df["sourceLocation_lat"], errors='coerce')
    flr_df["sourceLocation_lon"] = pd.to_numeric(flr_df["sourceLocation_lon"], errors='coerce') 

    gst_df = pd.DataFrame([{ "ds": parse_date(gst['startTime']), "y": max([kp['kpIndex'] for kp in gst.get('allKpIndex', [])], default=0) } for gst in gst_data if 'startTime' in gst]).dropna()

    # Функция для обучения и предсказания
    def train_and_forecast(df, label, periods=365):
        if df.empty:
            return None
        df["ds"] = pd.to_datetime(df["ds"])
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        return forecast[['ds', 'yhat']]

    cme_forecast = train_and_forecast(cme_df, 'CME')
    flr_forecast = train_and_forecast(flr_df, 'FLR')
    gst_forecast = train_and_forecast(gst_df, 'GST')

    # Создание XML
    root = ET.Element("forecast_data")
    forecasts = {"CME": cme_forecast, "FLR": flr_forecast, "GST": gst_forecast}
    
    for key, forecast in forecasts.items():
        if forecast is not None:
            category = ET.SubElement(root, key)
            forecast = forecast[forecast['ds'] > datetime.today()]
            for _, row in forecast.iterrows():
                record = ET.SubElement(category, "record")
                ET.SubElement(record, "date").text = row["ds"].strftime('%Y-%m-%d')
                ET.SubElement(record, "value").text = str(row["yhat"])

    # Преобразование в строку
    xml_str = ET.tostring(root, encoding='utf-8', method='xml')
    encoded_xml = base64.b64encode(xml_str).decode('utf-8')
    xml_size = len(xml_str)

    # Успешный статус
    status = "Success"

except Exception as e:
    error_message = f"Ошибка: {str(e)}"
    xml_str = error_message.encode('utf-8')
    encoded_xml = base64.b64encode(xml_str).decode('utf-8')
    xml_size = len(xml_str)

# Вывод результата
print(f"{encoded_xml} {xml_size} {status}")
