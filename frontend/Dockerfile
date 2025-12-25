FROM python:3.9-slim

WORKDIR /app/frontend

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование приложения
COPY UI.py .

# Открытие порта для Streamlit
EXPOSE 8501

# Запуск приложения
CMD ["streamlit", "run", "UI.py", "--server.port=8501", "--server.address=0.0.0.0"]