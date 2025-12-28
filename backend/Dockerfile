FROM python:3.9-slim

WORKDIR /app/backend

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y \
    libgl1\
    libglx-mesa0\
    libglib2.0-0t64\
    libsm6\
    libxext6\
    libxrender1\
    libgl1-mesa-dri\
    && rm -rf /var/lib/apt/lists/*

# Копирование весов
COPY yolov11n-face.pt .
# Копирование сервиса BentoML
COPY service.py .
COPY embedding.py .

# Создание bento (если нужно)
# RUN bentoml build

EXPOSE 3000

# Запуск BentoML сервиса
CMD ["bentoml", "serve", "service:svc", "--port", "3000", "--host", "0.0.0.0"]  