# Берём официальный лёгкий образ Python 3.11
FROM python:3.11-slim

# Устанавливаем ffmpeg системно внутрь образа
# (ключевой момент — именно здесь ffmpeg попадает в контейнер)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Рабочая директория внутри контейнера
WORKDIR /app

# Сначала копируем только список зависимостей (для кэша слоёв)
COPY requirements.txt .

# Устанавливаем Python-библиотеки
RUN pip install --no-cache-dir -r requirements.txt

# Копируем сам код приложения
COPY app.py .

# Команда запуска: слушаем 0.0.0.0 и порт, который даст Render через $PORT
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
