# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Установка системных зависимостей (если нужны дополнительные, добавьте сюда)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
       build-essential \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копируем зависимости и устанавливаем
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальной код и модель
COPY src/ src/
COPY models/ models/
COPY predictions/ predictions/

# По желанию: создаём пользователя с низкими правами
# RUN useradd --create-home appuser
# USER appuser

# Открываем порт
EXPOSE 8000

# Команда запуска
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
