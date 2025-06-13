# Gunakan base image Python 3.12 slim (sesuaikan versi Python kamu)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy semua file project ke container
COPY . .

# Install dependencies dari requirements.txt
RUN pip install --default-timeout=500 --retry 5 --no-cache-dir -r requirements.txt

# Ekspos port 5000 (sesuai Flask app)
EXPOSE 5000

# Command untuk menjalankan Flask app
CMD ["python", "app.py"]
