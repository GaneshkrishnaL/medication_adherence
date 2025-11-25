FROM python:3.11-slim

WORKDIR /app

# Install system deps (optional but good practice)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 5000

# Default command: run Flask API
CMD ["python", "src/api/app.py"]
