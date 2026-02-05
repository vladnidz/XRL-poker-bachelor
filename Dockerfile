FROM python:3.11-slim

WORKDIR /app

# System deps for open_spiel and eval7 C extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Default: launch Streamlit UI
CMD ["streamlit", "run", "src/ui/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
