# 1. Use official Python lightweight image
FROM python:3.9-slim

# 2. Prevent Python from writing pyc files to disc and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 3. Install system dependencies
# CHANGED: Replaced 'libgl1-mesa-glx' with 'libgl1' and added 'libglib2.0-0'
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Python dependencies
COPY requirements.txt .
# Install CPU-only version of torch to keep image small
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the entire project directory into the container
COPY . .

# 6. Expose Streamlit port
EXPOSE 8501

# 7. Run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]