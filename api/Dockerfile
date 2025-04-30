FROM python:3.9-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data for TextBlob
RUN python -m textblob.download_corpora

# Copy application code
COPY app.py .

# Create directory for uploads if needed
RUN mkdir -p uploads

# Run the application
CMD ["python", "app.py"]