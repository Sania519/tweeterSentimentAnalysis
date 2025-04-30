FROM python:3.9-slim

WORKDIR /app

# Copy requirements file
COPY client_requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r client_requirements.txt

# Copy application code
COPY client_ui.py .

# Create needed directories
RUN mkdir -p templates
RUN mkdir -p uploads
RUN mkdir -p results

# Copy templates
COPY templates/ ./templates/

# Run the client UI application
CMD ["python", "client_ui.py"]