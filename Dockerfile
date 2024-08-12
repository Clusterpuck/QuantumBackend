# Use the official Python image as the base image
FROM python:3.11.9

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/app/src

# Set the working directory in the container
WORKDIR /src

# Copy the dependencies file to the working directory
COPY requirements.txt .
COPY ./src /app/src

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000 2222

# Copy the rest of the application code to the working directory
COPY . .

# Command options to run the application

# Line to use when implemented with gunicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]

# For a non-gunicorn implementation
# CMD ["python3", "app2.py"]
