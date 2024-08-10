# Use the official Python image as the base image
FROM python:3.9

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /src

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install the ODBC driver for SQL Server
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list
RUN apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql17

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000 2222

# Copy the rest of the application code to the working directory
COPY . .

# Command options to run the application

# Line to use when implemented with gunicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# For a non-gunicorn implementation
# CMD ["python3", "app2.py"]
