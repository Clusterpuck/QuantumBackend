# This is the update version using FastAPI and gunicorn, caused backend to break
# Need to change requirements.txt before attempting to load this version in the docker. 

import random
import time
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pyodbc
import uvicorn

# Load environment variables from .env file
load_dotenv()

# Access environment variables
server = os.getenv('QUANTUM_DB_SERVER')
database = os.getenv('QUANTUM_DB_NAME')
username = os.getenv('QUANTUM_DB_USERNAME')
password = os.getenv('QUANTUM_DB_PASSWORD')

app = FastAPI()

# Database details
driver = '{ODBC Driver 17 for SQL Server}'
connection_string = f'Driver={driver};Server={server};Database={database};Uid={username};Pwd={password}'

# Variables for caching
cached_count = None
last_count_time = 0
cache_expiry = 3600  # Cache expiry time in seconds (1 hour)

def get_total_facts(connection):
    global cached_count, last_count_time
    current_time = time.time()
    if cached_count is None or current_time - last_count_time > cache_expiry:
        # Fetch total count from database
        cursor = connection.cursor()
        cursor.execute('SELECT COUNT(*) FROM QuantumFacts')
        cached_count = cursor.fetchone()[0]
        cursor.close()
        last_count_time = current_time
    return cached_count

@app.get('/randomfact')
async def random_fact():
    connection = pyodbc.connect(connection_string)
    total_facts = get_total_facts(connection)
    # Generate a random fact id
    random_fact_id = random.randint(1, total_facts)

    # Retrieve the random fact from the database
    cursor = connection.cursor()
    cursor.execute('SELECT FactText FROM QuantumFacts WHERE FactID = ?', (random_fact_id,))
    random_fact = cursor.fetchone()[0]
    cursor.close()
    connection.close()

    # Return the random fact as JSON
    return JSONResponse(content={'fact': random_fact})

if __name__ == '__main__':
   
    uvicorn.run('app:app', port=8000) #Windows sample uses app:app and port 8000 host is the same
    # app.run(host='0.0.0.0', port=80, debug=True)

