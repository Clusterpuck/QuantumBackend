# This version of app was reverted back to not use FastAPI or gunicorn to test original plan. 

import random
import time
import os
from dotenv import load_dotenv
from flask import Flask, jsonify
import pyodbc

# Load environment variables from .env file
load_dotenv()

# Access environment variables
server = os.getenv('QUANTUM_DB_SERVER')
database = os.getenv('QUANTUM_DB_NAME')
username = os.getenv('QUANTUM_DB_USERNAME')
password = os.getenv('QUANTUM_DB_PASSWORD')

app = Flask(__name__)

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

@app.route('/')
def default_test():
    return "Updated live from 3/05/2024, 11:07am"


@app.route('/randomfact')
def random_fact():
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
    return jsonify({'fact': random_fact})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000) # Replace with desired port number
