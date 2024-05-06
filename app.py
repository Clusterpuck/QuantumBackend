# This is the update version using FastAPI and gunicorn, caused backend to break
# Need to change requirements.txt before attempting to load this version in the docker. 
import random
import time
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import pyodbc
import threading

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

# Variable to hold the database connection
db_connection = None

def connect_to_database():
    global db_connection
    try:
        db_connection = pyodbc.connect(connection_string)
    except pyodbc.Error as e:
        print("Error connecting to database:", e)
        db_connection = None

def check_and_reconnect():
    global db_connection
    if db_connection is None:
        connect_to_database()

    # Schedule next check after 5 minutes
    threading.Timer(300, check_and_reconnect).start()

# Start the connection checking loop
check_and_reconnect()

def get_total_facts():
    global cached_count, last_count_time
    current_time = time.time()
    if cached_count is None or current_time - last_count_time > cache_expiry:
        if db_connection is None:
            connect_to_database()

        if db_connection:
            # Fetch total count from database
            try:
                cursor = db_connection.cursor()
                cursor.execute('SELECT COUNT(*) FROM QuantumFacts')
                cached_count = cursor.fetchone()[0]
                cursor.close()
                last_count_time = current_time
            except pyodbc.Error as e:
                print("Error executing SQL query:", e)
                cached_count = None
    return cached_count

@app.get("/")
def default_test():
    return "Testing pipeline for github to Azure 7/05/2024 12:25am"

@app.get("/randomfact")
def random_fact():
    total_facts = get_total_facts()
    if total_facts is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve facts from database")

    random_fact_id = random.randint(1, total_facts)

    if db_connection is None:
        connect_to_database()

    if db_connection:
        try:
            cursor = db_connection.cursor()
            cursor.execute('SELECT FactText FROM QuantumFacts WHERE FactID = ?', (random_fact_id,))
            random_fact = cursor.fetchone()[0]
            cursor.close()
            return {'fact': random_fact}
        except pyodbc.Error as e:
            print("Error executing SQL query:", e)
            raise HTTPException(status_code=500, detail="Failed to retrieve random fact from database")
    else:
        raise HTTPException(status_code=500, detail="Failed to establish connection to database")

# if __name__ == '__main__':
#   import uvicorn
#   uvicorn.run(app, host="0.0.0.0", port=8000)
