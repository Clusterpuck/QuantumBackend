import random
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import pyodbc

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

# Variable to hold the database connection
db_connection = None

def connect_to_database():
    global db_connection
    if db_connection is None:
        try:
            db_connection = pyodbc.connect(connection_string)
        except pyodbc.Error as e:
            print("Error connecting to database:", e)
            db_connection = None

def get_total_facts():
    global db_connection
    connect_to_database()  # Check connection before every request
    if db_connection is None:
        return None
    try:
        cursor = db_connection.cursor()
        cursor.execute('SELECT COUNT(*) FROM QuantumFacts')
        total_count = cursor.fetchone()[0]
        cursor.close()
        return total_count
    except pyodbc.Error as e:
        print("Error executing SQL query:", e)
        return None

@app.get("/")
def default_test():
    return "Updated github actions with better yml files to deploy 11/05/24 11:30AM"

@app.get("/randomfact")
def random_fact():
    total_facts = get_total_facts()
    if total_facts is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve facts from database")

    random_fact_id = random.randint(1, total_facts)

    connect_to_database()  # Check connection before every request
    if db_connection is None:
        raise HTTPException(status_code=500, detail="Failed to establish connection to database")

    try:
        cursor = db_connection.cursor()
        cursor.execute('SELECT FactText FROM QuantumFacts WHERE FactID = ?', (random_fact_id,))
        random_fact_text = cursor.fetchone()[0]
        cursor.close()
        return {'fact': random_fact_text}
    except pyodbc.Error as e:
        print("Error executing SQL query:", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve random fact from database")
