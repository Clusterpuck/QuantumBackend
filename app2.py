import random
from flask import Flask, jsonify

app = Flask(__name__)

facts = ["One", "Two", "Three", "Four", "Five"]

def get_total_facts():
        return len(facts)

@app.route('/')
def default_test():
    #return "Updated to define port explicitly in server string 11/05/24 7:44PM"
    return "Removed database connection"


@app.route('/randomfact')
def random_fact():
    total_facts = get_total_facts()
    if total_facts is None:
        #Sample exception handling
        return jsonify({'error': 'Failed to retrieve facts from database'}), 500

    random_fact_id = random.randint(1, total_facts-1)
    return jsonify({'fact': facts[random_fact_id]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000) #Port number is defined for container
