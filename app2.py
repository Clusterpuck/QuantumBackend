import random
from flask import Flask, jsonify, request
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)

facts = ["One", "Two", "Three", "Four", "Five"]

def get_total_facts():
        return len(facts)

@app.route('/')
def default_test():
    #return "Updated to define port explicitly in server string 11/05/24 7:44PM"
    return "Removed database connection"


#Get endpoint sample
@app.route('/randomfact')
def random_fact():
    total_facts = get_total_facts()
    if total_facts is None:
        #Sample exception handling
        return jsonify({'error': 'Failed to retrieve facts from database'}), 500

    random_fact_id = random.randint(1, total_facts-1)
    return jsonify({'fact': facts[random_fact_id]})


#Post end point sample
@app.route('/addfact', methods=['POST'])
def add_fact():
    new_fact = request.json.get('fact')
    if not new_fact:
        return jsonify({'error': 'No fact provided'}), 400

    facts.append(new_fact)
    return jsonify({'message': 'Fact added successfully', 'total_facts': get_total_facts()}), 201

# Swagger UI setup
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL, config={'app_name': "Flask Facts API"})
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

#Swagger must be manually added here like this. 
@app.route('/static/swagger.json')
def swagger_json():
    return jsonify({
        "swagger": "2.0",
        "info": {
            "title": "Flask Facts API",
            "description": "API for managing facts",
            "version": "1.0.0"
        },
        "basePath": "/",
        "schemes": ["http"],
        "paths": {
            "/": {
                "get": {
                    "summary": "Default Test",
                    "description": "Returns a test message",
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "schema": {
                                "type": "string"
                            }
                        }
                    }
                }
            },
            "/randomfact": {
                "get": {
                    "summary": "Get Random Fact",
                    "description": "Returns a random fact from the list",
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "fact": {
                                        "type": "string"
                                    }
                                }
                            }
                        },
                        "500": {
                            "description": "No facts available",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "error": {
                                        "type": "string"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/addfact": {
                "post": {
                    "summary": "Add a New Fact",
                    "description": "Adds a new fact to the list",
                    "parameters": [
                        {
                            "name": "fact",
                            "in": "body",
                            "description": "The fact to add",
                            "required": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "fact": {
                                        "type": "string"
                                    }
                                }
                            }
                        }
                    ],
                    "responses": {
                        "201": {
                            "description": "Fact added successfully",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "message": {
                                        "type": "string"
                                    },
                                    "total_facts": {
                                        "type": "integer"
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "No fact provided",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "error": {
                                        "type": "string"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000) #Port number is defined for container
