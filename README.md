# Quantum Computing Python Backend for Vehicle Routing  

Uses FastAPI to provide a service to generate delivery routes for a given vehicle routing problem using D-Wave's quantum annealers.

This repository aims to use D-Wave's quantum cloud services to develop a future-oriented solution for quantum routing optimisation. We are assuming that advancements in technology will enhance quantum computing hardware and lead to the discovery of new quantum algorithms. This solution aims to provide high modularity and scalability for future development.

More details on route optimisation can be viewed [here](src/route_optimisation/README.md)

More details on the quantum solution can be viewed [here]()

Endpoints aimed to be used by https://github.com/Clusterpuck/QuantumDelivery

The repository contains a [parameter sweeper](src/parameter_sweeper.py) for sweeping a range of D-Wave parameters for fine-tuning. The output file can be processed by https://github.com/Scrubzie/QuantumPostProcessing

## How To Run

[app.py](src/app.py) provides the FastAPI framework

To test [app.py](src/app.py) locally run with fastapi command while in the src directory: `uvicorn app:app`. API documents can be accessed from /docs end point

Test Docker locally by first building container (Right click Dockerfile and select buildimage) 
Run the built container from the DockerGUI program. If made in WSL run with terminal command `docker run -p 8000:8000 <image name>:<image tag>`

To view current deployment location and documentations see: https://quantumdeliverybackend.azurewebsites.net/docs
