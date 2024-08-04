# Quantum computing backend for routing  

Provides basic framework for exposing backend end points for providing quantum determinations of routing data.  

Both FastAPI and Flask frameworks have been provided. app.py provides FastAPI, app2.py provides for Flask. 

To test app2.py locally, run with python3. `python3 app2.py`. API documents can be accessed from /swagger end point

To test app.py locally run with fastapi command: `uvicorn app:app`. API documents can be accessed from /docs end point
Test Docker locally by first building container (Right click Dockerfile and select buildimage) 
Run the built container from the DockerGUI program. If made in WSL run with terminal command `docker run -p 8000:8000 <image name>:<image tag> 

To swap between FastAPI and Flask comment out and uncomment associated code in the DockerFile and the requirements.txt file

To view current deployment location and documentations see: https://quantumdeliverybackend.azurewebsites.net/swagger/


