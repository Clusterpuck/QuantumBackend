# Quantum Computing Backend for Vehicle Routing  

Provides basic framework for exposing backend end points for providing quantum determinations of routing data.

src/app.py provides the FastAPI framework

To test app.py locally run with fastapi command while in the src directory: `uvicorn app:app`. API documents can be accessed from /docs end point

Test Docker locally by first building container (Right click Dockerfile and select buildimage) 
Run the built container from the DockerGUI program. If made in WSL run with terminal command `docker run -p 8000:8000 <image name>:<image tag>`

To view current deployment location and documentations see: https://quantumdeliverybackend.azurewebsites.net/docs
