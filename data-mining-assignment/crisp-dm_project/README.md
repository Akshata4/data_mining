# Deployment of Machine Learning Model with FastAPI and Docker

This document outlines the steps to deploy the machine learning model using FastAPI and Docker.

## Demo Video

[Watch Demo Video here (use SJSU account)](https://sjsu0-my.sharepoint.com/:v:/g/personal/akshata_madavi_sjsu_edu/Eez4hJErRAtFrQhmA0tp16kB3cXFMNp6wRRJOATRKEeDIg?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=MKdpqs)

## Step-by-Step Guide

### Prerequisites

- Docker
- Python 3.8+
- Ensure you have the model and transformers saved as `model_v1.joblib` and `transformer.joblib` respectively in the `deployment/` directory.

### Installation and Setup

1. **Clone the Repository**
   
   Clone the project repository and navigate to the project root.
   ```bash
   git clone <repository-url>
   cd <repository-root>
   ```

2. **Setup Environment**

   Use `uv` or any virtual environment tool to create and activate a virtual environment:
   ```bash
   uv --create -r deployment/requirements.txt
   source your_env/bin/activate  # Command might vary depending on the tool used
   ```

3. **Build Docker Image**

   Build the Docker image for the FastAPI service:
   ```bash
   docker build -t model-api ./deployment
   ```

4. **Run Docker Container**

   Run the Docker container:
   ```bash
   docker run -p 8000:8000 model-api
   ```

### Usage

- **Health Check**

  Access the health check endpoint to verify the service is running:
  ```bash
  curl http://localhost:8000/health
  ```

- **Make a Prediction**

  Send a POST request to the predict endpoint:
  ```bash
  curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"feature1": 0.5}'
  ```

### Testing

- **Run Tests**

  Run the test suite to ensure all API endpoints are functioning as expected:
  ```bash
  pytest tests/
  ```

### Monitoring and Rollback

- Health and monitoring endpoints are available to verify service status.
- Rollback by deploying the previous tagged Docker images and model versions if required.

### Risks and Next Steps

- Implement further logging and monitoring tools for real-time insights.
- Plan periodic audits and updates to the model and code to ensure accuracy and compliance.

