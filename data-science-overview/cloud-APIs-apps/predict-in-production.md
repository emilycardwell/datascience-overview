# Predict in Production

### Web 101

client (user) → http request → server → index.html → client

### API

user interface → http request → server (predict) → response.json → client (user)

# Fast API - local server API

```python
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def index():
    return {'ok': True}
```

- endpoint: function
    - root: ‘/’
    -

```python
from fastapi import FastAPI

api = FastAPI()

@api.get('/')
def return_platypus():
		return {'platypus': 'Bob'}
```

- make a server to host the API: uvicorn

```
uvicorn app.simple:api --reload
>> running on localhost:port
```

- swagger documentation: [localhost:8000/docs](http://localhost:8000/docs) (requests, params, etc)
- redoc documentation: [localhost:8000/redoc](http://localhost:8000/redoc) (same but not as nice)

## Ask for prediction

```python
from fastapi import FastAPI

api = FastAPI()

@api.get('/')
def return_platypus():
		return {'platypus': 'Bob'}

@api.get('/predict_age')
def compute_age(length):
		return {'age': **int**(length)/20}
```

- in api_call.py:

```python
import requests

base_api_url = 'http://localhost:8000/'

endpoint = 'predict'

url = base_api_url + endpoint

params = {'length': 30}

response = requests.get(url, params=params).json()

print(response)
```

```python
python api_call.py
```

`{”age” : 1.5}`

# Non-local Production Platform (hosts)

- containers
    - web server to respond to API calls
    - reliable and scalable solution
    - flexible but don’t have to configure it yourself

### Virtualization

- hardware → os (macos) → virtual environment → application

## Docker

### Dockerfile: blueprint to create docker image

- from, copy run, cmd
- FROM:
    - select base layer: pre-built image of os with languages

    ```docker
    FROM python:3.8.6-buster
    ```

    ```
    docker build -t python-test .
    docker images
    >> list of images

    docker run python-test
    docker ps -a
    >> exits because it didn't have anything to do

    docker run -it python-test sh
    >> (now you're inside it)
    ```

- COPY
    - fills the image with content

    ```docker
    COPY app /app (file in directory)
    COPY requirements.txt requirements.txt
    ```

- RUN

    ```docker
    RUN pip install --upgrade pip
    RUN pip install -r requirements.txt
    ```

- CMD

    ```docker
    CMD uvicorn app.simple:api --host 0.0.0.0 ()
    ```


```
docker build -t platypus-api .
docker images
...
docker run --rm -p 8080:8000 platypus-api
(rm removes container when it stops)
(p gives it a port)
>> info:...
```

```
docker run -e PORT=8000 -p 8000:8000 --env-file .env $IMAGE:dev
```

- go to: http://localhost:8080
- end docker

    ```docker
    docker ps
    >> list of running containers

    docker stop 8 (first few digits of container ID)

    docker kill <ID>
    (hard stop, use with caution)
    ```


### Docker image: mold from which the containers are created

- api code, uvicorn, packages, python, …, system

### container: running instance

- uvicorn api:app, info:…

### consumer: request to the API

```python
import requests

...

print(response)
```

# Container Registry

- google container registry (for cloud run or kubernetes engine)
- console.cloud.google.com

```
docker push $GCR_MULTI_REGION/$PROJECT/$IMAGE
```

## Parameters

### GCP project identifier

```
gcloud projects list
```

```docker
export GCP_PROJECT_ID="replace-me-with-your-project-id"
```

### name of image

```docker
export DOCKER_IMAGE_NAME="name-of-my-image-in-kebab-case"
```

### GCP multi-region/region

```docker
export GCR_MULTI_REGION="eu.gcr.io"
export GCR_REGION="europe-west1"
(replace with the appropriate regions)
```

## Update Dockerfile for GCR

```docker
FROM python:3.8.6-buster

COPY api /api
COPY project /project
COPY model.joblib /model.joblib
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.simple:app --host 0.0.0.0 --port $PORT
```

## Build Image for Container Registry

```
docker build -t $GCR_MULTI_REGION/$GCP_PROJECT_ID/$DOCKER_IMAGE_NAME .
docker run -e PORT=8000 -p 8080:8000 $GCR_MULTI_REGION/$GCP_PROJECT_ID/$DOCKER_IMAGE_NAME
(e lets you use env variables)
```

- verify on [http://localhost:8080/](http://localhost:8080/)

## Push to CR

```
docker push $GCR_MULTI_REGION/$GCP_PROJECT_ID/$DOCKER_IMAGE_NAME
```

# Cloud Run

- send image to cloud run

```
gcloud run deploy --image $GCR_MULTI_REGION/$GCP_PROJECT_ID/$DOCKER_IMAGE_NAME --platform managed --region $GCR_REGION
> platypus-api
> y
>> base url
```

- gcp console → cloud run (see all the requests) → service url

re

```
docker rmi -f <image_id>
```

```
gcloud run deploy --image $GCR_MULTI_REGION/$PROJECT/$IMAGE:prod --memory $MEMORY --env-vars-file .env.yaml
```
