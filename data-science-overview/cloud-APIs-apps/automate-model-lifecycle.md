# Automate model lifecycle

### Previously…

- Google Cloud Platform
    - console vs cli vs code
    - authentication
- cloud storage
    - immutable data
    - files, imgs, sound, video
- big query
    - relational data
    - columnar storage & partitions
- compute engine
    - virtual machine
    - setup operating sys and code env
- direnv
    - environmental variables
    - .env
        - `BUCKET_NAME=le-wagon-data`
    - cli or makefile
        - `gsutil ls gs://$BUCKET_NAME`
    - code
        - `bucket_name = os.environ["BUCKET_NAME"]`

---

1. train the model in the data warehouse
2. save the trained model in a storage solution in the cloud
3. run the model training on a virtual machine

---

## Robust Lifecycle

### reproducibility

# track performance over time: MLflow server

### store parameters (data, code, environment)

1. tracking requirements:
    1. training data
    2. datetime or timestamp vs no. of rows
    3. code version, params
    4. training environment (python and package version)
    5. preprocessing type
    6. model hyperparams
    7. training metrics
    8. persisted training model
    9. version number
2. store model performance
3. store trained model

### How?

1. MLflow server:
    1. store tracking data
    2. store trained models in file storage sys
    3. hosted locally or cloud
2. MLflow UI
    1. web interface to visualize tracking data and trained models
3. MLflow CLI
4. MLflow code
    1. mlflow python package
    2. pushes data and trained models to server through API

---

Track & Save

```python
import mlflow

mlflow.set_tracking_uri("https://mlflow..ai")
mlflow.set_experiment(experiment_name="wagoncab taxifare")
```

```python
with mlflow.start_run():

    params = dict(batch_size=256, row_count=100_000)
    metrics = dict(rmse=0.456)

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    mlflow.keras.log_model(keras_model=model,
                           artifact_path="model",
                           keras_module="tensorflow.keras",
                           registered_model_name="taxifare_model")
```

Load model

```python
import mlflow

mlflow.set_tracking_uri("https://mlflow.organization.ai")

model_uri = "models:/taxifare_model/Production"

model = mlflow.keras.load_model(model_uri=model_uri)
```

### serve multiple versions of the model

# automate model lifecycle - Prefect

1. formalize model workflow
2. have it run periodically

---

### Tasks

```python
from taxifare_model.interface.main import (preprocess, train, evaluate)

from prefect import task

@task
def eval_perf(next_row):
    past_perf = evaluate(next_row)
    return past_perf

@task
def train_model(next_row):
    preprocess(first_row=next_row)
    new_perf = train(first_row=next_row, stage="Production")
    return new_perf

@task
def notify(past_perf, new_perf):
    print(f"Past perf: {past_perf}, new perf: {new_perf}")
```

### Workflow

```python
from prefect import Flow

def build_flow(schedule):

    with Flow(name="wagonwab taxifare workflow", schedule=schedule) as flow:

        next_row = 0
        past_perf = eval_perf(next_row)
        new_perf = train_model(next_row)
        notify(past_perf, new_perf)

    return flow
```

### Main

```python
import datetime

from prefect.schedules import IntervalSchedule
from prefect.executors import LocalDaskExecutor

if __name__ == "__main__":

    # schedule = None                    # no schedule
    schedule = IntervalSchedule(interval=datetime.timedelta(minutes=300))

    flow = build_flow(schedule)

    flow.visualize()

    # flow.run()                         # local run
    flow.executor = LocalDaskExecutor()  # parallel executor
    flow.register("wagoncab project")    # backend run
```

## Command Line

### Self-hosted Prefect Server

```
prefect server start \
    --postgres-port 5433 \
    --ui-port 8088                      # start prefect server
```

### Workflow

```
prefect backend server                  # switch backend to self hosted

prefect agent local start               # start local agent

prefect create project "test project"   # create project
```

### Prefect Cloud

```
prefect backend cloud                   # switch backend to prefect cloud
prefect auth login -k $API_KEY          # login to prefect cloud

prefect agent local start               # start local agent

prefect create project "test project"   # create project
```
