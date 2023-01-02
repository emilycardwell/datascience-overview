# Virtual environments quickstart

```
pyenv virtualenv <python version> <nameofenv>
```

```
pyenv virtualenv 3.10.6 lekuto_37    # create a new virtualenv for our project
pyenv virtualenvs          # list all virtualenvs
pyenv activate lekuto_37 . # enable our new virtualenv
pip install --upgrade pip  # install and upgrade pip
pip install -e .           # get requirements
pip list                   # list all installed packages
```