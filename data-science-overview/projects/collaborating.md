# Collaborating on Projects

# GitHub Repositories

- private vs public
- collaborators
- converting a user into an organization
- don’t upload data (each person downloads it), and add to gitignore
    - `cat .gitignore`

### Cloud hosting

- GitLab, BitBucket

### Create Repo

```
git config --global init.defaultBranch trunk
```

```
cd ~/code/YOUR_GITHUB_USERNAME
mkdir YOUR_PROJECT_NAME
cd YOUR_PROJECT_NAME
git init --initial-branch=trunk
echo '# YOUR_PROJECT_NAME' >> README.md
git add README.md
git commit -m 'kickstart YOUR_PROJECT_NAME'
```

```
gh repo create
> push local repository
> (.)
> name
> description
> visibility
> remote? (No)
```

```
git remote -v
git remote add origin <SSH url from GitHub>
git push origin main
```

## Git 101

```
cd lekuto

# Let's create empty files and track them with git
touch feature.py wip.py
git add feature.py wip.py
git commit -m "start tracking 2 files"

# Pretend you've worked on both, and one new has been created
echo "# WIP" >> wip.py
echo "# FINISHED" >> feature.py
touch new.py

# Say you are ready to commit your new feature!
git add feature.py

# Now check your status
git status
```

### Staging

- move to staging

```
git add wip.py
git status
git diff --staged # Check your diff for staged files
```

- remove from staging (but keep mods)

```
git restore --staged wip.py
git status
git diff # Check your diff for modified files
```

- discard changes

```
git restore wip.py # ❗️ Loose modifs since last commit
git status
```

### Commits

```
git add --all
git commit -m "mistake commit"
git lg # check commits history & ID
git revert <your_last_commit_id> # create a new commit undoing the selected commit
```

## Add Collaborators

- add via GitHub UI
- accept via email
- collabs clone:

    ```
    mkdir ~/code/OWNER_GITHUB_USERNAME
    cd ~/code/OWNER_GITHUB_USERNAME
    git clone git@github.com:OWNER_GITHUB_USERNAME/PROJECT_NAME.git
    ```


## Collaborating

- never commit directly to master/main (use branches)
- always make sure git status is clean before:
    - pull, checkout merge

```
git checkout -b addcleaning
```

### get latest master

```
git checkout master
git pull origin master
```

### Merging master in branches

```
# 1/ Commit your branch
(my-task) git add .
(my-task) git commit -m 'a meaningful message'
(my-task) git status # MAKE SURE STATUS IS CLEAN

# 2/ Check out master and pull the latest version
(my-task) git checkout master
(master)  git pull origin master

# 3/ Check out your branch again and merge
(master)  git checkout my-feature
(my-task) git merge master
```

### Resolve Conflicts

```
git status # ⚠️ ⚠️ ⚠️ Make sure it's clean before proceeding
git checkout master
git pull origin master          # pull the latest changes
git checkout unmergeable-branch # switch back to your branch
git merge master                # merge the new changes from master into your branch

# 😱 Conflicts will appear. It's normal!
# 👌 Open VS Code and solve conflicts (locate them with cmd + shift + f `<<<<<`)
# When solved, we need to finish the merge

git add .                           # add the files in conflict
git commit --no-edit                # commit using the default commit message
git push origin unmergeable-branch  # push our branch again
```

- or resolve on github

### Git full?

```
# move the large file to raw_data folder which is ignored by Git:
mv path/to/your/large/file raw_data/
# check that the large file is marked as deleted:
git status
git add path/to/your/large/file
git commit -m 'move large file to raw_data'
# check that the working directory is clean:
git status
# remove the large file from all commits:
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/your/large/file" \
  --prune-empty --tag-name-filter cat -- --all
git push origin YOUR_BRANCH
```

### API

```python
# ./lekuto/utils.py
import geocoder

def geocode(address):
    mapbox_api_key = 'pk.eyJ1Ijoia3Jva3JvYiIsImEiOiJjam83MjVrbWkwbWNoM3FwN2VhMm81eGRzIn0.yM3wkq5LJd8NeSYyPyTY4w'
    g = geocoder.mapbox(address, key=mapbox_api_key)
    return (g.json['lat'], g.json['lng'])
```

### Credentials

```
# ./.env
MAPBOX_API_KEY='pk.eyJ1Ijoia3Jva3JvYiIsImEiOiJjam83MjVrbWkwbWNoM3FwN2VhMm81eGRzIn0.yM3wkq5LJd8NeSYyPyTY4w'
```

- setup

```
echo 'python-dotenv' >> requirements.txt
pip install -r requirements.txt
touch .env
echo '.env' >> .gitignore
```

- fetch

```python
# ./lekuto/utils.py
import geocoder
import os
from dotenv import load_dotenv, find_dotenv

# point to .env file
env_path = join(dirname(dirname(__file__)),'.env') # ../.env
env_path = find_dotenv() # automatic find

# load your api key as environment variables
load_dotenv(env_path)

def geocode(address):
    mapbox_api_key = os.getenv('MAPBOX_API_KEY')
    g = geocoder.mapbox(address, key=mapbox_api_key)
    return (g.json['lat'], g.json['lng'])
```

or

```python
# ./lekuto/__init.py__
# ...
from os.path import join
from dotenv import load_dotenv
# ...
env_path = join(dirname(dirname(__file__)),'.env') # ../.env
load_dotenv(dotenv_path=env_path)
```

## Notebooks

- create your own
    - <lastname>_<projectname>.ipynb

### nbdime

- visualize rich differences between two notebooks

```
nbdime diff-web base_notebook.ipynb updated_notebook.ipynb
nbdime extensions --enable
```

## New Virtualenv

```
pyenv virtualenv lekuto_37 # create a new virtualenv for our project
pyenv virtualenvs          # list all virtualenvs
pyenv activate lekuto_37   # enable our new virtualenv
pip install --upgrade pip  # install and upgrade pip
pip list                   # list all installed packages
```

# Git stuff

```
git branch
git branch -d <name>
```
