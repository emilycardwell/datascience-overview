# User Interface

![Screen Shot 2022-11-25 at 9.01.44 AM.png](ui/Screen_Shot_2022-11-25_at_9.01.44_AM.png)

# http: hypertext transfer protocol

- +s : secure

### Request

- url: protocol scheme, domain host, port, edpoint path, params query-string
- verb: get, post, put, patch, delete
- headers
- body

### Response

- code:
    - 2xx: everything is okay
    - 3xx: redirect
    - 4xx: user mistake
    - 5xx: developer mistake
- data: json, html, css, js, images, etc.
- headers

# Frameworks

- jupyter
- streamlit - low complexity
- flask - med
- django - high

## Streamlit

```python
import streamlit as st

import numpy as np
import pandas as pd

st.markdown("""# This is a header
## This is a sub header
This is text""")

df = pd.DataFrame({
    'first column': list(range(1, 11)),
    'second column': np.arange(10, 101, 10)
})

# this slider allows the user to select a number of lines
# to display in the dataframe
# the selected value is returned by st.slider
line_count = st.slider('Select a line count', 1, 10, 3)

# and used to select the displayed lines
head_df = df.head(line_count)

head_df
```

```
streamlit run app.py
```

- @st.cache - only loads element once and caches it (faster interactivity)

# Deploy on the Cloud

### Project Repository

- don’t create repo in repo

```
# Go to the location of the data-challenges repo
cd ~/code/USER_NAME/

# Create a separate directory for your project
mkdir appname

# Sit inside of your app
cd appname

# Initialize a new git repo
git init

# Create GitHub repo
gh repo create

# Go to GitHub repo and change the visibility to private (in the settings of the repo)
gh browse
```

---

# Challenges

- create new repository in browser
- CLI:

    ```
    cd ~/code/emilycardwell
    git clone git@github.com:emilycardwell/taxi-fare-interface.git
    cd ~/code/emilycardwell/taxi-fare-interface
    ```

    ```
    python -m http.server
    ```

- browser: [http://localhost:8000/](http://localhost:8000/)

---

```
cd ~/code/emilycardwell
mkdir taxifare-website
cd taxifare-website
```

```
git init
gh repo create taxifare-website --private --source=. --remote=origin
gh browse
```

---

```
touch app.py
cp ~/code/emilycardwell/data-streamlit-api/Makefile ~/code/emilycardwell/taxifare-website/
tree
```
