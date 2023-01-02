# Online Data

# Online Data: Data Sourcing

### **SQL** databases

### **CSV**

- (plain text comma-separated values): export from software
    - doesn’t have to be separated by commas (: / etc.)

```python
import csv

with open('directory/filename.csv') as csvfile:
		reader = csv.reader(csvfile)

		for row in reader:
				print(row)
## prints list
```

- **Headings**: with dataset that has first column as a header column:

    ```python
    with open('directory/filename.csv') as csvfile:
    		reader = csv.DictReader(csvfile, skipinitialspace=True)

    		for row in reader:
    				print(row)
    ## prints dictionary using first column as first dict key
    ```

- Writing a CSV:

    ```python
    lofd = [{...}, {...}...]

    with open('directory/filename.csv', 'w') as csvfile:
    		writer = csv.DictWriter(csvfile, fieldnames=lofd[0].keys())
    		writer.writeheader()

    		for x in lofd:
    				writer.writerow(x)
    ```


### **API**

- application programing interface) fetching: service that gives access to data
- contract: communication protocol between client and server
- http: client-based protocol based on request/response cycle
- architecture/protocols:
    - [SOAP](https://en.wikipedia.org/wiki/SOAP) (old)
        - can only use xml
    - [REST](https://en.wikipedia.org/wiki/Representational_state_transfer) (current)
        - xml, json, plain text
    - [GraphQL](https://en.wikipedia.org/wiki/GraphQL) (very new, less frequent)
- data formats:
    - [XML](https://en.wikipedia.org/wiki/XML) (long-established)
    - [JSON](https://en.wikipedia.org/wiki/JSON) (currently very widespread)
- there is always documentation for APIs, but they are sometimes hard to use
    - online: base url > endpoint > parameters
    - RESTful: GET, POST
        - requests: HTTP for humans

        ```python
        import requests

        url = 'https:...'

        response = requests.get(url)
        print(response)
        ##Response [200]
        # perfect! other number? sucks
        ```

        ```python
        import requests

        url = "https://..."
        response = requests.get(url)
        print(response.status_code)
        data = response.json()

        # OR

        response = requests.get(url).json()
        print(type(response)
        ## dict
        ```

        ```python
        import requests

        isbn = '0-7475...'
        url = f'https://...?bibkeys=ISBN:{isbn}&format=json&jscmd=data'
        									# parameters: start with ? if single, & if mult.
        									# key                 format      filter
        response = requests.get(url).json()
        ```

        ```python
        response = requests.get(
        		'base url',
        		params={'bibkeys': key, 'format': 'json', 'jscmd': 'data'}
        		).json()
        print(response[key]['title']
        ```

    - Returns: **JSON** (dictionary)

        ```python
        import json

        json.load(file_name)
        # returns dictionary

        # or

        data = requests.get(url).json()
        jDict = {}
        for myDict in data:
        		for key, value in song.items():
        				jDict[key] = value
        ```


### **Scraping**: online repositories

- cmd + opt + i (see info in a website)
- datasetsearch.research.google.com
- kaggle.com
- HTML: unstructured data
    - <start tag, attribute=value>content<end tag>
    - <p> (paragraph smart tag)
- **BeautifulSoup**
    - package to browse, can also do xml

```python
import requests
from bs4 import BeautifulSoup

response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

soup.title.string
soup.find("h1") # html: heading 1
## <h1>Main title<h1>
soup.find_all("li") # html: list items
```

```python
import requests
from bs4 import BeautifulSoup

with open('url.html') as file:
		soup = BeautifulSoup(file, "html.parser")

		soup.find("h1").text
## Main title
		print(soup.find_all("li"))
## [<li="result>Result 1<li>, <li....]

		body = soup.find("body")
		print(body.find_all("li"))
# searches for & prints all lists in the body

		soup.find(id="...")

		soup.find_all("li", class_="pizza")
**# must use class_ because class is a global object in python**
```

# Online Sourcing w/ Pandas

- I/O API

```python
import pandas as pd

pd.read_sql
pd.read_bgq
```

### Jupyter Lab

- shows tables or dataframes in new windows
- can have a terminal window
- terminal: jupyter lab

```python
import matplotlib as mpl
%mpl inline
import numpy as np
import pandas as pd
```

```python
tracks_df = pd.read_csv('data/spotify_2017.csv')
artist_track = tracks_df[['artists', 'name']]
```

### API

- status check (200 is good, API might give specific error return - i.e. 400)
    - terminal: curl -i

```python
import requests
```

```python
url = 'http://lyrics.organization.ai/...'
response = requests.get(url)
data = response.json()['lyrics'] # key of the dictionary data returns
```

```html
new_df = pd.DataFrame(data)
```

```python
def fetch_lyrics(artist, title):
		url = f'http://lyrics.organization.ai/search?artists={artist}&title={title}'
		response = requests.get(url)
		if response.status_code == 200:
				data = response.json()
				return data['lyrics']
		return "No lyrics"
```

```python
from music import fetch_lyrics
```

```python
%load_ext autoreload
%autoreload 2
```

```python
tracks_df['lyrics'] = "" # new column in the df
# setting them as blank (not None/NaN)
```

```python
for index, row in tracks_df.iterrows():
		lyrics = fetch_lyrics(row["artists"], row['name'])
		tracks_df.loc[index, 'lyrics'] = lyrics
		return tracks_df.loc
```

### SQL

```python
import pandas as pd
import sqlite3
conn = sqlite3.connect("data/soccer.sqlite")
```

```python
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(cursor.fetchall())
```

```python
query = '''
    SELECT l.id, l.name, c.name as country_name
    FROM League l
    JOIN Country c ON c.id = l.country_id
'''
league_df = pd.read_sql(query, conn)
league_df.head(3)
```

### Google Big Query

- sql

```sql
SELECT faa_identifier, name, longitude, latitude, airport_type, service_city, country
FROM `bigquery-public-data.faa.us_airports`
WHERE airport_use = 'Public'
```

```python
import pandas_gbq
```

```python
project_id = 'my-project-name'
sql = """
		SELECT faa_identifier, name, longitude, latitude, airport_type, service_city, country
		FROM `bigquery-public-data.faa.us_airports`
		WHERE airport_use = 'Public'
"""

airports_df = pandas_gbq.read_gbq(sql, project_id=project_id)
airports_df.shape
```

## Scraping

- hard way:

```python
import re #regular expression
import requests
from bs4 import BeautifulSoup
import pandas
```

```python
url = "https://www.imdb.com/list/ls055386972/"
response = requests.get(url, headers={"Accept-Language":"en-US"})
soup = BeautifulSoup(response.content, "html.parser")
```

- from list:

```python
movies = []
for movie in soup.find_all("div", class_="lister-item-content"):
    title = movie.find("h3").find("a").string
#find the title
    duration = int(movie.find(class_="runtime").string.strip(' min'))
#find the duration
    year = int(re.search(r"\d{4}", movie.find(class_="lister-item-year").string).group(0))
#find the year
		movies.append({'title': title, 'duration': duration, 'year': year})
print(movies[0])
```

```python
movies_df = pd.DataFrame(movies)
movies_df.head()
movies.dtypes # columns and data types
```

---

- from dict:

```python
movies_dict = {'title': [], 'duration': [], 'year': []}

for movie in soup.find_all("div", class_="lister-item-content"):
    movies_dict['title'].append(movie.find("h3").find("a").string)
    movies_dict['duration'].append(int(movie.find(class_="runtime").string.strip(' min')))
    movies_dict['year'].append(int(re.search(r"\d{4}", movie.find(class_="lister-item-year").string).group(0)))

print(movies_dict['title'][0:2])
```

```python
movies_dict_df = pd.DataFrame.from_dict(movies_dict)
movies_dict_df.head()
```

---

### Scraping multiple pages

```python
def fetch_page(page):
    response = requests.get(
        "https://www.imdb.com/search/title/",
        params={"groups":"top_250", "sort":"user_rating","start": (1 + page * 50)},
        headers={"Accept-Language":"en-US"})
    soup = BeautifulSoup(response.content, "html.parser")
    return soup
```

```python
def parse_movies(soup):
    movies = []
    for movie in soup.find_all("div", class_="lister-item-content"):
        title = movie.find("h3").find("a").string
        duration = int(movie.find(class_="runtime").string.strip(' min'))
        year = int(re.search(r"\d{4}", movie.find(class_="lister-item-year").string).group(0))
        movies.append({'title': title, 'duration': duration, 'year': year})
    return movies
```

```python
all_movies = []
for page in range(5):
    print(f"Parsing page {page + 1}...")
    soup = fetch_page(page)
    all_movies += parse_movies(soup)
print("Done")
```

```python
all_movies_df = pd.DataFrame(all_movies)
all_movies_df.hist(grid=False, bins=12, figsize=(12, 4))
```

---

### Extra

```html
book_title = books_html[0].find("h3").find("a").attrs["title"]
```

```html
response = requests.get(query, timeout=200)
```
