# SQL

- add online dataset to user:
    - curl [https://wagon-public-datasets.s3.amazonaws.com/sql_databases/movies.sqlite](https://wagon-public-datasets.s3.amazonaws.com/sql_databases/movies.sqlite) \

    > ~/code/emilycardwell/data-sql-queries/data/movies.sqlite
    >
- DBeaver
- kitt db schema <3 <3 <3

```sql
SELECT match.id, season, stage, date FROM "Match"
												# date: 04-10-2020 00:00:00:00
SELECT match.id, season, stage, date(date) FROM "Match"
												# date: 04-10-2020

--comment
```

### Examples

```sql
SELECT *
	FROM Match
	AS mtc
	WHERE country_id = 4769
#only from France

WHERE mtc.country_id IN (123,1243)

WHERE player_name LIKE '%JONES%'

SELECT COUNT(*)
	WHERE player_height > 200

SELECT *
	FROM player
	ORDER BY weight DESC
	LIMIT 10
```

```sql
SELECT id, home_team_goal, away_team_goal
,CASE WHEN home_team_goal > away_team_goal THEN 'Win'
			WHEN home_team_goal = away_team_goal THEN 'Draw'
			ELSE 'Loss'
			END AS result --new column name
FROM Match
```

```sql
SELECT *
FROM Match AS matches
	JOIN League ON matches.league_id = League.id
	JOIN Country ON League.country_id = Country.id
GROUP BY League.id
```

### Python

```python
import sqlite3

conn = sqlite3.connect('/data/database.sqlite')
c = conn.cursor()
# point at a db in a certain area
result = c.execute('SELECT * FROM Match')
matches = result.fetchall()
print(matches)

single = result.fetchone()
print(single)

print(matches[0])
```

```python
import sqlite3

conn = sqlite3.connect('/data/database.sqlite')
c = conn.cursor()
conn.row_factory = sqlite3.Row
# creates ability to extract columns and access them as strings in key

result = c.execute('SELECT * FROM Match')
match1 = result.fetchone()

print(match1['name'])
```

- if WHERE points where there isn’t a value, it will give you back None
- Bigquery - can access online databases
    - host, port, username, pwd
    - downloads json file

# Recap

### Jupyter Notebooks

- ! curl
    - calls terminal on your computer
