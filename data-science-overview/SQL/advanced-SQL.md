# Advanced SQL

- %-% contains
- %- ends with
- -% begins with

### Parameter Substitution

```python
def funct(c, variable):
	query = 'SELECT.... ?'
	c.execute(query, (variable,))
```

# CRUD

### Create, Replace, Update, Delete

- GROUP BY and ORDER BY can use 1,2,3, etc (the values selected so you don’t have to type the actual path again)

- try out metabase

# Recap

- COUNT: the number of iterations
- SUM: total of values in iterations

### How many people liked at least one post?

```python
query = """
SELECT COUNT(DISTINCT user_id) AS likers
FROM likes
"""

pd.read_sql_query(query, conn)
```

### **Compute the cumulative number of likes per day**

```python
query = """
WITH daily_likes AS(
    SELECT created_at,
        COUNT(*) AS num_of_likes
    FROM likes
    GROUP BY created_at
)
SELECT created_at,
    SUM(num_of_likes) OVER(
        ORDER BY created_at) AS cululative_daily_likes
FROM daily_likes
"""

pd.read_sql_query(query, conn)
```
