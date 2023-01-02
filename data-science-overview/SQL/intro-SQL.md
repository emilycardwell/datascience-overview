# Intro to SQL

- language for databases
    - relational databases: tables of columns and rows with unique primary keys for each row
    - foreign key: what we link the outer table with
    - databases: SQLite, MySQL, oracle, IBM DB2, PostgreSQL
- NoSQL: document (couchdb), key-value (couchbase, dynamo), graph (neo4J)
    - single database instead of joining tables
- SQLite: self-contained database engine
- **No. 1 Rule:** be able to read and understand datasets

### Data Types

```sql
INTEGER PRIMARY KEY  # index
INTEGER
TEXT  # variable string
CHAR  # fixed string
REAL  # float
BLOB  # binary large object
NULL
--comment
```

### Statements

```sql
CREATE TABLE
INSERT INTO
SELECT  # extract
UPDATE
DELETE
CREATE DATABASE
ALTER DATABASE
ALTER TABLE
DROP TABLE  # delete
CREATE INDEX  # search key
DROP INDEX  # delete
```

### Keywords

```sql
VALUES
FROM
*  # all
OVER
PARTITION BY
WHERE or HAVING
GROUP BY  # makes equation apply to specific columns
ORDER BY
AS
CASE
DESC  # descending
LIMIT

WITH --temp table

WHEN
ELSE

ROUND()
SUM()
MAX()
AVG()

AUTOINCREMENT  # usefull for primary key

UPPER()
LOWER()

AND
OR
<=, >=, <, >, =
IN
NOT IN
LIKE %STRING%

CONCAT
--or
||
```

### Order within query

```sql
SELECT * ,CASE WHEN ... THEN ... ELSE ... END
	OVER (PARTITION BY ... ORDER BY ... )
	FROM Match AS matches
	-- JOIN...
	-- WHERE...
	GROUP BY matches.country_id
	WHERE match_count < 3000
	ORDER BY match_count DESC
	-- LIMIT...
```

### Order within project

```sql
CREATE TABLE example(id INT PRIMARY KEY AUTOINCREMENT, name TEXT, number INT;

INSERT INTO example(name, number)
		VALUES ("Jones", 1);

SELECT * FROM example WHERE type IN("a", "b", "c");
																 IN(SELECT FROM table_1);

SELECT * FROM example WHERE name LIKE("%TRUE%");
```

---

### Accessing Existing DB

```sql
SELECT type, number, CASE
		WHEN number > 100 THEN "above max"
		WHEN number < 50 THEN "below min"
		ELSE "in range"
		FROM example;

SELECT COUNT(*), CASE
		WHEN number > 100 THEN "above max"
		WHEN number < 50 THEN "below min"
		ELSE "in range"
		FROM example
		GROUP BY number_range;
```

### Creating (inserting data)

```sql
INSERT INTO table(column1, column2, ...)
VALUES(value1, value2, ...)
```

### Altering Tables

```sql
UPDATE table1 SET name = "B"
		WHERE id = 1;
# or
		WHERE user = "A" AND date = "2022-07-04"

DELETE FROM table1 WHERE id = 1;

ALTER TABLE table1 ADD column3 INT;
# doesn't break existing data in table

DROP TABLE table1;
# deletes table

BEGIN TRANSACTION;
...
COMMIT;
# makes 2+ create, update, insert, or delete commands
#  dependent on eachother's issuability
```

### Updating Tables

```sql
-- replace 1 row with 2 column values (the where must already exist)
UPDATE table
SET column_1 = new_value1
		column_2 = new_value2
WHERE
		id = 33
```

# Joins

### Inner Join

- only data that exists in both tables

```sql
-- two tables linked by key(s)
-- only get ids and rows in columns that exist only in both tables
SELECT * FROM table1
		INNER JOIN table2
		ON table1.name = table2.name;
```

### Left Joins

- all data from left table, with equivalent data (including nulls) from right

```sql
--left outer: everything from left table, with matching values from right
--null value from right table if none
SELECT * FROM table1
	LEFT JOIN table2
	ON table1.id = table2.user_id;
```

### Full Outer Join

- all data from both tables, filling nonequivalents with nulls

```sql
-- includes all values from both,
-- Null values if one doesn't exist in the other
SELECT *
FROM TableA
FULL OUTER JOIN TableB
  ON TableA.name = TableB.name;
```

### Excluding Joins

- data from left/either table ONLY when not shared

```sql
--left
SELECT *
FROM tableA
LEFT JOIN tableB
  ON tableA.name = tableB.name
WHERE tableB.name IS NULL

--outer
SELECT *
FROM tableA
FULL OUTER JOIN tableB
  ON tableA.name = tableB.name
WHERE tableA.name IS NULL
  OR tableB.name IS NULL
```

### Self Join

- using data from inside single table to replace a column (id val to name val for ex.)

```sql
-- can change id inside a table to a text using another id
-- i.e. manager_id to manager_name
SELECT table1.*, table2.name AS table2
		FROM table1
		LEFT JOIN table1 as table2
		ON table1.id = table2.id
```

## Examples

```sql
# cross join
SELECT * FROM table1, table2;

# implicit inner join
SELECT * FROM table1, table2
		WHERE table2.user_id = table1.id;

# explicit inner join
SELECT * FROM table1
		JOIN table2
		ON table1.id = table2.user_id;

# outer join - presents values even if null in left table
LEFT OUTER JOIN

# self join - makes new column based on information already in table
SELECT table1, **newtablealias**.**name**
		FROM table1
		JOIN table1 **newtablealias.alias**
		ON table1.user_id = **alias**.**id**
```

```sql
SELECT table1.name, table1.number, SUM(table2.price)
		FROM table1
		LEFT OUTER JOIN table2
		ON table1.id = table2.user_id
		GROUP BY table1.id  # new row of summed items by index
		ORDER BY table2.price DESC;
```

# Functions

```sql
-- strftime(format, timestring, modifier, modifier, ...)

SELECT
	STRFTIME('%Y-%m-%d', DATE(table1.column1)) AS period,
	COUNT(*) AS cnt
```

### Windowed Aggregate Functions

- Unlike regular aggregate functions, a window function does not cause rows to become grouped into a single output row — the rows retain their separate identities.
- A window function performs a calculation across a set of table rows that are somehow related to the current row. This is comparable to the type of calculation that can be done with an aggregate function.

```sql
-- happens right after SELECT
-- partition defines the groups into which the rows are divided
OVER ([PARTITION BY columns] [ORDER BY columns])
```

```sql
SUM() OVER (PARTITION BY ... ORDER BY ...)
RANK() OVER (PARTITION BY ... ORDER BY ...)
```

### With

- scaling & building on your previous tables
- virtual table

```sql
WITH temporary_table AS
(SELECT *
FROM table1
		 ....
)
SELECT * FROM temporary_table;
```

### COALESCE

- if value exists, return it, otherwise use a default value

```sql
SELECT COALESCE(AVG(column), "1")
FROM table1
```

### Optional Conditions

- only works when user inputs something

```sql
WHERE 1=1
		[[optional_clause1 {%user_input%}]]
		[[optional_clause2 {%user_input%}]]
```
