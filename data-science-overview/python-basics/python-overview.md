# Python Overview

- runs directly by an interpreter (don’t have to compile)
- garbage collected - automatically deletes datasets etc from memory
- from terminal to vs code:
    - code .
    - (write code in file)
    - python file.py
    - (runs code in terminal)
- REPL - read, eval, print, loop

```python
git:(main) ipython

In [1]: 1 + 1
Out [1]: 2]

In [2]: quit

git:(main)
```

- complex numbers?
- 0 == False:

```python
if 0: # or 0.0
		print("true")
## no output

bool([])
## False
bool([1])
## True
```

- underscores in numbers make it more readable but don’t change the value

```python
print(2_141_000)
## 2141000
```

- in dicts, keys can be any immutable objects: (they can’t be edited after assignment, only their values assigned to a new key and the old row deleted)
    - ints, floats, bools, tuples
- can’t search with “in” for values, only keys

```python
"key" in dict
## True
"value" in dict
## False
```

### String Interpolation

- f string

```python
print(f"{var1} is my {var2}")
## Henry is my Dog
```

### List functions

- CRUD; create, read, update, delete

```python
mylist = ["yes", "no"]

mylist.append("maybe")

print(mylist[2])
## 'maybe'

mylist[2] = "never"

del mylist[2]
#or
mylist.remove("never")
```

- slicing

```python
mylist[0:2]
# prints first two elements (call is inclusive of bottom number and
# noninclusive of upper number)

mylist[-1]
# last element
```

### Dicts

- slicing dicts

```python
dictionary = {"apples":3, "bananas":4, "pears":5, "lemons":10, "tomatoes": 7}

keys_for_slicing = ["apples","lemons"]

sliced_dict = {key: dictionary[key] for key in keys_for_slicing }

print(sliced_dict)

#Output:
{'apples': 3, 'lemons': 10}
```

- crud

```python
del dict["key"]
```

```python
dict = {"yes": 3, "no": 4}

dict.get("maybe", "default vlue")
## 'default value'

dict.get("no", "default vlue")
## 4
```

### Loops

- make sure order isn’t making some conditions obsolete
- make “for” loops shorter with list comprehension

```python
words = ["name", "name1", "name2"]

new_words = []
for word in words:
		new_words.append(word.capitalize())

new_words = [word.capitalize() for word in words]
						# output <- loop
```

```python
for ix, (key, value) in enumerate(dict.items()):
		print(ix, key, value)
## 0 key1 value1
## 1 key2 value2...
```

### Functions

- kwargs: keyword arguments (fluid position with default value)

```python
def is_even(other=0,number=0):
	print number % 2 == other

# if you provide a default value for the parameter, it becomes a **keyword argument**
# allows the position to become fluid in calling the function, by assigning
	# the values in the call:
print(is_even(number=2))
#or
print("is odd: ", is_even(number=2, other=1)

def mydef(arg1, arg2, keywordarg=0)
		return ....
# keyword args must happen after positional args
```

### Internal Modules

- directory > __init__.py > functions

```python
from module(directory) import function
```

## Debugging

- print variables between lines of suspect code
- ipython > run lines of code in an isolated way (can import variables)
- NameError (variables), KeyError, TypeError, NonetypeError (probably forgot to return func), SyntaxErrror

### Advanced Debugger

- breakpoint() (ends code at a certain point) in **terminal**
    - python app.py
    - breakpoint() (insert break point function **in file)**
    - next() (runs next line)
    - var1 (prints variable)
    - continue() (executes the rest of the code or until the next break point)

# Recap

- markdown cell - m
- back to code cell - y
- shift+return - run

```python
def fahdjfkhs(number):
		mapping = {
				"1": "black",
				"2": "white",
				...
				}

return mapping.get(fahdjfkhs[0], other) ????
```
