# Syntax and Data Types

# Naming Conventions

### Packages/Modules

- pandas, numpy

### Classes

- DataFrame, Student

### Variables/Functions

- name, first_name, birth_year()

# Printing

```python
print("Hello World!")

print("Hey,\n how's it going?")
# \n is a line break

print("\"Hey,\" he said,")
# \" adds parentheses inside a string
```

# Inputs

- inputs are always a string

### Strings

```python
name = input("Enter your name: ")
print("Hello " + name + "!")

color = input("Enter a color: ")
plural_noun = input("Enter a plural noun: ")
celebrity = input("Enter a celebrity: ")
print("Roses are " + color + ",")
print(plural_noun + " are blue,")
print("I love " + celebrity)
```

### Calculator

```python
num1 = input("Enter a number: ")
num2 = input("Enter second number: ")
result = int(num1) + int(num2)
print(result)

num1 = input("Enter a number: ")
num2 = input("Enter second number: ")
result = float(num1) + float(num2)
print(result)
```

# Formatting

```python

<str> = '{}, {}'.format(<el_1>, <el_2>)  
# Or: '{0}, {a}'.format(<el_1>, a=<el_2>)

args = range(4)
print(("{}_"*4).format(*args))
## "1_2_3_3_"
# the * unpacks the following argument
```

# Introspection

```python
help(int)
# gets help doctring file on integers

def increment(n):
    return n + 1
print(increment)
## function increment at 0x7f420e2973a0
# gets info about object, function, etc
```

# Data Types

```python
eval(expression, globals=None, locals=None)
eval("1 + 3")
## 4
# can take string and run math inside

isinstance(obj, class)

type(object, bases, dict)
# bases is uple of classes from which the current class derives
# a dictionary that holds the namespaces for the class
print(type(123))
## class int
print(type("fits"))
## class str

dir(object)
# return a valid list of attributes of the object it is called upon
# class obj: list of names of the valid attributes and base att.
# mods/libs: list of atts.
# if None: list of names in current local scope
print(dir(roger))
## ['__class__', '__delattr__',...]

id(object)
# memory address (unique id) of object
print(id(1))     
## 140227521172384
```

## Strings - plain text

```python
your_name_here = "Emily"
my_name_here = "LeeAnne"

our_names = your_name_here + " and " + my_name_here
#appending strings

str(...)
```

### Functions of strings

```python
my_name = "emily"

print(my_name.upper())
## EMILY

print(my_name.lower())
## emily

print(my_name.capitalize())
## Emily

print(my_name.islower())
#prints bool of if the string is all lowercase

print(len(my_name))
#prints the **length** of the string (5)

print(my_name[0])
#prints the character with **index** of 0, which is the first character

print(my_name.index("i"))
#prints the index **first** location of the character - 2

print(my_name.replace("E", "e"))
#prints a version of the string with altered contents

my_name.strip("e")
## mily

<list> = <str>.split()                       
# Splits on one or more whitespace characters.

<list> = <str>.split(sep=None, maxsplit=-1)  
# Splits on 'sep' str at most 'maxsplit' times.

<str>  = <str>.strip()                       
# Strips all whitespace characters from both ends.

<str>  = <str>.strip('<chars>')              
# Strips all passed characters from both ends.

<str>  = <str>.join(<coll_of_strings>)       
# Joins elements using string as a separator.
```

### String interpolation

```python
n1 = 'Hello'
n2 = 'GeeksforGeeks'
print(f"{n1}! This is {n2}")
##Hello! This is GeeksforGeeks
```

### Slicing

Slice notation takes the form `[start:stop:step]`
. In this case, we omit the `start`
 and `stop`
 positions since we want the whole string. We also use `step = -1`
, which means, "repeatedly step from right to left by 1 character".

```python
'hello world'[::-1]
## 'dlrow olleh'
```

### Split

## Numbers - integers and floating points (decimals)

```python
your_age_here = 31
your_happiness_here = 6.58

int(...)
float(...)
```

### Functions of numbers

```python
# operations: + - / * %(mod: remainder from division)

my_num = 5
print(str(my_num))
#prints string of the number

print(abs(my_num))
#prints absolute value

print(pow(my_num, 2))
#prints first number squared

print(max(my_num, 10))
#prints largest number

print(min(my_num))
#prints smallest number

print(round(my_num, 2))
#prints rounded number, to number of decimals

new_num = int(4.39)
#will also truncate (floor) without needing math

sum(my_num, optional_added_value)

from math import *
#imports more comlicated python math functions
print(floor(my_num))
#prints number rounded down to next integer

print(ceil(my_num))
#prints number rounded up to next integer

print(sqrt(my_num))
#prints square root of a number
```

## Boolean: Bools - true/false values

```python
your_truth_value_here = False or True
```

## Constants & Enums

There is no constant class in Python (two workarounds)

```python
WIDTH = 67
# variables in all uppercase are not to be changed (even if they can be)

from enum import Enum

class Constants(Enum):
		WIDTH = 123
		HEIGHT = 45
# Constants is the State

print(Constants.WIDTH.value)

AREA = Constants.WITH * Constants.HEIGHT
#or
AREA = Constants(1) * Constants(2)
#or
AREA = Constants['WIDTH'] * Constants['HEIGHT']
```

### Functions of enums

```python
list(State) 
# [<State.INACTIVE: 0>, <State.ACTIVE: 1>]

len(State) 
# length
```

## Variables

global variable

```python
color = "blue"
def test():
		print(color)
# print(color) & print(test) will have same val
```

local variable

```python
def test():
		color = "blue"
		print(color)
# print(color) will come back as error
```

## Lists

```python
friends = ["Larissa", "Ana", "Komali"]

print(friends[2])
#prints friend with third index

print(friends[-2])
#prints friend at second to last index

print(friends[1:])
#prints friends including and after index of 1

print(friends[1:2])
#prints friends in second index, up to but not including limit

friends[1] = "Amrit"
#modifies second friend in list
```

### Functions of lists

```python
lucky_numbers = [3, 7, 21, 32, 60, 77]
friends = ["Larissa", "Ana", "Amrit", "Komali", "Ellen"]

friends.extend(lucky_numbers)
# adds second list to end of first list

new_friend = "Eleanor"
friends.append(new_friend)
# adds element to end of list

friends.insert(1, "Rhonda")
# adds element to specific index

friends.remove("Komali")
# removes element from list

friends.clear()
# deletes all elements in list

friends.pop()
# removes last element in list

print(friends.index("Ana"))
# finds element in list and if present, prints the index location

print(friends.count("Ana"))
# prints the number of repetitions/duplicates of an element

len(friends)
#counts the elements in a list

friends.sort()
print(friends)
# prints list sorted in alphabetical order

lucky_numbers.sort()
#sorts list in ascending order

lucky_numbers.reverse()
#reverses the order of the list

friends2 = friends.copy()
#makes new list with same contents of existing list
```

### 2D (Nested) Lists

```python
number_grid = [
		[1, 2, 3],
		[4, 5, 6],
		[7, 8, 9],
		[0]
]

print(number_grid[0][0])
# each number is the index location of the row, then column

list1 = []
number_of_items = int(input())

for i in range(number_of_items):
		list2 = []
		size_of_sublist = int(input())
		for j in range(size_of_sublist):
				value = int(input())
				list2.append(value)
		list1.append(list2)
print(list1)
# takes input and makes nested lists
```

### Nested Lists Adv

```python
# takes user input of the number of students, 
# then their names and grades. returns the names 
# in alphabetical order of the students with 
# the second-to-lowest grades

students = []
student_count = int(input())

for i in range(student_count):
    student_info = [str(input()), float(input())]
    students.append(student_info)

students.sort(key = lambda x : x[1])

i = 1
for row in range(student_count):
    if students[i][1] > students[0][1]:
        second_lowest_grade = students[i][1]
    else:
        i += 1

sec_low_alph = []
n = 0

for student_name in range(student_count):
    if second_lowest_grade == students[n][1]:
        sec_low_alph.append(students[n][0])
        n += 1
    else:
        n += 1
        
sec_low_alph.sort()
j = len(sec_low_alph)
k = 0
for names in range(j):
    print(sec_low_alph[k])
    k += 1
```

## Dictionaries

- stores data in key value pairs

```python
monthConversions = {
		"Jan": "January",
		"Feb": "Febuary",
		"Mar": "March",
		"Apr": "April",
}
# keys must be unique

print(monthConversions["Feb"])
# prints value associated with the key

print(monthConversions.get("Dec", "Not a valid key"))
# passes a default value if key not found

for key, value in monthConversions.items():
    print(key, ' : ', value)
#prints each row on new line

for key in monthConversions:
  print(key)
#prints keys

for key in monthConversions:
  print(thisdict[key])
#prints values

monthConversions["Jan"].append("Cold")
#append to dictionary value
```

### Dictionary Functions

```python
monthConversions["Feb"] = "February"
# edits value from key

april = monthConversions.pop("Apr")
# returns value and deletes row

lastmo = monthConversions.popitem()
# retuns and deletes last row

ifFeb = "Feb" in monthConversions
# searches for key, creates bool

abreviations = list(monthConversions.keys())
# gets a list of the keys

months = list(monthConversions.values())
# gets a list of the values

monthTups = list(dog.items())
# gets a list of tuples of the keys and values

len(monthConversions)
# gets rows

monthConversions["May"] = "May"
# add new row

del monthConversions["Feb"]
# deletes row

monthConversionsCopy = monthConversions.copy()
# copies data into new dict
```

### Iteration

```python
for key in myDict:
    print(key)
# iterates over keys

for value in myDict.values():
    print(value)
# iterates over values

for key, value in myDict.items():
    print(key, ":", value)
# iterates over both

for key in myDict:
    print(key, ':', myDict[key])
# both

```

### Nested Dicts

```python
myDict = {
		"letters": {"a": 1, "b": 2, "c": 3}, 
		"numerals": {"I": 1, "V": 5, "X": 10}}

three = myDict["letters"]["c"]
five = myDict["numerals"]["V"]
```

### Lists to dict using dict comprehension

```python
students = ['Peter', 'Mary', 'George', 'Emma']
student_ages = [24, 25, 22, 20]
student_ages_dict = {key: value for key, value in zip(students, student_ages)}
print(student_ages_dict)
#=> {'Peter': 24, 'Mary': 25, 'George': 22, 'Emma': 20}
```

## Tuple

- unchangeable, un-editable list, good for constants

```python
coordinates = (4, 5)
```

## Sets

- like tuple but unordered and editable

```python
set0 = {"Rhonda"}
set1 = {"David", "Rhonda"}
set2 = {"Bill", "Rhonda"}
set3 = {"Jose", "Rhonda"}
set4 = {"Todd", "Deanna"}
set5 = {"David", "Sheryl"}

intersect = set1 & set2
# returns what's the same of sets {"Rhonda"}

union = set3 | set5
# returns combo of sets {"Jose", "Rhonda", "David", "Sheryl"}

difference = set1 - set0
# subtracts elements of second from first set {"David"}

isSuperset = set3 > set0
isSubset = set0 < set2

setlist = list(set3)

setdict = {
  "Rhonda": {
    1970: set0,
    1980: set1,
    1990: set1,
    2000: set2,
    2010: set3,
    2020: set3
  },
  "David": {
    1980: set1,
    1990: set1,
    2000: set5,
    2010: set5,
    2020: set5
  },
  "Todd": {
    "all": set4
  }
}

print("Rhonda" in set2)
# bool check
```

# Docstrings

```python
def increment(n):
    """Increment a number"""
    return n + 1

"""Dog module

This module does ... bla bla bla and provides the following classes:

- Dog
...
"""

class Dog:
    """A class representing a dog"""
    def __init__(self, name, age):
        """Initialize a new dog"""
        self.name = name
        self.age = age

    def bark(self):
        """Let the dog bark"""
        print('WOF!')
```

# Annotations

- optional specification of the type of variable, function return, or function parameter

```python
count: int = 0

def increment(n: int) -> int:
		return n + 1
```

# Date

```python
import datetime

today = datetime.date.today()

tomorrow = datetime.date.today() + datetime.**timedelta**(days=1)
# tomorrow

formatted_dt = datetime.datetime.replace(second=0, microsecond=0)
```
