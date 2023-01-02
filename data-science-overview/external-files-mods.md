# External Files & Mods

## Reading Files

```python
open("food.txt", "r")
# read only

open("food.txt", "w")
# can write file

open("food.txt", "a")
# can append info into file, but not modify existing

open("food.txt", "r+")
# read and write
```

```python
food_file = open("food.txt", "r")
...
food_file.close()
```

```python
food_file = open("food.txt", "r")
print(food_file.readable())
# bool val
print(food_file.read())
# prints info
print(food_file.readline())
# prints 1st line, next iteration will read second line, etc.
print(food_file.readlines())
# creates a list with each line as a new elemet
print(food_file.readlines()[i])
# prints i position in list of lines
food_file.close()

food_file = open("food.txt", "r")
for food in food_file.readlines():
		print(food)
food_file.close()
# prints each element as new line
```

## With

- built-in exception handling
- for files: close() will be called automatically

```python
filename = '/Users/flavio/test.txt'

with open(filename, 'r') as file:
		content = file.read()
		print(content)
```

## Writing & Appending to Files

```python
food_file = open("food.txt", "a")
food_file.write("Bananas - Fruit")
food_file.write("\nApples - Fruit")
# \n adds line break
food_file.close()
# adds to next line in file

food_file = open("food.txt", "w")
food_file.write("Avocados - ?")
# replaces contents of file with new line (deletes existing info)
food_file.close()

food_file = open("food1.txt", "w")
# creates new file 

food_file = open("index.html", "w")
food_file.write("<p>This is HTML,<p>")
food_file.close()
```

# External Libraries

```python
import math 
print(math.pi)
# math module

help(math.log)
# explaines what .log does

import pandas as pd
#imports module with shorter name

from math import *
print(pi)
#imports variables so you don't need math.pi

type(pi)
# tells you what it is

dir(rolls)
# directory

.tolist 
#changes array to list
```

## Modules & Pip

- some modules are build in, but others are “external” and need to be imported

```python
import conversions
# finds file with functions (module)
print(conversions.meters_to_feet(5))
# prints result of external function file
```

- pip is a preinstalled package manager

```python
## go to terminal:

pip3 --version
# check to see if it's there
pip install python-docx
# installs python-docx 
# places into externals>lib>site packages

## then back to compiler:

import docx
```

```python
pip install -U <package>
# ungrades package to latest version

pip install <package>==<version>
# installs specific version

pip uninstall <package>

pip show <package>
#show details, version, documentation, author
```

## Command Line

```python
# REPL, python <filename>.py
# python <filename>.py <argument1> <argument2>

import sys
print(len(sys.argv))
print(sys.argv)
# The sys.argv list contains as the first item the name of the file that was run

import argparse
parser = argparse.ArgumentParser(
    description='This program prints a color HEX value'
)
parser.add_argument('-c', '--color', metavar='color', required=True, help='the color to search for')
args = parser.parse_args()
print(args.color) 
# 'red'

parser.add_argument('-c', '--color', metavar='color', required=True, choices={'red','yellow'}, help='the color to search for')
# options to have specific set of values
```

# Virtual Environments

```python
# create:
python -m venv .venv

source .venv/bin/activate
# now, running pip will use this VE instead of the global
```