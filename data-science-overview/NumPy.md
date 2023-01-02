# NumPy

- numeric python
- NumPy Array replaces Py list
- calcs over entire arrays w/o loops
- only one data type per array, or will convert all to one (speed)
- numpy.ndarray(*shape*, *dtype=float*, *buffer=None*, *offset=0*, *strides=None*, *order=None*)

```python
pip3 install numpy

import numpy as np

height = [1.73, 1.71, 1.68, 1.98, 1.79]
np_height = np.array(height)
## array([1.73, 1.71, 1.68, 1.98, 1.79])

bmi = np_weight / np_height ** 2
## array([...])
```

### conversions

```python
np.array([1,2,3,True,False])
## 1,2,3,1,0
# true == 1, false == 0
```

### 2-D Arrays

```python
np_2d = np.array([[1.75, 1.64, 1.94, 1.45], 
								[65.4, 75.3, 59.5, 70.2]])

np_2d.shape
## (2, 4) 
# 2 rows, 4 columns

**# [row][column]**
np_2d[0][2]
np_2d[0, 2]
## 1.94

np_2d[:, 1:3]
## array([[1.64, 1.94],
##				[75.3, 59.5]])
# all rows, columns 1, up to 3 (but not incluing 3)

np_2d[1, :]
# first row, all columns

np_2d[:, 1:]
# all rows, second column on
```

### Homogeneity

```python
np.array(['Bob', 3, True])
# all to strings
np.array([1,2,True])
# all to ints
np.array([1, 'string', {1:2}])
# no change
```

### 3D+ arrays (library)

- location on screen(3d coords), color(3d rgb), location in movie(timecode), etc

```python
np.array([ [ [1,2,3] ] ])
```

### Attributes

```python
array.ndim
# number of dimensions
array.shape
# size along each dimension
array.size
# total number of elements
array.dtype
## int64 (integer, 64 bites)
```

### Data Selection (slicing)

```python
data_list = [[0,1,2,3,4],...]
data_np = np.array(data_list)

data_np[1:5, 2:5]
# slices to second through fifth row, third through fifth column 
**# upper non-inclusive**
```

```python
data_np[:, 0:5:2]
# all rows, every other column from first to fifth
```

### Axes

- 1D: column vectors (4,)
- 2D: row, column (2,3)
- 3D: row, column, depth (4,3,2)

### Vectorized Operations

```python
python_list = [1,2,3]
numpy_array = np.array([1,2,3])

python_list + python_list
## [1,2,3,1,2,3]
# appends to list

numpy_array + numpy_array
## [2,4,6]
# sums list w/ itself
```

```python
A1 = np.array([1,2,3,4,5])
A2 = np.array([1,2,3,4,5])

A1 * A2
## 1, 4, 9, 16, 25

B1 = np.array([1,2,3])
A1 * B1
## error

# broadcasting:
B = ([[1,2,3],[...],[...]])
B * B1
## 1,4,9
## 5,10,15
## 3,4,21
```

### Boolean Indexing (sub-setting)

```python
bmi = np.array([20, 21, 22, 23, 24])
bmi[1]
## 21
bmi > 23
## array([False, False, False, True])
bmi[bmi > 23]
array([24])
# gives back an array that's only the values that were true

b = np.array([[1,2,3],[4,5,6], [7,8,9]])
b[b > 4] = 99
# replaces "True" values
## ([[1,2,3],[4,99,99],[99,99,99]])
```

```python
np.sum(array_1, axis=1)
# sums the columns
np.sum(array_1, axis=0)
# sums the rows

np.prod(array, axis)
#multiplies
```

## Statistics

```python
import numpy as np
np_city  ... # impolementation left out
np_city

## array([[1.64, 71.78], 
##				[1.37, 63.35], 
##			  [1.6, 55.09]...])

np.mean(np_city[:, 0])
## 1.7472
# all rows, first column

np.median(np_city[:, 0])
## 1.75
# all rows, first column

np.corrcoef(np_city[:, 0], np_city[:, 1])
## array([[1.      , -0.01802], 
##			  [-0.01803,  1.     ]])
# check if they are correlated

np.sum()
np.sort()
```

```python
np.random.normal(distribution mean, distribution standard deviation, 
								 number of samples)

height = np.round(np.random.normal(1.75, 0.20, 5000), 2)
weight = np.round(np.random.normal(60.32, 15, 5000), 2)
np_city = np.column_stack((height, weight))
```

### Examples

```python
a = np.array([[1, 0], [0, 1]])
F = np.tile(a, (4, 4))
```