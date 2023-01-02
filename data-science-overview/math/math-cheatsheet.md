# Math

- ∈ : an element of
- scalar: container for a single value (data type: a, b, x, y, ø)
- vector: collection of scalars (row or column list)
- matrix: 2d list

- funtions:
    - f : X → Y
    - y = f(x), x is an element of X, y is an element of Y
    - f : x → y, x is an element of X, y is an element of Y
- X is the domain, Y is the codomain
- x is the input variable and y is the output var

### Derivatives: (d/dx) - f’(x)

![definition-derivative-formula.png](math-cheatsheet/definition-derivative-formula.png)

- slope of tangent line to curve at any point
- power rule:

$d/dx(x^n) = n * x^{n-1}$

$d/dx(cx) = c$

$d/dx(x) = 1$

$d/dx(c) = 0$

- ex:

    $f(x) = x^4 + 2x^3 -x^2 + 4x - 1$

    $f'(x) = 4x^3 + 6x^2 - 2x + 4$


$d/dx(x^{-2}) = -2x^{-3}$

- vector magnitude:

    $||a||p=(n∑i=1|ai|p)^(1/p)$


# Intro to Stats

## Histograms

- 2d graph that shows the amount of measurements per value

|  |  | o |  |
| --- | --- | --- | --- |
|  | o | o |  |
| o | o | o | o |

## Distribution

- The probability of where measurements are distributed
- histograms, graphs, curve

### Normal distribution

- Gaussian curve: “bell curve”
- greatest distribution in the middle (mean) of the scale of measurements
- height, weight, commuting times
1. average measurement (center)
2. standard deviation (width)

# Population Parameters

- normal:
    - population mean
    - population SD (standard deviation)
- exponential distribution:
    - population rate
- gamma distribution: (mean is not in the center, but not the edge)
    - pop shape
    - pop rate
- training dataset:
    - 5 parameters that allow you to see if your curve is reproducible (within SD)
    - more data sets, closer to the true value, more confidence
    - p value - confidence interval

# Calculating the numbers

- population mean µ
    - average of the data (the true mean because there are so many)
- variance:
    - x = value, u = average, n = number of values

        $(Σ(x - µ)^2) / n$

- standard deviation

$√ (Σ(x - µ)^2) / n$

- square root of the population variance

# Estimating the numbers

- **sample mean**: taken from few samples x̄
- estimated pop variance:

    $(Σ(x - x̄)^2) / n - 1$

- estimated SD

    $√ estimated pop var$


# Data Sets

- individuals: the things being studied or measured (identifier)
- categorical variables: measurements that lie in 2+ categories (words)
- quantitative variables: measurements that don’t fall into categories (numbers)
- marginal distribution: %s of a selection of data from a wider dataset
- conditional distribution: %s of a selection of data (between two smaller endpoints as well)
    - distribution of (y axis) for each (x axis)
- stem & leaf plot


    | 0 | 0 0 2 4 7 7 9 |
    | --- | --- |
    | 1 | 1 1 3 8  |
    | 2 | 0 |
- distribution shapes:
    - left/right - tailed
    - symmetrical
    - skewed to the left/right
- median: middle value
- mean: average
- mode: most common measurement
- interquartile ranges (IQR)
    - measure of spread; how far apart the data points are
    - find the median, then the median of the first and last halves
    - median of the second half (minus) the median of the first half
- MAD:
    - abs value of summed value - mean/med/mode

$(Σ | x_i - m(X)|)/n$
