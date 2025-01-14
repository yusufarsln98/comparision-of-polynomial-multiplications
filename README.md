# Symbolic Computation Project

This project implements and compares different polynomial multiplication algorithms, including:

- Fast Fourier Transform (FFT) based multiplication
- Classical O(n²) multiplication
- Karatsuba algorithm O(n^1.59) multiplication

The project includes benchmarking tools to analyze and visualize the performance differences between these algorithms.

## Project Structure

- `multiplications.py`: Contains the implementation of three polynomial multiplication algorithms
- `benchmarking.py`: Performance analysis and visualization tools for comparing the algorithms
- `polynomial_benchmark.png`: Generated performance comparison plot

## Setup and Installation

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the virtual environment:

- Windows:

```bash
venv\Scripts\activate
```

- Unix/MacOS:

```bash
source venv/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

After setting up the environment, you can:

1. Run benchmarks to compare algorithm performance:

```bash
python benchmarking.py
```

2. Use individual multiplication algorithms in your code:

```python
from multiplications import (
    multiply_polynomials,  # FFT-based
    multiply_polynomials_classical,
    multiply_polynomials_karatsuba
)

# Example: Multiply two polynomials
poly1 = [1, 2, 3]  # Represents 1 + 2x + 3x²
poly2 = [4, 5, 6]  # Represents 4 + 5x + 6x²
result = multiply_polynomials(poly1, poly2)
```

The benchmarking script will generate a plot comparing the performance of different multiplication methods across various polynomial sizes.

## Requirements

See `requirements.txt` for a list of dependencies.
