import numpy as np


def fft(a, n, invert=False):
    """
    Perform the Fast Fourier Transform (FFT) or Inverse FFT.

    Parameters:
        a (list): Coefficient array of the polynomial.
        n (int): The length of the polynomial (next power of 2).
        invert (bool): Whether to perform the inverse FFT.

    Returns:
        list: The FFT-transformed array.
    """
    if len(a) < n:
        a.extend([0] * (n - len(a)))  # Pad with zeros to length n

    if n == 1:
        return a

    omega_n = np.exp((2j * np.pi / n) * (-1 if invert else 1))
    omega = 1

    a_even = fft(a[0::2], n // 2, invert)
    a_odd = fft(a[1::2], n // 2, invert)

    y = [0] * n
    for k in range(n // 2):
        y[k] = a_even[k] + omega * a_odd[k]
        y[k + n // 2] = a_even[k] - omega * a_odd[k]
        if invert:
            y[k] /= 2
            y[k + n // 2] /= 2
        omega *= omega_n

    return y


def multiply_polynomials(f, g):
    """
    Multiply two polynomials using FFT.

    Parameters:
        f (list): Coefficients of the first polynomial.
        g (list): Coefficients of the second polynomial.

    Returns:
        list: Coefficients of the resulting polynomial.
    """
    m = len(f)
    n = len(g)
    size = 1
    while size < m + n - 1:
        size *= 2

    f_fft = fft(f, size)
    g_fft = fft(g, size)

    result_fft = [f_fft[i] * g_fft[i] for i in range(size)]

    result = fft(result_fft, size, invert=True)

    return [round(x.real) for x in result]


# Example usage
if __name__ == "__main__":
    poly1 = [1, 2, 3]  # Represents 1 + 2x + 3x^2
    poly2 = [4, 5, 6]  # Represents 4 + 5x + 6x^2

    result = multiply_polynomials(poly1, poly2)
    print("Resulting Polynomial Coefficients:", result)
