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


def multiply_polynomials_classical(f, g):
    """
    Multiply two polynomials using classical O(n²) method.

    Parameters:
        f (list): Coefficients of the first polynomial.
        g (list): Coefficients of the second polynomial.

    Returns:
        list: Coefficients of the resulting polynomial.
    """
    m, n = len(f), len(g)
    result = [0] * (m + n - 1)

    for i in range(m):
        for j in range(n):
            result[i + j] += f[i] * g[j]

    return result


def multiply_polynomials_karatsuba(f, g):
    """
    Multiply two polynomials using Karatsuba algorithm.
    Time complexity: O(n^log₂3) ≈ O(n^1.58)

    Parameters:
        f (list): Coefficients of the first polynomial.
        g (list): Coefficients of the second polynomial.

    Returns:
        list: Coefficients of the resulting polynomial.
    """

    def pad_zeros(arr, length):
        return arr + [0] * (length - len(arr))

    def karatsuba_recursive(f, g):
        n = max(len(f), len(g))
        if n <= 1:
            return [f[0] * g[0]] if f and g else [0]

        # Make lengths equal and a power of 2
        m = (n + 1) // 2
        f = pad_zeros(f, n)
        g = pad_zeros(g, n)

        # Split polynomials
        f_low = f[:m]
        f_high = f[m:]
        g_low = g[:m]
        g_high = g[m:]

        # Recursive steps
        p1 = karatsuba_recursive(f_low, g_low)  # Low * Low
        p2 = karatsuba_recursive(f_high, g_high)  # High * High

        # (f_low + f_high)(g_low + g_high)
        f_sum = [f_low[i] + f_high[i] for i in range(m)]
        g_sum = [g_low[i] + g_high[i] for i in range(m)]
        p3 = karatsuba_recursive(f_sum, g_sum)

        # p3 - p1 - p2 gives us the middle term
        for i in range(len(p1)):
            p3[i] -= p1[i]
        for i in range(len(p2)):
            p3[i] -= p2[i]

        # Combine results
        result = [0] * (2 * n)
        for i in range(len(p1)):
            result[i] += p1[i]
        for i in range(len(p3)):
            result[i + m] += p3[i]
        for i in range(len(p2)):
            result[i + 2 * m] += p2[i]

        return result

    return karatsuba_recursive(f, g)


# Example usage
if __name__ == "__main__":
    poly1 = [1, 2, 3]  # Represents 1 + 2x + 3x^2
    poly2 = [4, 5, 6]  # Represents 4 + 5x + 6x^2

    print("FFT Multiplication:", multiply_polynomials(poly1, poly2))
    print("Classical Multiplication:", multiply_polynomials_classical(poly1, poly2))
    print("Karatsuba Multiplication:", multiply_polynomials_karatsuba(poly1, poly2))
