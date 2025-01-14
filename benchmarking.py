import numpy as np
import time
import matplotlib.pyplot as plt
from multiplications import (
    multiply_polynomials,
    multiply_polynomials_classical,
    multiply_polynomials_karatsuba,
)


def generate_random_polynomial(size):
    """Generate a random polynomial with integer coefficients."""
    return list(np.random.randint(-10, 10, size))


def benchmark_multiplication(sizes, num_trials=3):
    """
    Benchmark different polynomial multiplication methods.

    Parameters:
        sizes (list): List of polynomial sizes to test
        num_trials (int): Number of trials for each size to average over

    Returns:
        dict: Dictionary containing timing results for each method
    """
    results = {"FFT": [], "Classical": [], "Karatsuba": []}

    for size in sizes:
        fft_times = []
        classical_times = []
        karatsuba_times = []

        for _ in range(num_trials):
            # Generate random polynomials of current size
            poly1 = generate_random_polynomial(size)
            poly2 = generate_random_polynomial(size)

            # Benchmark FFT multiplication
            start_time = time.time()
            multiply_polynomials(poly1.copy(), poly2.copy())
            fft_times.append(time.time() - start_time)

            # Benchmark Classical multiplication
            start_time = time.time()
            multiply_polynomials_classical(poly1.copy(), poly2.copy())
            classical_times.append(time.time() - start_time)

            # Benchmark Karatsuba multiplication
            start_time = time.time()
            multiply_polynomials_karatsuba(poly1.copy(), poly2.copy())
            karatsuba_times.append(time.time() - start_time)

        # Store average times
        results["FFT"].append(np.mean(fft_times))
        results["Classical"].append(np.mean(classical_times))
        results["Karatsuba"].append(np.mean(karatsuba_times))

        # Print progress
        print(f"Completed benchmarking for size {size}")

    return results


def plot_results(sizes, results):
    """Plot the benchmark results in linear scale."""
    plt.figure(figsize=(10, 6))

    for method, times in results.items():
        plt.plot(sizes, times, marker="o", label=method)

    plt.xlabel("Polynomial Size")
    plt.ylabel("Time (seconds)")
    plt.title("Performance Comparison")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("polynomial_benchmark.png")
    plt.close()


if __name__ == "__main__":
    # Test with polynomials of increasing sizes
    # Using powers of 2 for FFT compatibility
    # sizes = [2**i for i in range(3, 11)]  # From 8 to 1024
    sizes = [256, 512, 1024, 2048, 4096, 8192]

    print("Starting benchmark...")
    results = benchmark_multiplication(sizes)

    # Print results in a table format
    print("\nResults (time in seconds):")
    print("Size\t\tFFT\t\tClassical\tKaratsuba")
    print("-" * 50)
    for i, size in enumerate(sizes):
        print(
            f"{size}\t\t{results['FFT'][i]:.6f}\t{results['Classical'][i]:.6f}\t{results['Karatsuba'][i]:.6f}"
        )

    # Plot results
    plot_results(sizes, results)
    print(
        "\nBenchmark complete! Results have been plotted to 'polynomial_benchmark.png'"
    )
