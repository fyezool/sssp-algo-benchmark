# Breaking the Sorting Barrier: A Benchmark for SSSP Algorithms

This repository contains a Python implementation and a comprehensive benchmark suite for the groundbreaking "Sorting Barrier" algorithm for the Single-Source Shortest Path (SSSP) problem. It compares the performance of this novel algorithm against classic Dijkstra implementations across a variety of graph structures and sizes.

The implementation is based on the concepts presented in the research paper: **["Breaking the Sorting Barrier for Directed Single-Source Shortest Paths" (arXiv:2504.17033)](https://arxiv.org/pdf/2504.17033)**.

## ✨ Features

*   **Pure Python Implementation**: Requires no external libraries or dependencies. Just a standard Python interpreter.
*   **Algorithm Comparison**: Benchmarks a novel **Sorting Barrier SSSP** algorithm against **Dijkstra's algorithm** (with a binary heap) and a simulated **Fibonacci Heap Dijkstra**.
*   **Comprehensive Benchmark Suite**: Includes tests for scalability (increasing node count), graph density, and various graph topologies (random sparse/dense, grid, path).
*   **Correctness Verification**: Ensures all algorithms produce identical, correct results before measuring performance.
*   **Detailed Analysis**: Automatically calculates and prints a summary analysis, including performance speedup, theoretical efficiency, and win/loss rates.
*   **Data Export**: Saves all raw benchmark results to a `sssp_benchmark_results.csv` file for further analysis.
*   **User-Friendly CLI**: An interactive menu allows you to easily run different test suites, from a quick verification to a full analysis.

## 🚀 How to Run

1.  Make sure you have Python 3 installed.
2.  Save the code as `benchmark.py`.
3.  Run the script from your terminal:
    ```sh
    python benchmark.py
    ```
4.  You will be prompted to choose a benchmark mode:

    ```
    SSSP Sorting Barrier Algorithm Benchmark
    Choose benchmark mode:
    1. Quick verification test (recommended first)
    2. Full benchmark suite
    3. Detailed complexity analysis
    4. All tests (comprehensive)

    Enter choice (1-4) [default: 1]:
    ```

### Benchmark Modes Explained

*   **1. Quick verification test**: Runs the algorithms on a small, fixed graph to verify that the implementations are correct and working. **Recommended to run first.**
*   **2. Full benchmark suite**: Executes a series of tests for scalability, density, and graph structure. This provides a broad overview of performance.
*   **3. Detailed complexity analysis**: Focuses on testing the theoretical speedup claims of the paper by running tests on various graph sizes and densities and comparing the actual speedup to the theoretical one.
*   **4. All tests (comprehensive)**: Runs all of the above tests sequentially.

## 📊 Expected Output

The script will print live results to the console, followed by:
1.  A summary analysis table comparing the performance of the algorithms.
2.  A text-based visualization of the performance results.
3.  A detailed analysis of the "sorting barrier breakthrough," showing where the new algorithm outperforms the classics.

Additionally, a CSV file named `sssp_benchmark_results.csv` will be generated in the same directory, containing the detailed results from all benchmark runs.

**Example Console Output:**
```
================================================================================
BREAKTHROUGH ANALYSIS SUMMARY
================================================================================
📊 OVERALL PERFORMANCE:
   Average actual speedup:      1.153x
   Average theoretical speedup: 1.581x
   Implementation efficiency:   72.9%

🎯 SORTING BARRIER BREAKTHROUGH:
   Cases where sorting barrier wins:        11/15 (73.3%)
   Cases with >10% speedup:                 6/15 (40.0%)
   Best speedup achieved:                   1.549x
   Average speedup when winning:            1.291x
   Best case: n=750, m=7490, density=0.013

📈 DENSITY ANALYSIS:
   Density 0.05: 60.0% win rate, 1.109x avg speedup
   Density 0.10: 80.0% win rate, 1.189x avg speedup
   Density 0.20: 80.0% win rate, 1.161x avg speedup

📏 SCALABILITY ANALYSIS:
   n=100:  33.3% win rate, 0.963x avg speedup
   n=200: 100.0% win rate, 1.180x avg speedup
   n=350: 100.0% win rate, 1.205x avg speedup
   n=500: 100.0% win rate, 1.228x avg speedup
   n=750: 100.0% win rate, 1.353x avg speedup
```

## 🧠 About the "Sorting Barrier" Algorithm

The `OptimizedSSSPSortingBarrier` class is an implementation based on the algorithm introduced in "Breaking the Sorting Barrier for Directed Single-Source Shortest Paths." Traditional SSSP algorithms like Dijkstra's are fundamentally limited by a "sorting barrier," leading to a complexity of `O(m + n log n)`.

The new algorithm cleverly avoids this barrier by using techniques like limited-depth graph exploration and a specialized heap structure (`PartialSortHeap`) to achieve a theoretical complexity of **O(m log⅔ n)**. For many graphs, particularly sparse ones, this provides a significant theoretical (and practical) performance improvement. This benchmark suite is designed to empirically test and validate these claims.

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.
