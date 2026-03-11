"""Performance benchmark: C++ CPU vs CUDA vs JAX adding-doubling solver.

Mirrors the benchmark_cuda.cu configurations and measures all three backends.
Usage:
    python tests/benchmark_all.py [--no-cuda] [--no-cpp]
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

# Add parent directory to path for src_jax import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_jax.batch_solver import BatchConfig, solve_batch


# ============================================================================
#  Test atmosphere (mirrors benchmark_cuda.cu buildAtmosphere)
# ============================================================================

def build_atmosphere(nlay, nwav, nmu, nmom, source="thermal"):
    """Build test atmosphere matching the C++ benchmark.

    Args:
        source: "thermal" (Planck only), "solar" (beam only),
                "mixed" (thermal + solar beam).
    """
    nlev = nlay + 1
    g = 0.7

    pmom = np.zeros((nlay, nmom))
    for l in range(nlay):
        for m in range(nmom):
            pmom[l, m] = g ** m

    delta_tau = np.zeros((nwav, nlay))
    ssa = np.zeros((nwav, nlay))
    planck = np.zeros((nwav, nlev))

    for w in range(nwav):
        wfrac = w / nwav
        for l in range(nlay):
            lfrac = l / nlay
            delta_tau[w, l] = 0.05 + 0.5 * lfrac + 0.1 * wfrac
            ssa[w, l] = 0.7 + 0.2 * (1.0 - lfrac)
        if source in ("thermal", "mixed"):
            for l in range(nlev):
                lfrac = l / nlay
                T = 200.0 + 100.0 * lfrac
                planck[w, l] = T * T * (1.0 + 0.5 * wfrac)

    atm = {
        "nlay": nlay, "nwav": nwav, "nmu": nmu, "nmom": nmom,
        "delta_tau": delta_tau, "ssa": ssa, "pmom": pmom, "planck": planck,
        "source": source,
    }

    if source in ("solar", "mixed"):
        atm["solar_flux"] = 1.0
        atm["solar_mu"] = 0.5

    return atm


# ============================================================================
#  JAX batched benchmark
# ============================================================================

def benchmark_jax_batched(atm, nruns):
    """Benchmark batched JAX solver, returns time in ms."""
    import jax

    bcfg = BatchConfig()
    bcfg.num_wavenumbers = atm["nwav"]
    bcfg.num_layers = atm["nlay"]
    bcfg.num_quadrature = atm["nmu"]
    bcfg.num_moments_max = atm["nmom"]
    bcfg.surface_albedo = 0.1

    if "solar_flux" in atm:
        bcfg.solar_flux = atm["solar_flux"]
        bcfg.solar_mu = atm["solar_mu"]

    delta_tau = atm["delta_tau"]
    ssa = atm["ssa"]
    pmom = atm["pmom"]
    planck = atm["planck"]

    # Warmup (JIT compilation)
    flux_up, flux_down = solve_batch(bcfg, delta_tau, ssa, pmom, planck)
    # Block until done
    flux_up.block_until_ready()
    flux_down.block_until_ready()

    total = 0.0
    t0 = time.perf_counter()

    for _ in range(nruns):
        flux_up, flux_down = solve_batch(bcfg, delta_tau, ssa, pmom, planck)
        flux_up.block_until_ready()
        total += float(flux_up[0])

    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0 / nruns

    if total == -999.0:
        print(total)

    return ms


# ============================================================================
#  Run compiled C++ benchmark binary
# ============================================================================

def run_cpp_benchmark(build_dir):
    """Run the compiled C++ benchmark and parse results."""
    benchmark_path = os.path.join(build_dir, "ad_cuda_benchmark")
    if not os.path.exists(benchmark_path):
        return None

    try:
        result = subprocess.run(
            [benchmark_path],
            capture_output=True, text=True, timeout=600
        )
        return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def parse_cpp_benchmark_output(output):
    """Parse the C++ benchmark output into a dict of {label: {cpu_ms, cuda_ms}}."""
    results = {}
    if not output:
        return results

    for line in output.split("\n"):
        line = line.strip()
        if not line or line.startswith("Adding") or line.startswith("===") or line.startswith("GPU"):
            continue
        if line.startswith("Config") or line.startswith("-"):
            continue
        if line.startswith("*"):
            continue

        parts = line.rsplit(None, 3)
        if len(parts) >= 3:
            try:
                label = line[:46].strip()
                tokens = line[46:].split()
                nums = []
                for t in tokens:
                    t = t.replace("*", "").replace("x", "")
                    try:
                        nums.append(float(t))
                    except ValueError:
                        pass
                if len(nums) >= 3:
                    results[label] = {"cpu_ms": nums[0], "cuda_ms": nums[1], "speedup": nums[2]}
            except (ValueError, IndexError):
                pass

    return results


# ============================================================================
#  Main benchmark
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark C++ CPU vs CUDA vs JAX")
    parser.add_argument("--no-cpp", action="store_true",
                        help="Skip C++ / CUDA benchmark")
    parser.add_argument("--build-dir", default="build",
                        help="Path to CMake build directory (default: build)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    build_dir = os.path.join(project_dir, args.build_dir)

    print("Adding-Doubling Solver Benchmark: C++ CPU vs CUDA vs JAX (batched)")
    print("=" * 66)

    try:
        import jax
        print(f"JAX precision: {'float64' if jax.config.jax_enable_x64 else 'float32'}")
        devices = jax.devices()
        for d in devices:
            if d.platform == "gpu":
                print(f"JAX GPU: {d.device_kind}")
                break
        else:
            print("JAX device: CPU")
    except Exception:
        pass

    print()

    configs = [
        # --- Original thermal-only configs ---
        {"nlay": 10,  "nwav": 100,   "nmu": 8,  "nmom": 16, "source": "thermal",
         "jax_runs": 5, "label": "Small  (10 layers, 100 wn, N=8)"},
        {"nlay": 50,  "nwav": 1000,  "nmu": 8,  "nmom": 16, "source": "thermal",
         "jax_runs": 3, "label": "Medium (50 layers, 1000 wn, N=8)"},
        {"nlay": 100, "nwav": 20000, "nmu": 8,  "nmom": 16, "source": "thermal",
         "jax_runs": 1, "label": "Large  (100 layers, 20000 wn, N=8)"},
        {"nlay": 100, "nwav": 20000, "nmu": 4,  "nmom": 8,  "source": "thermal",
         "jax_runs": 1, "label": "Large  (100 layers, 20000 wn, N=4)"},
        {"nlay": 100, "nwav": 20000, "nmu": 16, "nmom": 32, "source": "thermal",
         "jax_runs": 1, "label": "Large  (100 layers, 20000 wn, N=16)"},

        # --- New source-type configs ---
        {"nlay": 100, "nwav": 20000, "nmu": 8,  "nmom": 16, "source": "thermal",
         "jax_runs": 1, "label": "Large  thermal  (100 lay, 20k wn, N=8)"},
        {"nlay": 100, "nwav": 20000, "nmu": 8,  "nmom": 16, "source": "solar",
         "jax_runs": 1, "label": "Large  solar    (100 lay, 20k wn, N=8)"},
        {"nlay": 100, "nwav": 20000, "nmu": 8,  "nmom": 16, "source": "mixed",
         "jax_runs": 1, "label": "Large  mixed    (100 lay, 20k wn, N=8)"},
    ]

    # ---- Run C++ / CUDA benchmark ----
    cpp_results = {}
    if not args.no_cpp:
        print("Running C++ / CUDA benchmark...")
        output = run_cpp_benchmark(build_dir)
        if output:
            cpp_results = parse_cpp_benchmark_output(output)
            if cpp_results:
                print("  Done (parsed compiled benchmark output)")
            else:
                print("  Warning: could not parse output. Raw:")
                print(output)
        else:
            print(f"  Binary not found at {build_dir}/ad_cuda_benchmark")
            print("  Build: cmake -B build -DADRT_ENABLE_CUDA=ON && cmake --build build")
    print()

    # ---- Run JAX benchmark ----
    print("Running JAX batched benchmark...")
    jax_results = {}

    for c in configs:
        label = c["label"]
        nwav_actual = c["nwav"]
        jax_runs = c["jax_runs"]
        source = c.get("source", "thermal")

        if nwav_actual > 1000:
            nwav_bench = 1000
            atm = build_atmosphere(c["nlay"], nwav_bench, c["nmu"], c["nmom"], source)
            ms = benchmark_jax_batched(atm, 1)
            jax_ms = ms * (nwav_actual / nwav_bench)
            extrapolated = True
        else:
            atm = build_atmosphere(c["nlay"], nwav_actual, c["nmu"], c["nmom"], source)
            jax_ms = benchmark_jax_batched(atm, jax_runs)
            extrapolated = False

        jax_results[label] = {"ms": jax_ms, "extrapolated": extrapolated}
        marker = " *" if extrapolated else ""
        print(f"  {label}: {jax_ms:10.2f} ms{marker}")

    print()

    # ---- Combined results table ----
    print()
    print("Results")
    print("=" * 108)
    header = (f"{'Configuration':<46} {'C++ CPU (ms)':>13} {'CUDA (ms)':>11} "
              f"{'JAX (ms)':>11} {'CPU/CUDA':>9} {'CPU/JAX':>9}")
    print(header)
    print("-" * 108)

    for c in configs:
        label = c["label"]

        cpp_ms = None
        cuda_ms = None

        if label in cpp_results:
            cpp_ms = cpp_results[label]["cpu_ms"]
            cuda_ms = cpp_results[label]["cuda_ms"]
            cpp_ms_str = f"{cpp_ms:10.2f}" + (" *" if label.startswith("Large") else "  ")
            cuda_ms_str = f"{cuda_ms:9.2f}  "
            cuda_speedup_str = f"{cpp_ms / cuda_ms:7.1f}x "
        else:
            cpp_ms_str = "       N/A  "
            cuda_ms_str = "      N/A  "
            cuda_speedup_str = "     N/A "

        jax_ms = jax_results.get(label, {}).get("ms")
        jax_extrap = jax_results.get(label, {}).get("extrapolated", False)

        if jax_ms is not None:
            jax_ms_str = f"{jax_ms:9.2f}" + (" *" if jax_extrap else "  ")
            if cpp_ms is not None and cpp_ms > 0:
                jax_speedup_str = f"{cpp_ms / jax_ms:7.1f}x "
            else:
                jax_speedup_str = "     N/A "
        else:
            jax_ms_str = "      N/A  "
            jax_speedup_str = "     N/A "

        print(f"{label:<46} {cpp_ms_str}{cuda_ms_str}{jax_ms_str}{cuda_speedup_str}{jax_speedup_str}")

    print()
    print("* = time extrapolated from smaller subset")
    print("CPU/CUDA, CPU/JAX = speedup relative to C++ CPU (higher is faster)")
    print()

    if not cpp_results:
        print("Note: C++/CUDA results unavailable. Build the project to include them:")
        print("  cmake -B build -DADRT_ENABLE_CUDA=ON && cmake --build build")


if __name__ == "__main__":
    main()
