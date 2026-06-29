#!/bin/bash
# ==============================================================================
# 7950X3D Optimization Script for Llama.cpp (Ubuntu 26.04)
# Goal: Automate CCD-specific EPP settings and memory throughput optimizations.
# ==============================================================================

# 1. Global Performance Governor
cpupower frequency-set -g performance

# 2. CCD0 Optimization (Cores 0-7 + SMT 16-23: V-Cache)
for i in {0..7} {16..23}; do
    echo "balance_performance" > /sys/devices/system/cpu/cpu$i/cpufreq/energy_performance_preference
done

# 3. CCD1 Optimization (Cores 8-15 + SMT 24-31: Frequency)
for i in {8..15} {24..31}; do
    echo "performance" > /sys/devices/system/cpu/cpu$i/cpufreq/energy_performance_preference
done

# 4. Memory & Kernel Optimizations
# Set Hugepages to 'always' for large KV Cache allocations
echo "always" > /sys/kernel/mm/transparent_hugepage/enabled
# Increase map count for massive MoE expert files
sysctl -w vm.max_map_count=1000000
# Disable NMI Watchdog to reduce jitter during inference
echo "0" > /proc/sys/kernel/nmi_watchdog

echo "🚀 7950X3D Optimized for Dual-Model Inference."
