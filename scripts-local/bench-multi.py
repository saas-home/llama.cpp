#!/usr/bin/env python3
"""
Multi-port llama-server benchmark tool.
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import URLError

STAGGER_MAX_DELAY = 1.0

def get_payload():
    return json.dumps({
        "messages": [{"role": "user", "content": "Explain the technical difference between Monolithic and Microservices architecture. Discuss trade-offs in data consistency and network latency."}],
        "temperature": 0.8,
        "max_tokens": 128,
    }).encode()

def health_check(url):
    try:
        with urlopen(f"{url}/health", timeout=5) as resp:
            return json.loads(resp.read()).get("status") in ["ok", "loading", "error"] # loading is fine for health check if we just want to see it alive
    except:
        return False

def do_request(url, label):
    req = Request(f"{url}/v1/chat/completions", data=get_payload(), headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())

    t = result.get("timings", {})
    return {
        "label": label,
        "pp_speed": t.get("prompt_per_second", 0),
        "pp_tokens": t.get("prompt_n", 0),
        "gen_speed": t.get("predicted_per_second", 0),
        "gen_tokens": t.get("predicted_n", 0),
    }

async def run_single(url, label, delay=0.0):
    if delay > 0:
        await asyncio.sleep(delay)
    
    start_t = time.perf_counter()
    try:
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, do_request, url, label)
        elapsed = time.perf_counter() - start_t
        return res
    except Exception as e:
        print(f"  [-] {label} on {url}: Failed - {e}")
        return None

def print_summary(results, wall_time=None):
    if not results: return
    gen_speeds = [r["gen_speed"] for r in results if r]
    pp_speeds = [r["pp_speed"] for r in results if r]
    if not gen_speeds: return

    print("-" * 50)
    print(f"Generation  : avg {statistics.mean(gen_speeds):>6.1f} tok/s")
    print(f"Prompt Eval : avg {statistics.mean(pp_speeds):>6.1f} tok/s")
    
    if wall_time:
        total_tokens = sum(r["gen_tokens"] for r in results if r)
        throughput = total_tokens / wall_time
        print(f"Wall Time   : {wall_time:>6.2f}s")
        print(f"Throughput  : {throughput:>6.1f} tok/s (aggregate)")
    print("-" * 50)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", type=str, default="8080", help="Comma separated ports")
    parser.add_argument("-p", "--parallel", type=int, default=1, help="Parallel requests per port")
    args = parser.parse_args()

    ports = args.ports.split(",")
    urls = [f"http://localhost:{p}" for p in ports]

    for url in urls:
        if not health_check(url):
            print(f"Error: Server not responding at {url}")
            sys.exit(1)

    print(f"Benchmarking ports: {ports} with {args.parallel} parallel reqs each\n")
    
    start_t = time.perf_counter()
    tasks = []
    for url in urls:
        for i in range(args.parallel):
            tasks.append(run_single(url, f"Port {url.split(':')[-1]} req {i+1}"))
    
    results = await asyncio.gather(*tasks)
    wall_time = time.perf_counter() - start_t
    
    valid_results = [res for res in results if res]
    
    # Split results by port
    for url in urls:
        port = url.split(":")[-1]
        port_results = [r for r in valid_results if f"Port {port}" in r["label"]]
        print(f"\nRESULTS FOR PORT {port}:")
        print_summary(port_results)

    if len(urls) > 1:
        print("\nAGGREGATE RESULTS (CONCURRENT):")
        print_summary(valid_results, wall_time)

if __name__ == "__main__":
    asyncio.run(main())
