#!/usr/bin/env python3
"""
llama-server benchmark tool (Zero-Dependency Async Implementation).
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
import time

STAGGER_MAX_DELAY = 3.0

def get_payload(model_name, budget):
    return json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": "Explain the technical difference between Monolithic and Microservices architecture. Discuss trade-offs in data consistency and network latency."}],
        "thinking_budget_tokens": budget,
        "temperature": 0.8,
        "max_tokens": 512,
    })

async def health_check(host, port):
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=5)
        header = f"GET /health HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n"
        writer.write(header.encode())
        await writer.drain()
        resp = await reader.read(1024)
        writer.close()
        await writer.wait_closed()
        return b"200 OK" in resp
    except:
        return False

async def async_post(url, model_name, budget):
    from urllib.parse import urlparse
    u = urlparse(url)
    host = u.hostname
    port = u.port or (443 if u.scheme == 'https' else 80)
    path = u.path + "/v1/chat/completions"
    
    payload = get_payload(model_name, budget)
    content_length = len(payload.encode('utf-8'))
    
    header = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {content_length}\r\n"
        f"Connection: close\r\n\r\n"
    )
    
    reader, writer = await asyncio.open_connection(host, port)
    writer.write(header.encode('utf-8'))
    writer.write(payload.encode('utf-8'))
    await writer.drain()
    
    response = b""
    while True:
        data = await reader.read(8192)
        if not data:
            break
        response += data
    writer.close()
    await writer.wait_closed()
    
    try:
        resp_str = response.decode('utf-8', errors='ignore')
        header_part, body = resp_str.split("\r\n\r\n", 1)
        
        if "200 OK" not in header_part:
            raise Exception(f"HTTP Error: {header_part.splitlines()[0]}")
            
        result = json.loads(body)
        t = result.get("timings", {})
        return {
            "pp_speed": t.get("prompt_per_second", 0),
            "pp_tokens": t.get("prompt_n", 0),
            "gen_speed": t.get("predicted_per_second", 0),
            "gen_tokens": t.get("predicted_n", 0),
        }
    except Exception as e:
        raise Exception(f"Failed to parse response: {e}")

async def run_single(url, label, model_name, budget, delay=0.0):
    if delay > 0:
        await asyncio.sleep(delay)
    
    print(f"  [>] {label}: Started...")
    start_t = time.perf_counter()
    try:
        res = await async_post(url, model_name, budget)
        elapsed = time.perf_counter() - start_t
        print(f"  [+] {label}: Completed in {elapsed:.1f}s ({res['gen_speed']:.1f} tok/s)")
        return res
    except Exception as e:
        print(f"  [-] {label}: Failed - {e}")
        return None

def print_summary(results, wall_time=None):
    if not results: return
    gen_speeds = [r["gen_speed"] for r in results if r]
    pp_speeds = [r["pp_speed"] for r in results if r]
    if not gen_speeds: return

    print("\n" + "="*50)
    print(f"STATISTICS ({len(results)} requests)")
    print("-" * 50)
    print(f"Generation  : avg {statistics.mean(gen_speeds):>6.1f} | min {min(gen_speeds):>5.1f} | max {max(gen_speeds):>5.1f} tok/s")
    if len(gen_speeds) > 1:
        print(f"Std Dev     : {statistics.stdev(gen_speeds):>6.1f} tok/s")
    print(f"Prompt Eval : avg {statistics.mean(pp_speeds):>6.1f} tok/s")
    
    if wall_time:
        total_tokens = sum(r["gen_tokens"] for r in results if r)
        throughput = total_tokens / wall_time
        print(f"Wall Time   : {wall_time:>6.2f}s")
        print(f"Throughput  : {throughput:>6.1f} tok/s (aggregate)")
    print("="*50 + "\n")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("count", nargs="?", type=int, default=1)
    parser.add_argument("-p", "--parallel", type=int, default=1)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", type=str, default="default")
    parser.add_argument("--budget", type=int, default=500)
    args = parser.parse_args()

    if not await health_check("localhost", args.port):
        print(f"Error: Server not responding at http://localhost:{args.port}")
        sys.exit(1)

    url = f"http://localhost:{args.port}"
    print(f"Benchmarking: {url} (Model: {args.model}, Budget: {args.budget}) | {args.count} round(s) x {args.parallel} parallel\n")
    
    all_results = []
    for r in range(args.count):
        start_t = time.perf_counter()
        delays = [random.uniform(0, STAGGER_MAX_DELAY) if i > 0 else 0 for i in range(args.parallel)]
        tasks = [run_single(url, f"Slot {i+1}", args.model, args.budget, delays[i]) for i in range(args.parallel)]
        
        round_results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - start_t
        
        valid_results = [res for res in round_results if res]
        all_results.extend(valid_results)
        
        if args.parallel > 1:
            print_summary(valid_results, wall_time)

    if args.count > 1:
        print("OVERALL SUMMARY")
        print_summary(all_results)

if __name__ == "__main__":
    asyncio.run(main())
