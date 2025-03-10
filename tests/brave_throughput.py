import asyncio
import time
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from verifiers.tools.search import search_brave

async def search_brave_async(query, num_results, session=None):
    """Async wrapper for search_brave function"""
    def _search():
        return search_brave(query, num_results)
    
    # Run the synchronous function in a thread pool
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, _search)
    return result

async def test_throughput(target_rps=20, duration=30):
    """
    Test Brave search throughput with parallelism targeting a specific requests per second rate.
    
    Args:
        target_rps: Target requests per second (default: 20)
        duration: Test duration in seconds (default: 30)
    """
    query = "python programming"
    successful_requests = 0
    failed_requests = 0
    start_time = time.time()
    end_time = start_time + duration
    
    print(f"Starting Brave search throughput test targeting {target_rps} req/sec for {duration} seconds...")
    
    # Track when we hit rate limit
    rate_limited = False
    rate_limit_time = None
    
    tasks = []
    
    try:
        while time.time() < end_time and not rate_limited:
            # Calculate how many requests to launch in this batch
            batch_start = time.time()
            current_tasks = []
            
            # Launch a batch of requests to meet our target RPS
            for _ in range(target_rps):
                if time.time() >= end_time:
                    break
                    
                task = asyncio.create_task(search_brave_async(query, num_results=3))
                current_tasks.append(task)
                tasks.append(task)
            
            # Wait for the batch to complete
            batch_results = await asyncio.gather(*current_tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    failed_requests += 1
                    print(f"Error: {result}")
                elif "Error: HTTP error occurred: 429" in result:
                    failed_requests += 1
                    rate_limited = True
                    rate_limit_time = time.time() - start_time
                    print(f"\nHit rate limit after {rate_limit_time:.2f} seconds!")
                    break
                else:
                    successful_requests += 1
            
            # Calculate time to sleep to maintain target RPS
            batch_duration = time.time() - batch_start
            sleep_time = max(0, 1 - batch_duration)  # Aim for 1 second per batch
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                
            elapsed = time.time() - start_time
            print(f"Progress: {successful_requests} successful, {failed_requests} failed ({elapsed:.2f}s elapsed)")
            
    except KeyboardInterrupt:
        print("\nTest manually interrupted.")
    
    # Calculate and display results
    total_time = time.time() - start_time
    actual_rps = successful_requests / total_time if total_time > 0 else 0
    
    print("\n--- Results ---")
    print(f"Total successful requests: {successful_requests}")
    print(f"Total failed requests: {failed_requests}")
    print(f"Total time elapsed: {total_time:.2f} seconds")
    print(f"Average throughput: {actual_rps:.2f} requests/second")
    
    if rate_limited:
        print(f"Hit rate limit after {rate_limit_time:.2f} seconds")
        print(f"Sustainable rate appears to be below {target_rps} req/sec")
    else:
        print(f"No rate limiting detected at {target_rps} req/sec")

async def main():
    await test_throughput(target_rps=10, duration=30)

if __name__ == "__main__":
    asyncio.run(main())