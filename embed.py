# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "semantic-text-splitter",
#   "numpy",
#   "tqdm",
#   "openai",
# ]
# ///

import hashlib
import httpx
import json
import numpy as np
import os
import time
from pathlib import Path
from semantic_text_splitter import MarkdownSplitter
from tqdm import tqdm
from openai import OpenAI


class RateLimiter:
    def __init__(self, requests_per_minute=500, requests_per_second=10):  # OpenAI has higher limits
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second
        self.request_times = []
        self.last_request_time = 0

    def wait_if_needed(self):
        current_time = time.time()

        # Per-second rate limiting
        time_since_last = current_time - self.last_request_time
        if time_since_last < (1.0 / self.requests_per_second):
            sleep_time = (1.0 / self.requests_per_second) - time_since_last
            time.sleep(sleep_time)

        # Per-minute rate limiting
        current_time = time.time()
        self.request_times = [
            t for t in self.request_times if current_time - t < 60]

        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        # Clean up old requests after sleeping
        current_time = time.time()
        self.request_times = [
            t for t in self.request_times if current_time - t < 60]

        self.request_times.append(current_time)
        self.last_request_time = current_time


rate_limiter = RateLimiter(requests_per_minute=500, requests_per_second=10)


def get_openai_client():
    """Get OpenAI client with API key and base URL configuration"""
    api_key = os.getenv("OPENAI_API_KEY")
    # Optional: for proxies or alternative endpoints
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    # Support for custom base URLs (proxies, local models, etc.)
    if base_url:
        # Ensure base_url ends with /v1 for proper API routing
        if not base_url.endswith('/v1'):
            if base_url.endswith('/'):
                base_url = base_url + 'v1'
            else:
                base_url = base_url + '/v1'
        return OpenAI(api_key=api_key, base_url=base_url, timeout=30.0)
    else:
        return OpenAI(api_key=api_key, timeout=30.0)


def validate_embedding(embedding) -> bool:
    """Validate that embedding is properly formatted"""
    try:
        if not embedding:
            return False

        # Check if it's a list/array of numbers
        if not isinstance(embedding, (list, np.ndarray)):
            return False

        # Check if all elements are numbers
        for val in embedding:
            if not isinstance(val, (int, float, np.number)):
                return False
            # Check for invalid values
            if np.isnan(val) or np.isinf(val):
                return False

        # Check reasonable dimension size (OpenAI embeddings are typically 512-3072 dims)
        if len(embedding) < 100 or len(embedding) > 5000:
            return False

        return True
    except Exception:
        return False


def get_embedding(text: str, max_retries: int = 3) -> list[float]:
    """Get embedding for text chunk with rate limiting and retry logic"""

    client = get_openai_client()
    embedding_model = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    for attempt in range(max_retries):
        try:
            # Apply rate limiting
            rate_limiter.wait_if_needed()

            # Add some logging for debugging
            if attempt == 0:  # Only log on first attempt to avoid spam
                print(
                    f"Processing chunk {len(all_chunks) + 1}/{total_chunks}", end=" ", flush=True)

            # Correct embeddings API call
            response = client.embeddings.create(
                model=embedding_model,
                input=text,
                encoding_format="float"  # Explicitly specify float format
            )

            # Validate response structure
            if not hasattr(response, 'data') or not response.data:
                raise ValueError(
                    "API response missing 'data' field or data is empty")

            if len(response.data) == 0:
                raise ValueError("API response data array is empty")

            if not hasattr(response.data[0], 'embedding'):
                raise ValueError(
                    "API response missing 'embedding' field in data[0]")

            # Extract embedding from response
            embedding = response.data[0].embedding

            # Validate embedding format
            if not validate_embedding(embedding):
                raise ValueError(
                    f"Invalid embedding format: type={type(embedding)}, length={len(embedding) if hasattr(embedding, '__len__') else 'unknown'}")

            # Convert to list of floats to ensure consistency
            embedding = [float(x) for x in embedding]

            return embedding

        except Exception as e:
            error_str = str(e).lower()

            # Handle malformed response errors
            if any(keyword in error_str for keyword in ["missing", "invalid embedding", "data field", "empty"]):
                print(f"Malformed response on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    print(
                        f"Failed to get valid embedding after {max_retries} attempts due to malformed responses")
                    raise

            # Handle rate limiting
            elif "rate limit" in error_str or "quota" in error_str or "429" in error_str:
                # Exponential backoff for rate limit errors
                wait_time = 2 ** attempt * 5  # 5s, 10s, 20s
                print(f"Rate limit hit, waiting {wait_time} seconds ... ")
                time.sleep(wait_time)

            # Handle authentication errors
            elif "401" in error_str or "unauthorized" in error_str or "api key" in error_str:
                print(f"Authentication error: {e}")
                print("Please check your OPENAI_API_KEY environment variable")
                raise

            # Handle model errors
            elif "404" in error_str or "model" in error_str:
                print(f"Model error: {e}")
                print(
                    f"Please check if model '{embedding_model}' is available")
                raise

            # Handle invalid request errors
            elif "400" in error_str or "invalid" in error_str:
                print(f"Invalid request error: {e}")
                print(f"Text length: {len(text)} characters")
                if len(text) > 8191:  # OpenAI's token limit
                    print(
                        "Text might be too long. Consider splitting into smaller chunks.")
                raise

            # Handle JSON parsing errors (malformed response body)
            elif "json" in error_str or "decode" in error_str or "parse" in error_str:
                print(f"JSON parsing error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise ValueError(
                        f"Failed to parse API response after {max_retries} attempts")

            # Handle connection errors
            elif any(keyword in error_str for keyword in ["connection", "timeout", "network", "unreachable"]):
                print(f"Connection error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt * 2  # 2s, 4s, 8s
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise

            # Final attempt - raise the error
            elif attempt == max_retries - 1:
                print(
                    f"Failed to get embedding after {max_retries} attempts: {e}")
                raise

            # Retry with exponential backoff
            else:
                print(f"Attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff

    raise Exception("Max retries exceeded.")


def get_chunks(file_path: str, chunk_size: int = 1000):
    """Split markdown file into chunks"""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        if not content.strip():
            print(f"Warning: File {file_path} is empty")
            return []

        splitter = MarkdownSplitter(chunk_size)
        chunks = splitter.chunks(content)

        # Filter out empty chunks
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        return chunks
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []


if __name__ == "__main__":
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        exit(1)

    # Print configuration
    embedding_model = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    base_url = os.getenv("OPENAI_BASE_URL")

    print(f"Configuration:")
    print(f"  Embedding Model: {embedding_model}")
    print(f"  Base URL: {base_url if base_url else 'OpenAI default'}")
    print(f"  Chunk Size: 1000")
    print()

    # Check if combined_markdowns directory exists
    if not Path("combined_markdowns").exists():
        print("ERROR: 'combined_markdowns' directory not found")
        print("Please create the directory and add your markdown files")
        exit(1)

    # Find all markdown files in the "combined_markdowns" directory and its subdirectories
    files = [*Path("combined_markdowns").glob("*.md"),
             *Path("combined_markdowns").rglob("*.md")]

    if not files:
        print("No markdown files found in 'combined_markdowns' directory")
        exit(1)

    print(f"Found {len(files)} markdown files")

    all_chunks = []
    all_embeddings = []
    total_chunks = 0
    file_chunks = {}

    # Get all chunks from all markdown files
    print("Analyzing files and creating chunks...")
    for file_path in files:
        try:
            chunks = get_chunks(file_path)
            if chunks:
                file_chunks[file_path] = chunks
                total_chunks += len(chunks)
                print(f"  {file_path.name}: {len(chunks)} chunks")
            else:
                print(f"  {file_path.name}: No valid chunks found")
        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")
            continue

    if total_chunks == 0:
        print("No chunks found to process!")
        exit(1)

    print(f"\nTotal chunks to process: {total_chunks}")
    print("Starting embedding generation...\n")

    # Test the API connection with a small sample first
    print("Testing API connection...")
    try:
        test_embedding = get_embedding("Hello, this is a test.")
        print(
            f"✓ API connection successful! Embedding dimension: {len(test_embedding)}")
    except Exception as e:
        print(f"✗ API connection failed: {e}")
        exit(1)

    # Process chunks and get embeddings
    successful_chunks = 0
    failed_chunks = 0
    skipped_chunks = []  # Track failed chunks for potential retry
    # Track actual chunks processed (not skipped empty ones)
    processed_chunks = 0

    with tqdm(total=total_chunks, desc="Processing embeddings") as pbar:
        for file_path, chunks in file_chunks.items():
            for chunk_idx, chunk in enumerate(chunks):
                # Skip empty chunks but don't count them in progress
                if not chunk.strip():
                    print(
                        f"\nSkipping empty chunk {chunk_idx+1} from {file_path.name}")
                    # Update total to reflect actual chunks to process
                    pbar.total = pbar.total - 1 if pbar.total > 0 else 0
                    continue

                processed_chunks += 1

                try:
                    pbar.set_description(
                        f"Processing chunk {processed_chunks}")

                    embedding = get_embedding(chunk)
                    all_chunks.append(chunk)
                    all_embeddings.append(embedding)
                    successful_chunks += 1

                    pbar.set_postfix({
                        "file": file_path.name[:20],
                        "success": successful_chunks,
                        "failed": failed_chunks,
                        "dims": len(embedding),
                        "processed": processed_chunks
                    })
                    pbar.update(1)

                except Exception as e:
                    failed_chunks += 1
                    skipped_chunks.append({
                        'file': file_path,
                        'chunk_idx': chunk_idx,
                        'chunk': chunk,
                        'error': str(e)
                    })
                    print(
                        f"\nSkipping chunk {chunk_idx+1} from {file_path.name} due to error: {e}")
                    pbar.set_postfix({
                        "file": file_path.name[:20],
                        "success": successful_chunks,
                        "failed": failed_chunks,
                        "processed": processed_chunks
                    })
                    pbar.update(1)
                    continue

    print(f"\nProcessing complete!")
    print(f"  Total chunks found: {total_chunks}")
    print(f"  Chunks actually processed: {processed_chunks}")
    print(f"  Empty chunks skipped: {total_chunks - processed_chunks}")
    print(f"  Successful embeddings: {successful_chunks}")
    print(f"  Failed embeddings: {failed_chunks}")

    # Verify math
    if processed_chunks != (successful_chunks + failed_chunks):
        print(
            f"  ⚠️  WARNING: Math doesn't add up! processed={processed_chunks}, success+failed={successful_chunks + failed_chunks}")

    # Show success rate based on actual processed chunks
    if processed_chunks > 0:
        success_rate = (successful_chunks / processed_chunks) * 100
        print(f"  Success rate: {success_rate:.1f}% (of processed chunks)")

    if successful_chunks > 0:
        # Save embeddings
        print(f"\nSaving embeddings to 'embeddings.npz'...")
        np.savez_compressed("embeddings.npz",
                            chunks=all_chunks,
                            embeddings=all_embeddings)
        print(
            f"Saved {len(all_chunks)} chunks and {len(all_embeddings)} embeddings")

        # Print embedding info
        if all_embeddings:
            embedding_dims = len(all_embeddings[0])
            print(f"Embedding dimensions: {embedding_dims}")

            # Estimate file size
            file_size = os.path.getsize("embeddings.npz")
            print(f"File size: {file_size / (1024*1024):.2f} MB")

            # Print sample stats
            embeddings_array = np.array(all_embeddings)
            print(f"Embeddings shape: {embeddings_array.shape}")
            print(f"Embeddings dtype: {embeddings_array.dtype}")

        # Save failed chunks for potential retry
        if skipped_chunks:
            print(
                f"\nSaving {len(skipped_chunks)} failed chunks to 'failed_chunks.json' for potential retry...")
            with open('failed_chunks.json', 'w', encoding='utf-8') as f:
                # Convert Path objects to strings for JSON serialization
                serializable_chunks = []
                for chunk_info in skipped_chunks:
                    serializable_chunks.append({
                        'file': str(chunk_info['file']),
                        'chunk_idx': chunk_info['chunk_idx'],
                        'chunk': chunk_info['chunk'],
                        'error': chunk_info['error']
                    })
                json.dump(serializable_chunks, f, indent=2, ensure_ascii=False)
    else:
        print("No embeddings were successfully generated!")
        exit(1)

    # Test chunk processing (optional)
    test_file = "combined_markdowns/Contribute1.md"
    if os.path.exists(test_file):
        chunks = get_chunks(test_file, chunk_size=1000)
        print(f"\nTest: Total chunks in {test_file}: {len(chunks)}")
        if chunks:
            print(f"Sample chunk (first 200 chars): {chunks[0][:200]}...")
    else:
        print(f"\nTest file {test_file} not found, skipping test")
