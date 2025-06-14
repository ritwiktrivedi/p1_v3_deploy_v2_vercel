# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "argparse",
#     "fastapi",
#     "httpx",
#     "markdownify",
#     "numpy",
#     "semantic_text_splitter",
#     "tqdm",
#     "uvicorn",
#     "openai",
#     "pillow",
# ]
# ///

import argparse
import base64
import json
import numpy as np
import os
import re
import traceback
import time
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, CORSMiddleware
from pydantic import BaseModel
import httpx
from openai import OpenAI


app = FastAPI()

# should be restricted in production! But this is an academic project, so we allow all origins for evaluation purposes.
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RateLimiter:
    def __init__(self, requests_per_minute=50, requests_per_second=5):  # OpenAI has higher limits
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second
        self.request_times = []
        self.last_request_time = 0

    def wait_if_needed(self):
        current_time = time.time()

        # Per-second rate limiting
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            print(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        # Per-minute rate limiting
        current_time = time.time()
        self.request_times = [
            t for t in self.request_times if current_time - t < 60]

        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                print(f"Per-minute rate limit: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
                # Clean up old requests after sleeping
                current_time = time.time()
                self.request_times = [
                    t for t in self.request_times if current_time - t < 60]

        self.request_times.append(current_time)
        self.last_request_time = current_time


rate_limiter = RateLimiter(requests_per_minute=30, requests_per_second=3)


def get_openai_client():
    """Get OpenAI client with API key and base URL configuration"""
    api_key = os.getenv("OPENAI_API_KEY")
    # Optional: for proxies or alternative endpoints
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    # Support for custom base URLs (proxies, local models, etc.)
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    else:
        return OpenAI(api_key=api_key)


def get_image_description(image_base64):
    """Get a description of the image using OpenAI's vision model."""
    try:
        client = get_openai_client()

        response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-4-vision-preview"
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe the content of this image in detail, focusing on any text, objects, or relevant features that could help answer questions about it."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting image description: {e}")
        raise


def get_embedding(text: str, max_retries: int = 3) -> list[float]:
    """Get embedding for text chunk with rate limiting and retry logic"""

    # Check if API key is available
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    print(
        f"Getting embedding for text: '{text[:100]}...' (length: {len(text)})")

    client = get_openai_client()
    last_error = None

    for attempt in range(max_retries):
        try:
            print(
                f"Attempting to get embedding (attempt {attempt + 1}/{max_retries})")

            # Apply rate limiting
            rate_limiter.wait_if_needed()

            print(f"Making API call to text-embedding-3-small...")
            response = client.embeddings.create(
                model="text-embedding-3-small",  # or "text-embedding-3-large"
                input=text
            )

            print(f"API call successful, checking response...")
            if not response.data or len(response.data) == 0:
                raise ValueError("Empty embedding response")

            embedding = response.data[0].embedding
            print(
                f"Successfully got embedding with {len(embedding)} dimensions")
            return embedding

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            error_type = type(e).__name__
            print(f"Attempt {attempt + 1} failed with {error_type}: {e}")
            print(f"Full error details: {traceback.format_exc()}")

            if "rate limit" in error_str or "quota" in error_str or "429" in error_str:
                wait_time = 2 ** attempt * 5  # Exponential backoff
                print(f"Rate limit detected, waiting {wait_time} seconds...")
                time.sleep(wait_time)
            elif "api key" in error_str or "authentication" in error_str or "401" in error_str or "unauthorized" in error_str:
                print("Authentication error detected - check your API key")
                raise ValueError(f"Invalid or missing API key: {e}")
            elif "model" in error_str or "404" in error_str or "not found" in error_str:
                print("Model not found error detected")
                raise ValueError(
                    f"text-embedding-3-small model not available: {e}")
            elif "400" in error_str or "bad request" in error_str:
                print("Bad request error - possibly text too long")
                raise ValueError(f"Invalid request to embedding API: {e}")
            elif "500" in error_str or "503" in error_str:
                print("Server error - will retry")
                wait_time = 2 ** attempt * 2
                print(
                    f"Server error, waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            elif attempt == max_retries - 1:
                print(f"Final attempt failed after {max_retries} attempts")
                break
            else:
                # General retry with shorter wait
                wait_time = 2 ** attempt
                print(
                    f"General error, waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

    # If we get here, all retries failed
    error_msg = f"Max retries ({max_retries}) exceeded for embedding generation. Last error: {last_error}"
    print(error_msg)
    raise Exception(error_msg)


def load_embeddings():
    """Load chunks and embeddings from npz file"""
    try:
        if not os.path.exists("embeddings.npz"):
            raise FileNotFoundError(
                "embeddings.npz file not found. Please generate embeddings first.")

        data = np.load("embeddings.npz", allow_pickle=True)
        chunks = data["chunks"]
        embeddings = data["embeddings"]

        print(f"Loaded {len(chunks)} chunks and {len(embeddings)} embeddings")
        return chunks, embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        raise


def extract_discourse_urls(text):
    """Extract discourse URLs from text"""
    # Multiple patterns to catch different URL formats
    patterns = [
        # Standard format
        r'https://discourse\.onlinedegree\.iitm\.ac\.in/t/[^\s<>"\'\)\],]*',
        # Any discourse URL
        r'https://discourse\.onlinedegree\.iitm\.ac\.in/[^\s<>"\'\)\],]*',
        # Without https
        r'discourse\.onlinedegree\.iitm\.ac\.in/t/[^\s<>"\'\)\],]*',
        # Without https, any path
        r'discourse\.onlinedegree\.iitm\.ac\.in/[^\s<>"\'\)\],]*',
    ]

    urls = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Ensure URL starts with https
            if not match.startswith('https://'):
                match = 'https://' + match
            # Clean up any trailing punctuation
            match = re.sub(r'[.,;!?]+$', '', match)
            urls.append(match)

    # Remove duplicates and return
    return list(set(urls))


def generate_discourse_response(question: str, context: str, chunks: list):
    """Generate a response in discourse format with answer and links."""
    try:
        client = get_openai_client()

        # Collect all discourse URLs from chunks
        all_urls = []
        for chunk in chunks:
            urls = extract_discourse_urls(chunk)
            all_urls.extend(urls)

        all_urls = list(set(all_urls))  # Remove duplicates

        # Debug: Print what URLs we found
        print(f"Found discourse URLs: {all_urls}")
        print(f"Sample chunks (first 200 chars each):")
        for i, chunk in enumerate(chunks[:3]):
            print(f"  Chunk {i}: {chunk[:200]}...")

        system_prompt = f"""You are a knowledgeable teaching assistant helping with discourse forum questions.

Your task is to:
1. Provide a clear, concise answer to the question using the context provided
2. Use ONLY the actual discourse URLs found in the context (listed below)

ACTUAL DISCOURSE URLs FOUND: {all_urls}

Format your response as JSON with this exact structure:
{{
  "answer": "Your clear, direct answer here",
  "links": [
    {{
      "url": "EXACT URL from the list above",
      "text": "brief description of what this link contains"
    }}
  ]
}}

CRITICAL RULES:
- ONLY use URLs from this exact list: {all_urls}
- DO NOT make up or generate any URLs
- If the list is empty, use an empty links array: []
- Each URL must be copied exactly from the provided list
- Keep the answer concise but informative
- If the context doesn't contain enough information, say "I don't have enough information to answer this question"
"""

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ],
            max_tokens=800,
            temperature=0.1,  # Lower temperature for more consistent output
            top_p=0.95
        )

        response_text = response.choices[0].message.content
        print(f"LLM Response: {response_text}")

        # Try to parse as JSON
        try:
            parsed_response = json.loads(response_text)

            # Validate that URLs in response are from our actual list
            if "links" in parsed_response:
                valid_links = []
                for link in parsed_response["links"]:
                    if link["url"] in all_urls:
                        valid_links.append(link)
                    else:
                        print(
                            f"Warning: Filtered out invalid URL: {link['url']}")
                parsed_response["links"] = valid_links

            return parsed_response

        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            # If JSON parsing fails, create a structured response with actual URLs
            links = []
            for url in all_urls[:5]:  # Limit to 5 links
                links.append({
                    "url": url,
                    "text": "Related discussion"
                })

            return {
                "answer": response_text,
                "links": links
            }

    except Exception as e:
        print(f"Error generating discourse response: {e}")
        raise


def answer(question: str, image: str = None):
    try:
        # Load embeddings
        loaded_chunks, loaded_embeddings = load_embeddings()

        # Process image if provided
        if image:
            try:
                image_description = get_image_description(image)
                question += f" {image_description}"
            except Exception as e:
                print(f"Warning: Could not process image: {e}")
                # Continue without image description

        # Get the embedding for the question
        question_embedding = get_embedding(question)

        # Convert to numpy array if needed
        question_embedding = np.array(question_embedding)

        # Calculate cosine similarity
        similarities = np.dot(loaded_embeddings, question_embedding) / (
            np.linalg.norm(loaded_embeddings, axis=1) *
            np.linalg.norm(question_embedding)
        )

        # Get the index of the 10 most similar chunks
        top_indices = np.argsort(similarities)[-10:][::-1]

        # Get the top chunks
        top_chunks = [loaded_chunks[i] for i in top_indices]

        # Generate discourse-formatted response
        discourse_response = generate_discourse_response(
            question, "\n".join(top_chunks), top_chunks)

        # Return in the requested format
        return {
            "question": question,
            "discourse_response": discourse_response,
            "top_chunks": top_chunks,
            "similarities": similarities[top_indices].tolist()
        }

    except Exception as e:
        print(f"Error in answer function: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise


@app.post("/api/")
async def api_answer(request: Request):
    try:
        data = await request.json()
        print(f"Received request: {data}")

        question = data.get("question")
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")

        result = answer(question, data.get("image"))

        # Return just the discourse response format
        return result["discourse_response"]

    except ValueError as e:
        print(f"ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/detailed")
async def api_answer_detailed(request: Request):
    """Return detailed response including chunks and similarities"""
    try:
        data = await request.json()
        print(f"Received detailed request: {data}")

        question = data.get("question")
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")

        result = answer(question, data.get("image"))
        return result

    except ValueError as e:
        print(f"ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/quota-status")
async def check_quota_status():
    """Check if API is currently experiencing quota issues"""
    try:
        client = get_openai_client()

        # Try a minimal request
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="test"
        )

        return {"status": "ok", "message": "API quota available"}

    except Exception as e:
        error_str = str(e).lower()
        if "429" in error_str or "quota" in error_str or "rate limit" in error_str:
            return {
                "status": "quota_exhausted",
                "message": "API quota exceeded. Try again later or check your quota limits.",
                "error": str(e)
            }
        else:
            return {"status": "error", "message": str(e)}


@app.get("/test-embedding")
async def test_embedding():
    """Test endpoint to debug embedding API"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"status": "error", "message": "OPENAI_API_KEY not set"}

        print("Testing embedding API...")

        # Check quota first
        quota_check = await check_quota_status()
        if quota_check["status"] == "quota_exhausted":
            return quota_check

        client = get_openai_client()

        # Test with simple text
        test_text = "This is a test"
        print(f"Testing with text: '{test_text}'")

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=test_text
        )

        if response.data and len(response.data) > 0:
            embedding = response.data[0].embedding
            return {
                "status": "success",
                "message": f"Embedding generated successfully with {len(embedding)} dimensions",
                "sample_values": embedding[:5]  # First 5 values
            }
        else:
            return {"status": "error", "message": "Empty embedding response"}

    except Exception as e:
        print(f"Test embedding error: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        error_str = str(e).lower()
        if "429" in error_str or "quota" in error_str or "rate limit" in error_str:
            return {
                "status": "quota_exhausted",
                "message": "API quota exceeded. Wait for quota reset or check your limits.",
                "error": str(e)
            }
        return {"status": "error", "message": str(e), "type": type(e).__name__}


@app.get("/debug/chunks")
async def debug_chunks(q: str = "test", limit: int = 5):
    """Debug endpoint to see what chunks contain discourse URLs"""
    try:
        loaded_chunks, loaded_embeddings = load_embeddings()

        # Get top chunks for the query
        question_embedding = get_embedding(q)
        question_embedding = np.array(question_embedding)

        similarities = np.dot(loaded_embeddings, question_embedding) / (
            np.linalg.norm(loaded_embeddings, axis=1) *
            np.linalg.norm(question_embedding)
        )

        top_indices = np.argsort(similarities)[-limit:][::-1]
        top_chunks = [loaded_chunks[i] for i in top_indices]

        # Analyze chunks for URLs
        chunk_analysis = []
        for i, chunk in enumerate(top_chunks):
            urls = extract_discourse_urls(chunk)
            chunk_analysis.append({
                "chunk_index": i,
                "similarity": float(similarities[top_indices[i]]),
                "preview": chunk[:300] + "..." if len(chunk) > 300 else chunk,
                "urls_found": urls,
                "contains_discourse_url": len(urls) > 0
            })

        # Overall stats
        all_chunks_with_urls = []
        for chunk in loaded_chunks:
            urls = extract_discourse_urls(chunk)
            if urls:
                all_chunks_with_urls.extend(urls)

        return {
            "query": q,
            "total_chunks": len(loaded_chunks),
            "total_unique_discourse_urls": len(set(all_chunks_with_urls)),
            "sample_urls": list(set(all_chunks_with_urls))[:10],
            "top_chunks_analysis": chunk_analysis
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if embeddings file exists
        if not os.path.exists("embeddings.npz"):
            return {"status": "unhealthy", "reason": "embeddings.npz not found"}

        # Check if API key is set
        if not os.getenv("OPENAI_API_KEY"):
            return {"status": "unhealthy", "reason": "OPENAI_API_KEY not set"}

        # Check embeddings file details
        data = np.load("embeddings.npz", allow_pickle=True)
        chunks = data["chunks"]
        embeddings = data["embeddings"]

        return {
            "status": "healthy",
            "embeddings_count": len(chunks),
            "embedding_dimensions": len(embeddings[0]) if len(embeddings) > 0 else 0,
            "openai_base_url": os.getenv("OPENAI_BASE_URL", "default"),
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        }
    except Exception as e:
        return {"status": "unhealthy", "reason": str(e)}


if __name__ == "__main__":
    import uvicorn

    # Check prerequisites before starting
    if not os.path.exists("embeddings.npz"):
        print("WARNING: embeddings.npz file not found!")

    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable not set!")

    # Print configuration
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    print(f"Configuration:")
    print(f"  Model: {model}")
    print(f"  Base URL: {base_url if base_url else 'OpenAI default'}")
    print(f"  Embedding Model: text-embedding-3-small")

    uvicorn.run(app, host="0.0.0.0", port=8000)
