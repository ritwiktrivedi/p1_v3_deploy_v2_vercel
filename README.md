# TDS Course Content and Discourse Forum RAG Application

This is a RAG application for selected content from the Tools in Data Science Course at IIT Madras.

## Step 1 (Clone Repo of Course Content)

Code to clone Jan 2025 Repo of TDS official course content:

```BASH
gh repo clone sanand0/tools-in-data-science-public -- --branch tds-2025-01 --single-branch
```

## Step 2 (Run html_to_md.py)

```BASH
uv run html_to_md.py
```

## Step 3 (Run discourse_extractor.py)

```BASH
uv run discourse_extractor.py
```

## Step 4 (Create embedding.npz file)

```BASH
uv run embed.py
```

## Step 5 (After steps 1 to 4 or directly as npz file is already in this repo, run answer.py)

```BASH
uv run answer.py
```

# Note:

The first question in the testing file is outside the timeline.

Can still test it locally though using:

```BASH
curl "http://0.0.0.0:8000/api/" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Should I use gpt-4o-mini which AI proxy supports, or gpt3.5 turbo?\", \"image\": \"$(base64 -w0 project-tds-virtual-ta-q1.webp)\"}"
```

More relevant question for testing within 1st Jan 2025 and 15th April 2025:

```BASH
curl "http://0.0.0.0:8000/api/" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"The License file is present in the github repository however i received a mail that said that it was absent. I thought that the ‘LICENSE’ file had to be renamed to ‘MIT LICENSE’.\", \"image\": \"$(base64 -w0 image.png)\"}"
```

# Debugging Endpoints

## Test basic health

curl "http://0.0.0.0:8000/health"

## Test embedding functionality

curl "http://0.0.0.0:8000/test-embedding"

## Check quota status

curl "http://0.0.0.0:8000/quota-status"
