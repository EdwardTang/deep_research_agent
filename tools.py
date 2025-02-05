"""
Tool implementations for interactive chat system.
Contains search, web scraping, file operations, package management tools, and text inspection.
"""

import asyncio
import logging
import os
import subprocess
import time
from multiprocessing import Pool
from typing import List, Optional, Union
from urllib.parse import urlparse
import base64
import json
import mimetypes
from io import BytesIO
import copy

import aiohttp
import html5lib
from duckduckgo_search import DDGS
from PIL import Image
from huggingface_hub import InferenceClient
import requests
import openai

from mdconvert import MarkdownConverter

logger = logging.getLogger(__name__)

# Search Engine Implementation
def search_with_retry(query: str, max_results: int = 10, max_retries: int = 3) -> List[dict]:
    """
    Search using DuckDuckGo and return results with URLs and text snippets.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        max_retries: Maximum number of retry attempts

    Returns:
        List of dictionaries containing search results
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Searching for query: {query} (attempt {attempt + 1}/{max_retries})")
            
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                
            if not results:
                logger.info("No results found")
                return []
            
            logger.info(f"Found {len(results)} results")
            return results
                
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info("Waiting 1 second before retry...")
                time.sleep(1)
            else:
                logger.error(f"All {max_retries} attempts failed")
                raise

def format_search_results(results: List[dict]) -> str:
    """
    Format search results into a readable string.

    Args:
        results: List of search result dictionaries

    Returns:
        Formatted string containing search results
    """
    output = []
    for i, result in enumerate(results, 1):
        output.append(f"\n=== Result {i} ===")
        output.append(f"URL: {result.get('href', 'N/A')}")
        output.append(f"Title: {result.get('title', 'N/A')}")
        output.append(f"Snippet: {result.get('body', 'N/A')}")
    return "\n".join(output)

# Web Scraper Implementation
async def fetch_page(url: str, session: Optional[aiohttp.ClientSession] = None) -> Optional[str]:
    """
    Asynchronously fetch a webpage's content.

    Args:
        url: URL to fetch
        session: Optional aiohttp session to use

    Returns:
        Page content as string if successful, None otherwise
    """
    async def _fetch(session: aiohttp.ClientSession) -> Optional[str]:
        try:
            logger.info(f"Fetching {url}")
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    logger.info(f"Successfully fetched {url}")
                    return content
                logger.error(f"Error fetching {url}: HTTP {response.status}")
                return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    if session is None:
        async with aiohttp.ClientSession() as new_session:
            return await _fetch(new_session)
    return await _fetch(session)

def parse_html(html_content: Optional[str]) -> str:
    """
    Parse HTML content and extract text with hyperlinks in markdown format.

    Args:
        html_content: HTML content to parse

    Returns:
        Extracted text in markdown format
    """
    if not html_content:
        return ""
    
    try:
        document = html5lib.parse(html_content)
        result = []
        seen_texts = set()
        
        def should_skip_element(elem) -> bool:
            """Check if the element should be skipped during parsing."""
            if elem.tag in ['{http://www.w3.org/1999/xhtml}script', 
                          '{http://www.w3.org/1999/xhtml}style']:
                return True
            if not any(text.strip() for text in elem.itertext()):
                return True
            return False
        
        def process_element(elem, depth: int = 0) -> None:
            """Process an HTML element and its children recursively."""
            if should_skip_element(elem):
                return
            
            if hasattr(elem, 'text') and elem.text:
                text = elem.text.strip()
                if text and text not in seen_texts:
                    if elem.tag == '{http://www.w3.org/1999/xhtml}a':
                        href = None
                        for attr, value in elem.items():
                            if attr.endswith('href'):
                                href = value
                                break
                        if href and not href.startswith(('#', 'javascript:')):
                            link_text = f"[{text}]({href})"
                            result.append("  " * depth + link_text)
                            seen_texts.add(text)
                    else:
                        result.append("  " * depth + text)
                        seen_texts.add(text)
            
            for child in elem:
                process_element(child, depth + 1)
            
            if hasattr(elem, 'tail') and elem.tail:
                tail = elem.tail.strip()
                if tail and tail not in seen_texts:
                    result.append("  " * depth + tail)
                    seen_texts.add(tail)
        
        body = document.find('.//{http://www.w3.org/1999/xhtml}body')
        if body is not None:
            process_element(body)
        else:
            process_element(document)
        
        filtered_result = []
        for line in result:
            if any(pattern in line.lower() for pattern in [
                'var ', 
                'function()', 
                '.js',
                '.css',
                'google-analytics',
                'disqus',
                '{',
                '}'
            ]):
                continue
            filtered_result.append(line)
        
        return '\n'.join(filtered_result)
    except Exception as e:
        logger.error(f"Error parsing HTML: {str(e)}")
        return ""

async def process_urls(urls: List[str], max_concurrent: int = 5) -> List[str]:
    """
    Process multiple URLs concurrently.

    Args:
        urls: List of URLs to process
        max_concurrent: Maximum number of concurrent requests

    Returns:
        List of processed content strings
    """
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_page(url, session) for url in urls]
        html_contents = await asyncio.gather(*tasks)
    
    with Pool() as pool:
        results = pool.map(parse_html, html_contents)
    
    return results

def validate_url(url: str) -> bool:
    """
    Validate if a string is a valid URL.

    Args:
        url: URL string to validate

    Returns:
        True if URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

# Main Tool Functions
def perform_search(query: str, max_results: int = 5, max_retries: int = 3) -> str:
    """
    Perform a web search and return formatted results.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        max_retries: Maximum number of retry attempts

    Returns:
        Formatted string containing search results or error message
    """
    try:
        results = search_with_retry(query, max_results, max_retries)
        return format_search_results(results)
    except Exception as e:
        return f"Error during search: {e}"

def fetch_web_content(urls: List[str], max_concurrent: int = 3) -> str:
    """
    Fetch and process web content from multiple URLs.

    Args:
        urls: List of URLs to fetch and process
        max_concurrent: Maximum number of concurrent requests

    Returns:
        Formatted string containing processed content or error message
    """
    try:
        # Validate URLs
        valid_urls = [url for url in urls if validate_url(url)]
        if not valid_urls:
            return "No valid URLs provided"
        
        # Process URLs
        results = asyncio.run(process_urls(valid_urls, max_concurrent))
        
        # Format output
        output = []
        for url, content in zip(valid_urls, results):
            output.append(f"\n=== Content from {url} ===\n")
            output.append(content)
        
        return "\n".join(output)
    except Exception as e:
        return f"Error during web scraping: {e}"

def create_file(filename: str, content: str) -> str:
    """
    Create a file with the given content and return its content.

    Args:
        filename: Name of the file to create
        content: Content to write to the file

    Returns:
        File content after writing or error message
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        # Read back the content to confirm
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error creating file: {str(e)}"

def execute_python(filename: str) -> str:
    """
    Execute a Python script and return its stdout.

    Args:
        filename: Name of the Python file to execute

    Returns:
        Script output or error message
    """
    try:
        result = subprocess.run(
            ["python", filename],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing Python script: stdout={e.stdout}, stderr={e.stderr}"
    except Exception as e:
        return f"Error executing Python script: {str(e)}"

def visual_qa(image_path: str, question: Optional[str] = None) -> str:
    """
    Answer questions about images using advanced vision models.
    
    Args:
        image_path: Path to the image file
        question: Optional question about the image
        
    Returns:
        Answer or description of the image
    """
    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}"
        
    try:
        # Encode image for API request
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"
            
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
        if not question:
            question = "Please write a detailed caption for this image."
            
        # Prepare API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        },
                    ]
                }
            ],
            "max_tokens": 1000,
        }
        
        # Make API request
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            return f"Error: API request failed with status {response.status_code}"
            
        result = response.json()
        output = result["choices"][0]["message"]["content"]
        
        if not question:
            output = f"You did not provide a particular question, so here is a detailed caption for the image: {output}"
            
        return output
        
    except Exception as e:
        return f"Error processing image: {str(e)}"

def inspect_file_as_text(file_path: str, question: Optional[str] = None) -> str:
    """
    Read a file as text and optionally answer questions about it.
    
    Args:
        file_path: Path to the file to inspect
        question: Optional question about the file content
        
    Returns:
        File content or answer to the question
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
        
    try:
        # Initialize converter
        converter = MarkdownConverter()
        
        # Convert file to text
        result = converter.convert(file_path)
        
        if not question:
            return result.text_content
            
        # If there's a question, use LLM to analyze
        client = openai.OpenAI()
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You will have to write a short caption for this file, then answer this question:"
                        + question,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the complete file:\n### "
                        + str(result.title)
                        + "\n\n"
                        + result.text_content[:100000],  # Limit text size
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Now answer the question below. Use these three headings: '1. Short answer', '2. Extremely detailed answer', '3. Additional Context on the document and question asked'."
                        + question,
                    }
                ],
            },
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error processing file: {str(e)}"

def reformulate_response(original_task: str, conversation_messages: List[dict]) -> str:
    """
    Reformulate a conversation into a concise final answer.
    
    Args:
        original_task: The original question or task
        conversation_messages: List of conversation messages
        
    Returns:
        Reformulated final answer
    """
    try:
        # Initialize OpenAI client
        client = openai.OpenAI()
        
        # Prepare messages for reformulation
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Earlier you were asked the following:

{original_task}

Your team then worked diligently to address that request. Read below a transcript of that conversation:""",
                    }
                ],
            }
        ]
        
        # Copy conversation messages
        try:
            for message in conversation_messages:
                if not message.get("content"):
                    continue
                message = copy.deepcopy(message)
                message["role"] = "user"  # All messages become user messages in reformulation
                messages.append(message)
        except Exception:
            messages.append({"role": "user", "content": str(conversation_messages)})
        
        # Add final answer request
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
Read the above conversation and output a FINAL ANSWER to the question. The question is repeated here for convenience:

{original_task}

To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and DO NOT INCLUDE UNITS such as $ or USD or percent signs unless specified otherwise.
If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
If you are unable to determine the final answer, output 'FINAL ANSWER: Unable to determine'
""",
                }
            ],
        })
        
        # Get reformulated response
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000
        )
        
        # Extract final answer
        final_answer = response.choices[0].message.content.split("FINAL ANSWER: ")[-1].strip()
        logger.info(f"> Reformulated answer: {final_answer}")
        
        return final_answer
        
    except Exception as e:
        return f"Error reformulating response: {str(e)}" 