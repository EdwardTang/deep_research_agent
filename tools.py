"""
Tool implementations for interactive chat system.
Contains search, web scraping, file operations, package management tools, and text inspection.
"""

import asyncio
import logging
import os
import subprocess
import time
import shutil
import zipfile
from multiprocessing import Pool
from typing import List, Optional, Union, Dict, Tuple
from urllib.parse import urlparse, quote, unquote
import base64
import json
import mimetypes
from io import BytesIO
import copy
import datetime

import aiohttp
import html5lib
from duckduckgo_search import DDGS
from PIL import Image
from huggingface_hub import InferenceClient
import requests
import openai
from requests.cookies import RequestsCookieJar

from mdconvert import MarkdownConverter

# Import cookies
try:
    from .cookies import COOKIES
except ImportError:
    # Create empty cookie jar if cookies.py is not available
    COOKIES = RequestsCookieJar()

logger = logging.getLogger(__name__)

# Constants for web scraping
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
COOKIE_EXPIRY_THRESHOLD = 24 * 60 * 60  # 24 hours in seconds
MAX_CONCURRENT_PER_DOMAIN = 3
REQUEST_TIMEOUT = 30  # seconds

class WebScrapingError(Exception):
    """Base class for web scraping errors."""
    pass

class CookieError(WebScrapingError):
    """Error related to cookie handling."""
    pass

class RateLimitError(WebScrapingError):
    """Error when rate limit is exceeded."""
    pass

class DomainConcurrencyTracker:
    """Track concurrent requests per domain."""
    def __init__(self):
        self._domain_locks: Dict[str, asyncio.Semaphore] = {}
    
    def get_lock(self, url: str) -> asyncio.Semaphore:
        """Get or create a semaphore for the domain."""
        domain = urlparse(url).netloc
        if domain not in self._domain_locks:
            self._domain_locks[domain] = asyncio.Semaphore(MAX_CONCURRENT_PER_DOMAIN)
        return self._domain_locks[domain]

_domain_tracker = DomainConcurrencyTracker()

def check_cookie_freshness(cookie: RequestsCookieJar) -> bool:
    """Check if a cookie is fresh enough to use."""
    if not cookie.expires:
        return True
    now = time.time()
    return (cookie.expires - now) > COOKIE_EXPIRY_THRESHOLD

async def fetch_page(url: str, session: Optional[aiohttp.ClientSession] = None, retry_count: int = 0) -> Optional[str]:
    """
    Asynchronously fetch a webpage's content with retries and error handling.

    Args:
        url: URL to fetch
        session: Optional aiohttp session to use
        retry_count: Current retry attempt number

    Returns:
        Page content as string if successful, None otherwise

    Raises:
        WebScrapingError: When all retries fail or other errors occur
        CookieError: When cookie-related errors occur
        RateLimitError: When rate limit is exceeded
    """
    if retry_count >= MAX_RETRIES:
        raise WebScrapingError(f"Max retries ({MAX_RETRIES}) exceeded for {url}")

    async def _fetch(session: aiohttp.ClientSession) -> Optional[str]:
        try:
            logger.info(f"Fetching {url}")

            # Get domain lock
            domain_lock = _domain_tracker.get_lock(url)

            # Convert requests cookies to aiohttp format
            cookies = {}
            domain = urlparse(url).netloc
            for cookie in COOKIES:
                if domain.endswith(cookie.domain.lstrip('.')):
                    if not check_cookie_freshness(cookie):
                        raise CookieError(f"Cookie for {domain} has expired or will expire soon")
                    cookies[cookie.name] = cookie.value

            # Use domain lock to limit concurrent requests
            async with domain_lock:
                async with session.get(url, cookies=cookies, timeout=REQUEST_TIMEOUT) as response:
                    if response.status == 200:
                        content = await response.text()
                        logger.info(f"Successfully fetched {url}")
                        return content
                    elif response.status == 429:
                        raise RateLimitError(f"Rate limit exceeded for {domain}")
                    elif response.status == 403:
                        raise CookieError(f"Access denied for {domain}. Cookie may be invalid.")
                    else:
                        # Any other error status should trigger a retry
                        logger.error(f"Error fetching {url}: HTTP {response.status}")
                        raise WebScrapingError(f"HTTP {response.status} error for {url}")

        except (CookieError, RateLimitError):
            # Don't retry these errors
            raise
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            raise WebScrapingError(str(e))

    try:
        if session is None:
            # Create session with default headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            async with aiohttp.ClientSession(headers=headers) as new_session:
                return await _fetch(new_session)
        return await _fetch(session)
    except (CookieError, RateLimitError):
        # Don't retry these errors
        raise
    except WebScrapingError:
        # Wait before retrying
        await asyncio.sleep(RETRY_DELAY * (2 ** retry_count))  # Exponential backoff
        return await fetch_page(url, session, retry_count + 1)

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
    Process multiple URLs concurrently with error handling.

    Args:
        urls: List of URLs to process
        max_concurrent: Maximum number of concurrent requests

    Returns:
        List of processed content strings
    """
    async def fetch_with_error_handling(url: str, session: aiohttp.ClientSession) -> Tuple[str, Optional[str]]:
        """Fetch a URL and handle errors."""
        try:
            content = await fetch_page(url, session)
            return url, content
        except (CookieError, RateLimitError) as e:
            logger.error(f"Authentication error for {url}: {str(e)}")
            return url, None
        except WebScrapingError as e:
            logger.error(f"Failed to fetch {url}: {str(e)}")
            return url, None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {str(e)}")
            return url, None

    # Create session with default headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    async with aiohttp.ClientSession(headers=headers) as session:
        # Create tasks for each URL
        tasks = [fetch_with_error_handling(url, session) for url in urls]
        
        # Process URLs in batches to limit concurrency
        results = []
        for i in range(0, len(tasks), max_concurrent):
            batch = tasks[i:i + max_concurrent]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
    
    # Process results
    processed_results = []
    for url, content in results:
        if content:
            try:
                processed_content = parse_html(content)
                processed_results.append(processed_content)
            except Exception as e:
                logger.error(f"Error parsing content from {url}: {str(e)}")
                processed_results.append("")
        else:
            processed_results.append("")
    
    return processed_results

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

def fetch_web_content(urls: List[str], use_existing_loop: bool = False) -> str:
    """
    Fetch content from multiple URLs and combine the results.
    This is a synchronous wrapper around async_fetch_web_content.

    Args:
        urls: List of URLs to fetch
        use_existing_loop: Whether to use the existing event loop (for testing)

    Returns:
        Combined content from all URLs
    """
    try:
        if use_existing_loop:
            # Use existing event loop (for testing)
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(process_urls(urls))
        else:
            # Create a new event loop for this function
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                should_close_loop = True
            else:
                should_close_loop = False

            results = loop.run_until_complete(process_urls(urls))
            
            if should_close_loop:
                loop.close()

        return _format_web_content(results)
    except Exception as e:
        logger.error(f"Error during web scraping: {str(e)}")
        return f"Error during web scraping: {str(e)}\nCheck logs for more details."

async def async_fetch_web_content(urls: List[str]) -> str:
    """
    Asynchronously fetch content from multiple URLs and combine the results.

    Args:
        urls: List of URLs to fetch

    Returns:
        Combined content from all URLs
    """
    try:
        results = await process_urls(urls)
        return _format_web_content(results)
    except Exception as e:
        logger.error(f"Error during web scraping: {str(e)}")
        return f"Error during web scraping: {str(e)}\nCheck logs for more details."

def _format_web_content(results: List[str]) -> str:
    """Format web content results into a single string."""
    # Filter out None values and join results
    valid_results = [r for r in results if r]
    if not valid_results:
        return "No valid content found"
    return "\n".join(valid_results)

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

def get_file_description(file_path: str, question: Optional[str] = None, extract_archives: bool = True) -> str:
    """
    Generate descriptions for various types of files.
    
    Args:
        file_path: Path to the file to describe
        question: Optional question to guide the description generation
        extract_archives: Whether to extract and process archive files
        
    Returns:
        Description of the file content
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
        
    try:
        # Get file extension
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Handle zip files
        if file_extension == '.zip' and extract_archives:
            return _process_zip_file(file_path, question)
            
        # Handle image files
        if file_extension in ['.png', '.jpg', '.jpeg']:
            description = visual_qa(file_path, question if question else "Please write a detailed caption for this image.")
            return f"Image description: {description}"
            
        # Handle document files
        if file_extension in ['.pdf', '.xls', '.xlsx', '.docx', '.doc', '.xml']:
            description = inspect_file_as_text(file_path, question if question else "Please write a detailed summary of this document.")
            return f"Document description: {description}"
            
        # Handle audio files
        if file_extension in ['.mp3', '.m4a', '.wav']:
            return f"Audio file detected: {os.path.basename(file_path)}"
            
        # Default case: try to read as text
        return inspect_file_as_text(file_path, question)
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return f"Error processing file: {str(e)}"

def _process_zip_file(zip_path: str, question: Optional[str] = None) -> str:
    """
    Process a zip file and generate descriptions for its contents.
    
    Args:
        zip_path: Path to the zip file
        question: Optional question to guide the description generation
        
    Returns:
        Combined description of all files in the archive
    """
    try:
        # Create temporary directory for extraction
        temp_dir = os.path.join(os.path.dirname(zip_path), '_temp_extract')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Extract archive
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        # Process each file
        descriptions = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                description = get_file_description(file_path, question, extract_archives=False)
                descriptions.append(f"- {os.path.relpath(file_path, temp_dir)}:\n  {description}")
                
        # Clean up
        shutil.rmtree(temp_dir)
        
        return "Archive contents:\n" + "\n".join(descriptions)
        
    except Exception as e:
        logger.error(f"Error processing zip file {zip_path}: {str(e)}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return f"Error processing zip file: {str(e)}"

def generate_targeted_description(file_path: str, question: str, description_type: str = "document") -> str:
    """
    Generate a targeted description of a file that provides relevant details for answering a specific question.
    
    Args:
        file_path: Path to the file to describe
        question: The question that the description should help answer
        description_type: Type of description to generate ("image" or "document")
        
    Returns:
        A targeted description of the file
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
        
    try:
        # Prepare the prompt template
        prompt_template = """Write a detailed description of this {file_type}, focusing on aspects relevant to this question: {question}
Please include the following in your description:
1. Mention specific details that would help answer the question
2. Use precise and descriptive language
3. Write exactly 5 sentences
4. Only include information that is actually present in the {file_type}
5. Do not try to directly answer the question"""
        
        prompt = prompt_template.format(
            file_type="image" if description_type == "image" else "document",
            question=question
        )
        
        if description_type == "image":
            # Use visual_qa for images
            return visual_qa(file_path, prompt)
        else:
            # Use inspect_file_as_text for documents
            return inspect_file_as_text(file_path, prompt)
            
    except Exception as e:
        logger.error(f"Error generating description for {file_path}: {str(e)}")
        return f"Error generating description: {str(e)}" 