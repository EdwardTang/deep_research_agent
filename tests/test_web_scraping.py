"""
Test script for web scraping functionality.
"""

import os
import sys
import pytest
import asyncio
import aiohttp
import pytest_asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import List, Optional
from urllib.parse import urlparse

# Import from deep_research_agent package
from deep_research_agent.tools import (
    fetch_page,
    process_urls,
    fetch_web_content,
    WebScrapingError,
    CookieError,
    RateLimitError,
    check_cookie_freshness,
    DomainConcurrencyTracker,
    MAX_RETRIES,
    RETRY_DELAY,
    async_fetch_web_content
)

# Test URLs
TEST_URLS = [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://another-site.com/page1"
]

# Mock responses
MOCK_HTML = """
<html>
<body>
    <h1>Test Page</h1>
    <p>This is a test paragraph.</p>
    <a href="https://example.com/link">Test Link</a>
</body>
</html>
"""

@pytest_asyncio.fixture
async def mock_response():
    """Create a mock aiohttp response."""
    response = AsyncMock()
    response.status = 200
    response.text.return_value = MOCK_HTML
    response.__aenter__.return_value = response
    response.__aexit__.return_value = None
    return response

@pytest_asyncio.fixture
async def mock_session():
    """Create a mock aiohttp session."""
    session = AsyncMock(spec=aiohttp.ClientSession)
    return session

@pytest.mark.asyncio
async def test_fetch_page_success(mock_session, mock_response):
    """Test successful page fetch."""
    mock_session.get.return_value = mock_response
    content = await fetch_page("https://example.com", mock_session)
    assert content == MOCK_HTML
    mock_session.get.assert_called_once()

@pytest.mark.asyncio
async def test_fetch_page_retry_on_error(mock_session):
    """Test retry mechanism on temporary errors."""
    # Create two responses - first fails, second succeeds
    error_response = AsyncMock()
    error_response.status = 500
    error_response.__aenter__.return_value = error_response
    error_response.__aexit__.return_value = None

    success_response = AsyncMock()
    success_response.status = 200
    success_response.text.return_value = MOCK_HTML
    success_response.__aenter__.return_value = success_response
    success_response.__aexit__.return_value = None

    mock_session.get.side_effect = [error_response, success_response]
    
    content = await fetch_page("https://example.com", mock_session)
    assert content == MOCK_HTML
    assert mock_session.get.call_count == 2

@pytest.mark.asyncio
async def test_fetch_page_rate_limit(mock_session):
    """Test handling of rate limit errors."""
    response = AsyncMock()
    response.status = 429
    response.__aenter__.return_value = response
    response.__aexit__.return_value = None
    
    mock_session.get.return_value = response
    
    with pytest.raises(RateLimitError):
        await fetch_page("https://example.com", mock_session)

@pytest.mark.asyncio
async def test_fetch_page_cookie_error(mock_session):
    """Test handling of cookie errors."""
    response = AsyncMock()
    response.status = 403
    response.__aenter__.return_value = response
    response.__aexit__.return_value = None
    
    mock_session.get.return_value = response
    
    with pytest.raises(CookieError):
        await fetch_page("https://example.com", mock_session)

def test_check_cookie_freshness():
    """Test cookie freshness check."""
    import time
    from http.cookiejar import Cookie
    
    # Create test cookies
    fresh_cookie = Cookie(
        version=0,
        name="fresh",
        value="value",
        port=None,
        port_specified=False,
        domain="example.com",
        domain_specified=True,
        domain_initial_dot=False,
        path="/",
        path_specified=True,
        secure=True,
        expires=int(time.time()) + 48*3600,  # 48 hours from now
        discard=False,
        comment=None,
        comment_url=None,
        rest={},
        rfc2109=False
    )
    
    stale_cookie = Cookie(
        version=0,
        name="stale",
        value="value",
        port=None,
        port_specified=False,
        domain="example.com",
        domain_specified=True,
        domain_initial_dot=False,
        path="/",
        path_specified=True,
        secure=True,
        expires=int(time.time()) + 1*3600,  # 1 hour from now
        discard=False,
        comment=None,
        comment_url=None,
        rest={},
        rfc2109=False
    )
    
    assert check_cookie_freshness(fresh_cookie) is True
    assert check_cookie_freshness(stale_cookie) is False

def test_domain_concurrency_tracker():
    """Test domain concurrency tracking."""
    tracker = DomainConcurrencyTracker()
    
    # Same domain should get same lock
    lock1 = tracker.get_lock("https://example.com/page1")
    lock2 = tracker.get_lock("https://example.com/page2")
    assert lock1 is lock2
    
    # Different domains should get different locks
    lock3 = tracker.get_lock("https://another-site.com/page1")
    assert lock1 is not lock3

@pytest.mark.asyncio
async def test_process_urls(mock_session, mock_response):
    """Test processing multiple URLs."""
    mock_session.get.return_value = mock_response
    with patch('aiohttp.ClientSession', return_value=mock_session):
        results = await process_urls(TEST_URLS, max_concurrent=2)
        assert len(results) == len(TEST_URLS)
        assert all(isinstance(r, str) for r in results)

@pytest.mark.asyncio
async def test_fetch_web_content(mock_session, mock_response):
    """Test the main web content fetching function."""
    # Configure mock response
    mock_response.status = 200
    mock_response.text.return_value = MOCK_HTML
    mock_response.__aenter__.return_value = mock_response
    mock_response.__aexit__.return_value = None

    # Configure mock session
    mock_session.get.return_value = mock_response
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None

    with patch('aiohttp.ClientSession', return_value=mock_session):
        result = await async_fetch_web_content(TEST_URLS)
        assert isinstance(result, str)
        # Check for parsed content (markdown format)
        assert "Test Page" in result
        assert "This is a test paragraph" in result
        assert "[Test Link](https://example.com/link)" in result

if __name__ == "__main__":
    pytest.main([__file__]) 