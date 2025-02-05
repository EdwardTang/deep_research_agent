"""
Document conversion utilities for text inspection tool.
Adapted from Microsoft's AutoGen project.
"""

import base64
import copy
import html
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Union
from urllib.parse import parse_qs, quote, unquote, urlparse, urlunparse

import mammoth
import markdownify
import pandas as pd
import pdfminer.high_level
from bs4 import BeautifulSoup


class _CustomMarkdownify(markdownify.MarkdownConverter):
    """Custom Markdown converter with enhanced features."""

    def __init__(self, **options: Any):
        options["heading_style"] = options.get("heading_style", markdownify.ATX)
        super().__init__(**options)

    def convert_hn(self, n: int, el: Any, text: str, convert_as_inline: bool) -> str:
        """Convert headings with proper newlines."""
        if not convert_as_inline:
            if not re.search(r"^\n", text):
                return "\n" + super().convert_hn(n, el, text, convert_as_inline)
        return super().convert_hn(n, el, text, convert_as_inline)

    def convert_a(self, el: Any, text: str, convert_as_inline: bool):
        """Convert links with proper escaping."""
        prefix, suffix, text = markdownify.chomp(text)
        if not text:
            return ""
        href = el.get("href")
        title = el.get("title")

        if href:
            try:
                parsed_url = urlparse(href)
                if parsed_url.scheme and parsed_url.scheme.lower() not in ["http", "https", "file"]:
                    return "%s%s%s" % (prefix, text, suffix)
                href = urlunparse(parsed_url._replace(path=quote(unquote(parsed_url.path))))
            except ValueError:
                return "%s%s%s" % (prefix, text, suffix)

        if (
            self.options["autolinks"]
            and text.replace(r"\_", "_") == href
            and not title
            and not self.options["default_title"]
        ):
            return "<%s>" % href
        if self.options["default_title"] and not title:
            title = href
        title_part = ' "%s"' % title.replace('"', r"\"") if title else ""
        return "%s[%s](%s%s)%s" % (prefix, text, href, title_part, suffix) if href else text

    def convert_img(self, el: Any, text: str, convert_as_inline: bool) -> str:
        """Convert images with data URI handling."""
        alt = el.attrs.get("alt", None) or ""
        src = el.attrs.get("src", None) or ""
        title = el.attrs.get("title", None) or ""
        title_part = ' "%s"' % title.replace('"', r"\"") if title else ""
        if convert_as_inline and el.parent.name not in self.options["keep_inline_images_in"]:
            return alt

        if src.startswith("data:"):
            src = src.split(",")[0] + "..."

        return "![%s](%s%s)" % (alt, src, title_part)


class DocumentConverterResult:
    """Result of document conversion."""

    def __init__(self, title: Union[str, None] = None, text_content: str = ""):
        self.title = title
        self.text_content = text_content


class DocumentConverter:
    """Base class for document converters."""

    def convert(self, local_path: str, **kwargs: Any) -> Union[None, DocumentConverterResult]:
        raise NotImplementedError()


class PlainTextConverter(DocumentConverter):
    """Convert plain text files."""

    def convert(self, local_path: str, **kwargs: Any) -> Union[None, DocumentConverterResult]:
        content_type, _ = mimetypes.guess_type("__placeholder" + kwargs.get("file_extension", ""))
        if content_type is None:
            return None

        with open(local_path, "rt", encoding="utf-8") as fh:
            text_content = fh.read()
        return DocumentConverterResult(title=None, text_content=text_content)


class HtmlConverter(DocumentConverter):
    """Convert HTML files."""

    def convert(self, local_path: str, **kwargs: Any) -> Union[None, DocumentConverterResult]:
        extension = kwargs.get("file_extension", "")
        if extension.lower() not in [".html", ".htm"]:
            return None

        with open(local_path, "rt", encoding="utf-8") as fh:
            return self._convert(fh.read())

    def _convert(self, html_content: str) -> Union[None, DocumentConverterResult]:
        soup = BeautifulSoup(html_content, "html.parser")
        for script in soup(["script", "style"]):
            script.extract()

        body_elm = soup.find("body")
        webpage_text = ""
        if body_elm:
            webpage_text = _CustomMarkdownify().convert_soup(body_elm)
        else:
            webpage_text = _CustomMarkdownify().convert_soup(soup)

        return DocumentConverterResult(
            title=None if soup.title is None else soup.title.string,
            text_content=webpage_text
        )


class PdfConverter(DocumentConverter):
    """Convert PDF files."""

    def convert(self, local_path: str, **kwargs: Any) -> Union[None, DocumentConverterResult]:
        extension = kwargs.get("file_extension", "")
        if extension.lower() != ".pdf":
            return None

        return DocumentConverterResult(
            title=None,
            text_content=pdfminer.high_level.extract_text(local_path)
        )


class DocxConverter(HtmlConverter):
    """Convert DOCX files."""

    def convert(self, local_path: str, **kwargs: Any) -> Union[None, DocumentConverterResult]:
        extension = kwargs.get("file_extension", "")
        if extension.lower() != ".docx":
            return None

        with open(local_path, "rb") as docx_file:
            result = mammoth.convert_to_html(docx_file)
            return self._convert(result.value)


class XlsxConverter(HtmlConverter):
    """Convert XLSX files."""

    def convert(self, local_path: str, **kwargs: Any) -> Union[None, DocumentConverterResult]:
        extension = kwargs.get("file_extension", "")
        if extension.lower() not in [".xlsx", ".xls"]:
            return None

        sheets = pd.read_excel(local_path, sheet_name=None)
        md_content = []
        for sheet_name, df in sheets.items():
            md_content.append(f"## {sheet_name}")
            html_content = df.to_html(index=False)
            md_content.append(self._convert(html_content).text_content.strip())

        return DocumentConverterResult(
            title=None,
            text_content="\n\n".join(md_content)
        )


class MarkdownConverter:
    """Main converter class that handles multiple document types."""

    def __init__(self):
        self._converters = []
        self.register_converter(PlainTextConverter())
        self.register_converter(HtmlConverter())
        self.register_converter(PdfConverter())
        self.register_converter(DocxConverter())
        self.register_converter(XlsxConverter())

    def convert(self, file_path: str) -> DocumentConverterResult:
        """Convert a file to markdown text."""
        extension = os.path.splitext(file_path)[1].lower()
        
        for converter in self._converters:
            try:
                result = converter.convert(file_path, file_extension=extension)
                if result is not None:
                    return result
            except Exception as e:
                print(f"Error converting with {converter.__class__.__name__}: {e}", file=sys.stderr)
                continue

        raise ValueError(f"No converter found for file type: {extension}")

    def register_converter(self, converter: DocumentConverter) -> None:
        """Register a new document converter."""
        self._converters.insert(0, converter) 