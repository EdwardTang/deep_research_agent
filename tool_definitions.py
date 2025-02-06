"""
Function definitions and descriptions for the interactive chat system tools.
"""

function_definitions = [
    {
        "name": "perform_search",
        "description": "Perform a web search using our native search tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum search results",
                    "default": 10
                },
                "max_retries": {
                    "type": "integer",
                    "description": "Maximum number of retry attempts",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "fetch_web_content",
        "description": "Fetch the content of web pages using our web scraper tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of URLs to scrape"
                },
                "max_concurrent": {
                    "type": "integer",
                    "description": "Maximum number of concurrent requests",
                    "default": 3
                }
            },
            "required": ["urls"]
        }
    },
    {
        "name": "create_file",
        "description": "Create or change a file with the given content and return its content for verification.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Name of the file to create"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["filename", "content"]
        }
    },
    {
        "name": "execute_command",
        "description": "Execute a terminal command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The terminal command to execute"
                },
                "explanation": {
                    "type": "string",
                    "description": "Explanation of what the command does"
                }
            },
            "required": ["command", "explanation"]
        }
    },
    {
        "name": "visual_qa",
        "description": "Answer questions about images using advanced vision models",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the image file"
                },
                "question": {
                    "type": "string",
                    "description": "Question about the image"
                }
            },
            "required": ["image_path"]
        }
    },
    {
        "name": "inspect_file_as_text",
        "description": """
Read a file as text and optionally answer questions about it.
This tool handles various file types including PDF, DOCX, XLSX, HTML, and plain text.
DO NOT use this tool for images - use visual_qa instead.""",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to inspect"
                },
                "question": {
                    "type": "string",
                    "description": "Optional question about the file content",
                    "nullable": True
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "reformulate_response",
        "description": """
Reformulate a conversation into a concise final answer.
This tool helps standardize and optimize responses from other tools by extracting the key information and formatting it according to specific rules.""",
        "parameters": {
            "type": "object",
            "properties": {
                "original_task": {
                    "type": "string",
                    "description": "The original question or task"
                },
                "conversation_messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "content": {"type": "string"}
                        }
                    },
                    "description": "List of conversation messages to reformulate"
                }
            },
            "required": ["original_task", "conversation_messages"]
        }
    },
    {
        "name": "get_file_description",
        "description": """
Generate descriptions for various types of files including images, documents, and archives.
This tool can handle multiple file types:
- Images: png, jpg, jpeg
- Documents: pdf, xls, xlsx, docx, doc, xml
- Audio: mp3, m4a, wav
- Archives: zip""",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to describe"
                },
                "question": {
                    "type": "string",
                    "description": "Optional question to guide the description generation",
                    "nullable": True
                },
                "extract_archives": {
                    "type": "boolean",
                    "description": "Whether to extract and process archive files",
                    "default": True
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "generate_targeted_description",
        "description": """
Generate a targeted description of a file (image or document) that provides relevant details for answering a specific question.
The description will focus on details that might be useful for answering the question, but will not answer the question directly.
The description will be 5 sentences long and will only include information present in the file.""",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to describe"
                },
                "question": {
                    "type": "string",
                    "description": "The question that the description should help answer"
                },
                "description_type": {
                    "type": "string",
                    "description": "Type of description to generate",
                    "enum": ["image", "document"],
                    "default": "document"
                }
            },
            "required": ["file_path", "question"]
        }
    }
] 