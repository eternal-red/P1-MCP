"""KB and external function call tools for MCP Streamable HTTP server."""

import argparse
from typing import List, Dict, Any
import uvicorn
from mcp.server.fastmcp import FastMCP
from logical_functions.cal_functions import available_times, book_meeting
from logical_functions.rag_functions import (
    start_encoder, search_rag, get_chunk_logic, build_prompt_logic
)



# Initialize FastMCP server for Weather tools.
# If json_response is set to True, the server will use JSON responses instead of SSE streams
# If stateless_http is set to True, the server uses true stateless mode (new transport per request)
mcp = FastMCP(name="PeregrineOne MCP server", json_response=False, stateless_http=False)

### CAL.com Tools ###
@mcp.tool()
async def get_available_times(start_date: str, end_date: str, duration: int=30) -> dict:
        """Get available booking, scheduling, meeting, or demo times from the calendar API.
    
        Args:
            start_date: Start date in ISO 8601 format (e.g. "2025-08-13T00:00:00Z")
            end_date: End date in ISO 8601 format, must be at least 1 day after start date, (e.g. "2025-08-20T00:00:00Z")
            duration: Duration of the booking in minutes (default is 30)
        """
        return await available_times(start_date, end_date, duration)
@mcp.tool()
async def post_book_meeting(name: str, email: str, phone: str, start_time: str) -> dict:
    """Book a meeting, demo, or appointment using the calendar API. You must have available times first. You must also have all required fields filled out.
    
    Args:
        name: Name of the attendee
        email: Email address of the attendee
        phone: Phone number of the attendee
        start_time: Start time of the meeting in ISO 8601 format (e.g. "2025-08-14T15:00:00Z")
    """
    return await book_meeting(name, email, phone, start_time)

### Knowledge Base Tools ###
@mcp.tool()
async def search_chunks(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    Perform a semantic vector search over all cached document chunks and return the top-k most relevant results.

    Args:
        query: The search query string to match against the document chunks.
        k: The number of top results to return (default is 4).

    Returns:
        A list of dictionaries, each containing:
            id: The integer index of the chunk in the cache.
            score: The cosine similarity score (float) between the query and the chunk.
            preview: A short preview of the chunk's text (str).
    """
    return search_rag(query, k)

@mcp.tool()
async def get_chunk(id: int) -> Dict[str, Any]:
    """
    Retrieve the full text and metadata for a specific document chunk by its id.

    Args:
        id: The integer index of the chunk to retrieve (must be within range).

    Returns:
        A dictionary containing:
            id: The integer index of the chunk.
            text: The full text content of the chunk (str).
            length: The number of characters in the chunk (int).
        If the id is out of range, returns a dictionary with an 'error' key and message.
    """
    return get_chunk_logic(id)

@mcp.tool()
async def build_prompt(question: str, chunk_ids: List[int]) -> str:
    """
    Construct a RAG prompt using the full text of selected document chunks and a user question.
    Args:
        question: The user question to be answered using the provided context.
        chunk_ids: A list of integer chunk ids to include as context in the prompt.
    Returns:
        A string containing the formatted prompt, which includes the concatenated text of the selected chunks and the question, instructing the model to answer using only the provided context.
    """
    return build_prompt_logic(question, chunk_ids)

if __name__ == "__main__":
    port = 8080  
    start_encoder()  # Start encoder at server startup to reduce latency
    uvicorn.run(mcp.streamable_http_app, host="0.0.0.0", port=port)
