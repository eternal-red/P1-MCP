"""
MCP tools for cal_functions.py
"""
from cal_functions import get_available_times, book_meeting

def mcp_get_available_times(start_date: str, end_date: str, duration: int = 30) -> dict:
    """
    MCP tool wrapper for get_available_times.
    """
    return get_available_times(start_date, end_date, duration)


def mcp_book_meeting(name: str, email: str, phone: str, start_time: str) -> dict:
    """
    MCP tool wrapper for book_meeting.
    """
    return book_meeting(name, email, phone, start_time)
