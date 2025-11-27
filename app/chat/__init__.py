# app/chat/__init__.py
"""
Chat session management module.
"""

from app.chat.manager import (
    create_chat,
    list_chats,
    get_chat,
    append_message,
    get_history,
    update_chat_title,
    delete_chat,
    clear_all_chats,
    get_stats
)

__all__ = [
    "create_chat",
    "list_chats",
    "get_chat",
    "append_message",
    "get_history",
    "update_chat_title",
    "delete_chat",
    "clear_all_chats",
    "get_stats"
]