# app/config/file_server_config.py
"""
Configuration for Windows File Server integration.
"""

import os
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class FileServerSettings:
    """Settings for file server connection."""
    
    # Windows mapped drive or UNC path
    # Change this to match your setup:
    # - Mapped drive: "Z:\\" or "Z:/"
    # - UNC path: "\\\\VM-FILESERVER\\SharedDocs"
    smb_path: str = "\\VM-FILESERVER\SharedDocs"
    
    # Mount point - on Windows use the drive letter or UNC path
    mount_point: str = "Z:/"
    
    # Folder aliases - map Greek names to actual folder names
    folder_aliases: Dict[str, str] = field(default_factory=lambda: {
        # Exact match for your folder
        "απογραφή και συντήρηση": "απογραφή και συντήρηση",
        "απογραφη και συντηρηση": "απογραφή και συντήρηση",
        
        # Partial matches
        "απογραφή": "απογραφή και συντήρηση",
        "απογραφη": "απογραφή και συντήρηση",
        "συντήρηση": "απογραφή και συντήρηση",
        "συντηρηση": "απογραφή και συντήρηση",
        "inventory": "απογραφή και συντήρηση",
        "maintenance": "απογραφή και συντήρηση",
        
        # Other folders
        "folder0": "folder0",
        "folder1": "folder1",
        "folder2": "folder2",
    })
    
    # Allowed file extensions
    allowed_extensions: List[str] = field(default_factory=lambda: [
        '.xlsx', '.xls', '.csv',
        '.pdf',
        '.docx', '.doc',
        '.txt', '.md',
    ])
    
    # Max file size (50MB)
    max_file_size: int = 50 * 1024 * 1024


def get_settings() -> FileServerSettings:
    """Get file server settings with environment overrides."""
    settings = FileServerSettings()
    
    # Override from environment variables
    if mount := os.getenv("FILE_SERVER_MOUNT"):
        settings.mount_point = mount
    
    # if smb := os.getenv("FILE_SERVER_SMB"):
    #     settings.smb_path = smb
    
    return settings