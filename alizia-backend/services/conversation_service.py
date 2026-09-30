"""Alizia AI Backend - Conversation Service"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import UUID, uuid4

from core.config import settings


class Conversation:
    """Conversation entity representing a chat session."""
    
    def __init__(
        self,
        conversation_id: str,
        title: Optional[str] = None,
        model: str = settings.DEFAULT_MODEL,
        metadata: Optional[Dict[str, Any]] = None,
        organization_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        self.id = conversation_id
        self.title = title or f"Conversation {conversation_id[:8]}"
        self.model = model
        self.metadata = metadata or {}
        self.organization_id = organization_id
        self.workspace_id = workspace_id
        self.user_id = user_id
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.messages: List[Dict[str, Any]] = []
        self.files: List[str] = []
        self.memories: List[str] = []
        self.tool_executions: List[Dict] = []
        self.agent_runs: List[str] = []
    
    def add_message(
        self,
        role: str,
        content: List[Dict[str, Any]],
        tool_calls: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a message to the conversation."""
        message = {
            "id": str(uuid4()),
            "role": role,
            "content": content,
            "tool_calls": tool_calls or [],
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        return message
    
    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """Get the most recent message."""
        if self.messages:
            return self.messages[-1]
        return None
    
    def get_message_count(self) -> int:
        """Get total message count."""
        return len(self.messages)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "model": self.model,
            "metadata": self.metadata,
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "messages": self.messages,
            "files": self.files,
            "memories": self.memories,
            "tool_executions": self.tool_executions,
            "agent_runs": self.agent_runs,
        }


# In-memory conversation store (replace with database in production)
_conversations: Dict[str, Conversation] = {}


def get_conversation(conversation_id: str) -> Optional[Conversation]:
    """Retrieve a conversation by ID."""
    return _conversations.get(conversation_id)


def create_conversation(
    title: Optional[str] = None,
    model: str = settings.DEFAULT_MODEL,
    metadata: Optional[Dict[str, Any]] = None,
    organization_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Conversation:
    """Create a new conversation."""
    conv_id = str(uuid4())
    conversation = Conversation(
        conversation_id=conv_id,
        title=title,
        model=model,
        metadata=metadata,
        organization_id=organization_id,
        workspace_id=workspace_id,
        user_id=user_id,
    )
    _conversations[conv_id] = conversation
    return conversation


def list_conversations(
    user_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> List[Conversation]:
    """List conversations filtered by criteria."""
    conversations = list(_conversations.values())
    filtered = conversations
    
    if user_id:
        filtered = [c for c in filtered if c.user_id == user_id]
    if organization_id:
        filtered = [c for c in filtered if c.organization_id == organization_id]
    if workspace_id:
        filtered = [c for c in filtered if c.workspace_id == workspace_id]
    
    return filtered


def update_conversation_title(
    conversation_id: str,
    new_title: str,
) -> Optional[Conversation]:
    """Update conversation title."""
    conv = _conversations.get(conversation_id)
    if conv:
        conv.title = new_title
        conv.updated_at = datetime.utcnow()
    return conv


def delete_conversation(
    conversation_id: str,
) -> bool:
    """Delete a conversation."""
    if conversation_id in _conversations:
        del _conversations[conversation_id]
        return True
    return False


# FastAPI dependencies
async def get_conversation_dependency(
    conversation_id: str,
) -> Conversation:
    """Dependency to get a conversation or raise 404."""
    conv = get_conversation(conversation_id)
    if not conv:
        from fastapi import HTTPException
        from starlette import status
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )
    return conv


async def create_conversation_dependency(
    title: Optional[str] = None,
    model: str = settings.DEFAULT_MODEL,
    metadata: Optional[Dict[str, Any]] = None,
    organization_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Conversation:
    """Dependency to create a new conversation."""
    return create_conversation(
        title=title,
        model=model,
        metadata=metadata,
        organization_id=organization_id,
        workspace_id=workspace_id,
        user_id=user_id,
    )