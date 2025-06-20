import os
import json
import uuid
import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import tempfile

class EntryMethod(Enum):
    YOUTUBE = "youtube"

class SessionPhase(Enum):
    ENTRY = "entry"
    EXTRACTION = "extraction"
    ANALYSIS = "analysis"
    BUILDING = "building"
    REVIEW = "review"
    COMPLETE = "complete"

class SectionStatus(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    ERROR = "error"

# Legacy BulletPoint and ScriptSection classes removed - using full script approach

@dataclass
class ChatMessage:
    id: str
    type: str  # 'user', 'ai', 'system'
    content: str
    timestamp: datetime.datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

# ScriptAnalysis class removed - YouTube only workflow

@dataclass
class ScriptSession:
    session_id: str
    entry_method: Optional[EntryMethod]
    current_phase: SessionPhase
    
    # YouTube path data
    youtube_url: Optional[str] = None
    video_title: Optional[str] = None
    transcript: Optional[str] = None
    use_default_prompt: bool = True
    custom_prompt: Optional[str] = None
    
    # Chat history - DORMANT: Kept for potential future chat features (unused in current single-panel design)
    chat_history: List[ChatMessage] = field(default_factory=list)
    
    # Session metadata
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    # Final script
    final_script: Optional[str] = None
    is_finalized: bool = False

class ScriptSessionManager:
    def __init__(self, sessions_dir: str = None):
        self.sessions_dir = sessions_dir or os.path.join(tempfile.gettempdir(), "script_sessions")
        os.makedirs(self.sessions_dir, exist_ok=True)
        self.active_sessions: Dict[str, ScriptSession] = {}
    
    def create_session(self) -> ScriptSession:
        """Create a new script building session"""
        session_id = str(uuid.uuid4())
        session = ScriptSession(
            session_id=session_id,
            entry_method=None,
            current_phase=SessionPhase.ENTRY
        )
        
        self.active_sessions[session_id] = session
        self._save_session_to_disk(session)
        return session
    
    def get_session(self, session_id: str) -> Optional[ScriptSession]:
        """Get an existing session"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Try to load from disk
        session = self._load_session_from_disk(session_id)
        if session:
            self.active_sessions[session_id] = session
        
        return session
    
    def update_session(self, session: ScriptSession) -> ScriptSession:
        """Update session data"""
        session.updated_at = datetime.datetime.now()
        self.active_sessions[session.session_id] = session
        self._save_session_to_disk(session)
        return session
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        session_file = os.path.join(self.sessions_dir, f"{session_id}.json")
        if os.path.exists(session_file):
            os.remove(session_file)
            return True
        
        return False
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up sessions older than max_age_hours"""
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=max_age_hours)
        
        # Clean up in-memory sessions
        expired_sessions = [
            session_id for session_id, session in self.active_sessions.items()
            if session.updated_at < cutoff_time
        ]
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        # Clean up disk sessions
        for filename in os.listdir(self.sessions_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.sessions_dir, filename)
                if os.path.getmtime(file_path) < cutoff_time.timestamp():
                    os.remove(file_path)
    
    def add_chat_message(self, session_id: str, message_type: str, content: str, metadata: Dict[str, Any] = None) -> ChatMessage:
        """Add a chat message to a session - DORMANT: Currently unused in single-panel design"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        message = ChatMessage(
            id=str(uuid.uuid4()),
            type=message_type,
            content=content,
            timestamp=datetime.datetime.now(),
            metadata=metadata or {}
        )
        
        session.chat_history.append(message)
        self.update_session(session)
        return message
    
    # Legacy section management methods removed - using full script approach
    
    def _save_session_to_disk(self, session: ScriptSession):
        """Save session to disk as JSON"""
        session_file = os.path.join(self.sessions_dir, f"{session.session_id}.json")
        
        # Convert to dictionary with proper serialization
        session_dict = self._serialize_session(session)
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_dict, f, indent=2, ensure_ascii=False)
    
    def _load_session_from_disk(self, session_id: str) -> Optional[ScriptSession]:
        """Load session from disk"""
        session_file = os.path.join(self.sessions_dir, f"{session_id}.json")
        
        if not os.path.exists(session_file):
            return None
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_dict = json.load(f)
            
            return self._deserialize_session(session_dict)
        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
            return None
    
    def _serialize_session(self, session: ScriptSession) -> Dict[str, Any]:
        """Convert session to JSON-serializable dictionary"""
        session_dict = asdict(session)
        
        # Convert datetime objects to ISO strings
        session_dict['created_at'] = session.created_at.isoformat()
        session_dict['updated_at'] = session.updated_at.isoformat()
        
        # Convert enums to strings
        if session.entry_method:
            session_dict['entry_method'] = session.entry_method.value
        session_dict['current_phase'] = session.current_phase.value
        
        # Handle chat messages
        for message_dict in session_dict['chat_history']:
            message_dict['timestamp'] = message_dict['timestamp'].isoformat() if isinstance(message_dict['timestamp'], datetime.datetime) else message_dict['timestamp']
        
        return session_dict
    
    def _deserialize_session(self, session_dict: Dict[str, Any]) -> ScriptSession:
        """Convert dictionary back to ScriptSession object"""
        # Convert datetime strings back to datetime objects
        session_dict['created_at'] = datetime.datetime.fromisoformat(session_dict['created_at'])
        session_dict['updated_at'] = datetime.datetime.fromisoformat(session_dict['updated_at'])
        
        # Convert enum strings back to enums
        if session_dict['entry_method']:
            session_dict['entry_method'] = EntryMethod(session_dict['entry_method'])
        session_dict['current_phase'] = SessionPhase(session_dict['current_phase'])
        
        # Handle chat messages
        chat_history = []
        for message_dict in session_dict['chat_history']:
            message_dict['timestamp'] = datetime.datetime.fromisoformat(message_dict['timestamp'])
            chat_history.append(ChatMessage(**message_dict))
        session_dict['chat_history'] = chat_history
        
        # Remove legacy fields if present in old sessions (backward compatibility)
        legacy_fields = ['script_analysis', 'uploaded_script', 'original_filename', 
                        'bullet_points', 'sections', 'total_word_count', 'target_word_count', 
                        'completion_percentage']
        for field in legacy_fields:
            if field in session_dict:
                del session_dict[field]
        
        return ScriptSession(**session_dict)

# Global session manager instance
session_manager = ScriptSessionManager() 