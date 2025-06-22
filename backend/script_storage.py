import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

class ScriptStorage:
    def __init__(self, scripts_dir: str = "scripts"):
        self.scripts_dir = Path(scripts_dir)
        self.scripts_dir.mkdir(exist_ok=True)
    
    def save_script(self, script_data: Dict[str, Any]) -> str:
        """
        Save a script to the scripts directory
        Returns the script ID
        """
        try:
            # Generate unique ID if not provided
            script_id = script_data.get('id', str(uuid.uuid4()))
            
            # Add metadata
            script_data.update({
                'id': script_id,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            })
            
            # Create filename
            title_safe = self._sanitize_filename(script_data.get('title', 'untitled'))
            filename = f"script_{title_safe}_{script_id[:8]}.json"
            filepath = self.scripts_dir / filename
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(script_data, f, indent=2, ensure_ascii=False)
            
            return script_id
            
        except Exception as e:
            raise Exception(f"Failed to save script: {str(e)}")
    
    def load_script(self, script_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a script by ID
        """
        try:
            for filepath in self.scripts_dir.glob("*.json"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        script_data = json.load(f)
                        if script_data.get('id') == script_id:
                            return script_data
                except (json.JSONDecodeError, KeyError):
                    continue
            return None
            
        except Exception as e:
            raise Exception(f"Failed to load script: {str(e)}")
    
    def list_scripts(self) -> List[Dict[str, Any]]:
        """
        List all saved scripts with metadata
        """
        try:
            scripts = []
            
            for filepath in self.scripts_dir.glob("*.json"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        script_data = json.load(f)
                        
                        # Extract summary info
                        summary = {
                            'id': script_data.get('id'),
                            'title': script_data.get('title', 'Untitled Script'),
                            'word_count': script_data.get('word_count', 0),
                            'created_at': script_data.get('created_at'),
                            'updated_at': script_data.get('updated_at'),
                            'source_url': script_data.get('source_url'),
                            'filename': filepath.name,
                            'script_length': len(script_data.get('script_text', ''))
                        }
                        scripts.append(summary)
                        
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Invalid script file {filepath}: {e}")
                    continue
            
            # Sort by created_at (newest first)
            scripts.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            return scripts
            
        except Exception as e:
            raise Exception(f"Failed to list scripts: {str(e)}")
    
    def delete_script(self, script_id: str) -> bool:
        """
        Delete a script by ID
        Returns True if deleted, False if not found
        """
        try:
            for filepath in self.scripts_dir.glob("*.json"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        script_data = json.load(f)
                        if script_data.get('id') == script_id:
                            filepath.unlink()
                            return True
                except (json.JSONDecodeError, KeyError):
                    continue
            return False
            
        except Exception as e:
            raise Exception(f"Failed to delete script: {str(e)}")
    
    def get_latest_script(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recently created script
        """
        try:
            scripts = self.list_scripts()
            if scripts:
                latest_id = scripts[0]['id']
                return self.load_script(latest_id)
            return None
            
        except Exception as e:
            raise Exception(f"Failed to get latest script: {str(e)}")
    
    def update_script(self, script_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing script
        """
        try:
            script_data = self.load_script(script_id)
            if not script_data:
                return False
            
            # Update fields
            script_data.update(updates)
            script_data['updated_at'] = datetime.now().isoformat()
            
            # Save back
            self.save_script(script_data)
            return True
            
        except Exception as e:
            raise Exception(f"Failed to update script: {str(e)}")
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for safe file system usage
        """
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length and clean up
        filename = filename.strip()[:50]
        filename = filename.replace(' ', '_').lower()
        
        return filename or 'untitled'
    
    def create_script_from_session(self, session_data: Dict[str, Any]) -> str:
        """
        Create a script entry from session data
        """
        try:
            # Extract video title from source URL or use default
            title = "Generated Script"
            if session_data.get('source_url'):
                # Try to extract title from session or use URL
                title = f"Script from {session_data['source_url']}"
            
            script_data = {
                'title': title,
                'script_text': session_data.get('current_script', ''),
                'word_count': session_data.get('word_count', 0),
                'source_url': session_data.get('source_url'),
                'entry_method': session_data.get('entry_method'),
                'messages': session_data.get('messages', []),
                'session_id': session_data.get('id')
            }
            
            return self.save_script(script_data)
            
        except Exception as e:
            raise Exception(f"Failed to create script from session: {str(e)}")

# Global instance
script_storage = ScriptStorage() 