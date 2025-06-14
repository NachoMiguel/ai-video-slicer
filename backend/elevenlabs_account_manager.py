import os
import json
from datetime import datetime
from dotenv import load_dotenv
import re
import requests
import ffmpeg
import logging

class ElevenLabsAccountManager:
    def __init__(self, json_path='backend/elevenlabs_accounts.json', env_path='.env'):
        load_dotenv(env_path)
        self.json_path = json_path
        self._load_accounts()

    def _load_accounts(self):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.accounts = data['accounts']
        self.last_account_index = data.get('last_account_index', 0)
        self.total_accounts = len(self.accounts)

    def _save_accounts(self):
        data = {
            'last_account_index': self.last_account_index,
            'accounts': self.accounts
        }
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def get_next_account(self):
        next_index = (self.last_account_index + 1) % self.total_accounts
        account = self.accounts[next_index]
        api_key = os.getenv(f"ELEVENLABS_API_KEY_{account['id']}")
        email = os.getenv(f"ELEVENLABS_EMAIL_{account['id']}")
        if not api_key or not email:
            raise ValueError(f"API key or email missing for account id {account['id']}")
        return {
            'id': account['id'],
            'email': email,
            'api_key': api_key,
            'credits_used': account['credits_used'],
            'credits_remaining': account['credits_remaining'],
            'last_used': account['last_used']
        }

    def update_account_usage(self, account_id, chars_used):
        for account in self.accounts:
            if account['id'] == account_id:
                account['credits_used'] += chars_used
                account['credits_remaining'] = max(0, account['credits_remaining'] - chars_used)
                account['last_used'] = datetime.now().isoformat()
                self.last_account_index = (account_id - 1) % self.total_accounts
                self._save_accounts()
                return
        raise ValueError(f"Account id {account_id} not found.")

    def get_account_api_key(self, account_id):
        return os.getenv(f"ELEVENLABS_API_KEY_{account_id}")

    def get_account_email(self, account_id):
        return os.getenv(f"ELEVENLABS_EMAIL_{account_id}")

def chunk_script_into_paragraphs(script: str) -> list:
    """
    Splits the input script into a list of paragraphs (chunks) for TTS processing.
    Paragraphs are separated by double newlines or single newlines with surrounding whitespace.
    Returns a list of non-empty, trimmed paragraphs.
    """
    # Split by double newlines or by a newline followed by whitespace and another newline
    raw_chunks = re.split(r'\n\s*\n', script)
    # Remove empty and whitespace-only chunks
    chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]
    return chunks 

def synthesize_chunks_with_account_switching(chunks, voice_id, output_dir, account_manager):
    """
    Synthesizes each chunk using ElevenLabs TTS, switching accounts on credit error.
    Returns a list of audio file paths and a dict mapping account_id to total chars used.
    Logs errors and warnings for failed synthesis and account switching.
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_files = []
    account_usage = {}
    chunk_account_map = []

    for idx, chunk in enumerate(chunks):
        success = False
        attempts = 0
        max_attempts = account_manager.total_accounts
        while not success and attempts < max_attempts:
            account = account_manager.get_next_account()
            api_key = account['api_key']
            account_id = account['id']
            headers = {
                'xi-api-key': api_key,
                'Content-Type': 'application/json'
            }
            data = {
                'text': chunk,
                'voice_settings': {
                    'stability': 0.5,
                    'similarity_boost': 0.5
                }
            }
            url = f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}'
            try:
                response = requests.post(url, headers=headers, json=data)
            except Exception as e:
                print(f'[ERROR] Network or request error for chunk {idx+1}: {e}')
                break
            if response.status_code == 200:
                audio_path = os.path.join(output_dir, f'chunk_{idx+1}.mp3')
                with open(audio_path, 'wb') as f:
                    f.write(response.content)
                audio_files.append(audio_path)
                chunk_account_map.append(account_id)
                account_usage[account_id] = account_usage.get(account_id, 0) + len(chunk)
                success = True
            else:
                # Check for credit/limit error (status 402 or specific error message)
                try:
                    error_json = response.json()
                    error_message = error_json.get('detail', '').lower()
                except Exception:
                    error_message = ''
                if response.status_code == 402 or 'insufficient' in error_message or 'credit' in error_message:
                    print(f'[WARNING] Credit error for account {account_id} on chunk {idx+1}, switching account.')
                    account_manager.last_account_index = (account_id) % account_manager.total_accounts
                    attempts += 1
                else:
                    print(f'[ERROR] Synthesis failed for chunk {idx+1}: {response.status_code} {response.text}')
                    break
        if not success:
            print(f'[ERROR] Failed to synthesize chunk {idx+1} after trying all accounts.')
            audio_files.append(None)
            chunk_account_map.append(None)
    return audio_files, chunk_account_map, account_usage

def verify_full_script_coverage(audio_files):
    """
    Verifies that all chunks have corresponding audio files.
    Returns (True, []) if all are present, (False, missing_indices) otherwise.
    Logs missing chunks as warnings.
    """
    missing_indices = [i for i, path in enumerate(audio_files) if path is None or not os.path.exists(path)]
    if missing_indices:
        print(f'[WARNING] Missing audio for chunks: {missing_indices}')
    return (len(missing_indices) == 0, missing_indices)

def concatenate_audio_files(audio_files, output_path):
    """
    Concatenates a list of audio files into a single output file using ffmpeg-python.
    Returns the path to the final concatenated audio file.
    Logs errors if concatenation fails.
    """
    valid_files = [f for f in audio_files if f and os.path.exists(f)]
    if not valid_files:
        print('[ERROR] No valid audio files to concatenate.')
        raise ValueError("No valid audio files to concatenate.")
    list_file = output_path + '_inputs.txt'
    try:
        with open(list_file, 'w') as f:
            for file_path in valid_files:
                f.write(f"file '{os.path.abspath(file_path)}'\n")
        (
            ffmpeg
            .input(list_file, format='concat', safe=0)
            .output(output_path, acodec='copy')
            .run(overwrite_output=True, quiet=True)
        )
    except Exception as e:
        print(f'[ERROR] Audio concatenation failed: {e}')
        raise
    finally:
        if os.path.exists(list_file):
            os.remove(list_file)
    return output_path

def update_accounts_usage_from_dict(account_manager, account_usage):
    """
    Updates the usage stats for each account in account_usage dict.
    Calls account_manager.update_account_usage for each account_id and chars_used.
    """
    for account_id, chars_used in account_usage.items():
        account_manager.update_account_usage(account_id, chars_used) 