import os
import json
from datetime import datetime
from dotenv import load_dotenv

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