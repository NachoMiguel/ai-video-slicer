import os
import json
from datetime import datetime
from dotenv import load_dotenv
import openai
import logging
import requests

class OpenAIAccountManager:
    def __init__(self, json_path='backend/openai_accounts.json', env_path=None):
        # Use the same .env path as main.py - backend/.env
        if env_path is None:
            env_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(env_path)
        print(f"[DEBUG] OpenAI Account Manager loading .env from: {env_path}")
        print(f"[DEBUG] .env file exists: {os.path.exists(env_path)}")
        self.json_path = json_path
        self._load_accounts()
        self._validate_setup()

    def _load_accounts(self):
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate JSON structure
            if 'accounts' not in data:
                raise ValueError("Invalid JSON format: missing 'accounts' key")
            
            if not isinstance(data['accounts'], list):
                raise ValueError("Invalid JSON format: 'accounts' must be a list")
            
            if len(data['accounts']) == 0:
                raise ValueError("No accounts found in configuration file")
            
            # Validate each account has required fields
            for i, account in enumerate(data['accounts']):
                required_fields = ['id', 'email', 'tokens_used']
                for field in required_fields:
                    if field not in account:
                        raise ValueError(f"Account {i+1} missing required field: {field}")
            
            self.accounts = data['accounts']
            self.last_account_index = data.get('last_account_index', 0)
            self.total_accounts = len(self.accounts)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"OpenAI accounts file not found: {self.json_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in accounts file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading accounts file: {e}")

    def _save_accounts(self):
        data = {
            'last_account_index': self.last_account_index,
            'accounts': self.accounts
        }
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def _validate_setup(self):
        """Validate that all accounts have corresponding API keys in environment."""
        missing_keys = []
        invalid_keys = []
        
        for account in self.accounts:
            account_id = account['id']
            api_key = os.getenv(f"OPENAI_API_KEY_{account_id}")
            
            if not api_key:
                missing_keys.append(account_id)
            elif not api_key.startswith('sk-'):
                invalid_keys.append(account_id)
        
        if missing_keys:
            print(f"[WARNING] Missing API keys for accounts: {missing_keys}")
            print(f"[WARNING] Add these to your .env file: OPENAI_API_KEY_{missing_keys[0]}, etc.")
        
        if invalid_keys:
            print(f"[WARNING] Invalid API keys for accounts: {invalid_keys}")
            print(f"[WARNING] OpenAI API keys should start with 'sk-'")
        
        # Check if we have at least one valid account
        valid_accounts = len(self.accounts) - len(missing_keys) - len(invalid_keys)
        if valid_accounts == 0:
            raise ValueError("No valid OpenAI accounts found. Please check your .env file and account configuration.")
        
        print(f"[INFO] OpenAI Account Manager initialized with {valid_accounts}/{len(self.accounts)} valid accounts")

    def get_next_account(self):
        """Get the next valid account with proper API key validation."""
        attempts = 0
        max_attempts = self.total_accounts
        
        while attempts < max_attempts:
            next_index = (self.last_account_index + 1) % self.total_accounts
            account = self.accounts[next_index]
            api_key = os.getenv(f"OPENAI_API_KEY_{account['id']}")
            
            # Skip invalid accounts
            if not api_key or not api_key.startswith('sk-'):
                print(f"[WARNING] Skipping invalid account {account['id']}")
                self.last_account_index = next_index
                attempts += 1
                continue
            
            # Update index and return valid account
            self.last_account_index = next_index
            return {
                'id': account['id'],
                'email': account['email'],
                'api_key': api_key,
                'tokens_used': account['tokens_used'],
                'last_used': account['last_used']
            }
        
        # If we get here, no valid accounts found
        raise ValueError("No valid OpenAI accounts available. Please check your API keys.")

    def update_account_usage(self, account_id, tokens_used):
        for account in self.accounts:
            if account['id'] == account_id:
                account['tokens_used'] += tokens_used
                account['last_used'] = datetime.now().isoformat()
                self.last_account_index = (account_id - 1) % self.total_accounts
                self._save_accounts()
                return
        raise ValueError(f"Account id {account_id} not found.")

    def get_account_api_key(self, account_id):
        return os.getenv(f"OPENAI_API_KEY_{account_id}")

    def get_account_email(self, account_id):
        return os.getenv(f"OPENAI_EMAIL_{account_id}")
    
    def get_usage_statistics(self):
        """Get detailed usage statistics for all accounts."""
        total_tokens = sum(account['tokens_used'] for account in self.accounts)
        stats = {
            'total_accounts': self.total_accounts,
            'total_tokens_used': total_tokens,
            'last_used_account': self.last_account_index + 1,
            'accounts': []
        }
        
        for account in self.accounts:
            account_stats = {
                'id': account['id'],
                'email': account['email'],
                'tokens_used': account['tokens_used'],
                'last_used': account['last_used'],
                'percentage_of_total': (account['tokens_used'] / total_tokens * 100) if total_tokens > 0 else 0
            }
            stats['accounts'].append(account_stats)
        
        return stats
    
    def print_usage_summary(self):
        """Print a formatted usage summary for all accounts."""
        stats = self.get_usage_statistics()
        print(f"\n[OPENAI USAGE SUMMARY]")
        print(f"Total Accounts: {stats['total_accounts']}")
        print(f"Total Tokens Used: {stats['total_tokens_used']}")
        print(f"Last Used Account: {stats['last_used_account']}")
        print(f"\nAccount Details:")
        for account in stats['accounts']:
            print(f"  Account {account['id']} ({account['email']}): {account['tokens_used']} tokens ({account['percentage_of_total']:.1f}%)")
            if account['last_used']:
                print(f"    Last used: {account['last_used']}")
        print("="*50)
    
    def test_account_connectivity(self, account_id=None, timeout=10):
        """Test if a specific account or all accounts can connect to OpenAI API."""
        if account_id:
            # Test specific account
            accounts_to_test = [acc for acc in self.accounts if acc['id'] == account_id]
        else:
            # Test all accounts
            accounts_to_test = self.accounts
        
        results = {'working': [], 'failed': []}
        
        for account in accounts_to_test:
            api_key = os.getenv(f"OPENAI_API_KEY_{account['id']}")
            if not api_key or not api_key.startswith('sk-'):
                results['failed'].append({'id': account['id'], 'error': 'Invalid or missing API key'})
                continue
            
            try:
                client = create_openai_client(api_key)
                # Simple test request
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=1
                )
                results['working'].append({'id': account['id'], 'email': account['email']})
                print(f"[SUCCESS] Account {account['id']} connectivity test passed")
            except Exception as e:
                results['failed'].append({'id': account['id'], 'error': str(e)})
                print(f"[FAILED] Account {account['id']} connectivity test failed: {e}")
        
        return results

def create_openai_client(api_key):
    """Create an OpenAI client with the given API key."""
    import os
    import requests
    
    # Clear any proxy environment variables that might interfere
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
    for var in proxy_vars:
        if var in os.environ:
            del os.environ[var]
    
    try:
        return openai.OpenAI(api_key=api_key)
    except TypeError as e:
        if "proxies" in str(e):
            print("[DEBUG] Proxies error in account manager, using workaround...")
            # Create a simple wrapper that works around the issue
            
            class WorkaroundOpenAIClient:
                def __init__(self, api_key):
                    self.api_key = api_key
                    self.base_url = "https://api.openai.com/v1"
                    self.chat = self
                    self.completions = self
                
                def create(self, **kwargs):
                    # Make direct HTTP request to OpenAI API
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    # Convert parameters to API format
                    data = {
                        "model": kwargs.get("model", "gpt-4"),
                        "messages": kwargs.get("messages", []),
                        "temperature": kwargs.get("temperature", 0.7)
                    }
                    
                    response = requests.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=60.0
                    )
                    
                    if response.status_code != 200:
                        raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
                    
                    result = response.json()
                    
                    # Create a simple response object that mimics OpenAI's response
                    class SimpleResponse:
                        def __init__(self, data):
                            self.choices = [SimpleChoice(data['choices'][0])]
                            # Add usage tracking
                            if 'usage' in data:
                                self.usage = SimpleUsage(data['usage'])
                            else:
                                # Fallback usage estimation
                                self.usage = SimpleUsage({
                                    'total_tokens': 100,
                                    'prompt_tokens': 50,
                                    'completion_tokens': 50
                                })
                    
                    class SimpleChoice:
                        def __init__(self, choice_data):
                            self.message = SimpleMessage(choice_data['message'])
                    
                    class SimpleMessage:
                        def __init__(self, message_data):
                            self.content = message_data['content']
                    
                    class SimpleUsage:
                        def __init__(self, usage_data):
                            self.total_tokens = usage_data.get('total_tokens', 100)
                            self.prompt_tokens = usage_data.get('prompt_tokens', 50)
                            self.completion_tokens = usage_data.get('completion_tokens', 50)
                    
                    return SimpleResponse(result)
            
            return WorkaroundOpenAIClient(api_key)
        else:
            raise e

def generate_with_account_switching(prompt, system_prompt, account_manager, model="gpt-3.5-turbo"):
    """
    Generate text using OpenAI API with automatic account switching on quota errors.
    Returns the generated text and the account ID used.
    Logs errors and warnings for failed generation and account switching.
    """
    success = False
    attempts = 0
    max_attempts = account_manager.total_accounts
    last_error = None

    while not success and attempts < max_attempts:
        account = account_manager.get_next_account()
        api_key = account['api_key']
        account_id = account['id']
        
        try:
            client = create_openai_client(api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Update account usage with detailed token tracking
            tokens_used = response.usage.total_tokens
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            account_manager.update_account_usage(account_id, tokens_used)
            
            print(f'[SUCCESS] OpenAI generation successful using account {account_id}')
            print(f'[TOKENS] Total: {tokens_used}, Prompt: {prompt_tokens}, Completion: {completion_tokens}')
            
            return response.choices[0].message.content, account_id
            
        except Exception as e:
            last_error = e
            error_message = str(e).lower()
            
            # Categorize different types of errors
            if ('insufficient_quota' in error_message or 
                'rate_limit' in error_message or 
                'quota' in error_message or
                'billing' in error_message or
                'exceeded' in error_message):
                print(f'[WARNING] Quota/Rate limit error for account {account_id}, switching account.')
                print(f'[WARNING] Error details: {str(e)[:100]}...')
                account_manager.last_account_index = account_id % account_manager.total_accounts
                attempts += 1
            elif ('api_key' in error_message or 
                  'authentication' in error_message or
                  'unauthorized' in error_message):
                print(f'[ERROR] Authentication error for account {account_id}: {e}')
                print(f'[ERROR] Skipping invalid account and continuing...')
                attempts += 1
            elif ('connection' in error_message or 
                  'timeout' in error_message or
                  'network' in error_message):
                print(f'[WARNING] Network error for account {account_id}, retrying with next account.')
                attempts += 1
            else:
                # For other errors, raise immediately (e.g., invalid model, malformed request)
                print(f'[ERROR] Non-recoverable error for account {account_id}: {e}')
                raise e

    # If we get here, all accounts failed with quota errors
    if not success:
        print(f'[ERROR] Failed to generate after trying all {max_attempts} accounts.')
        raise Exception(f"Failed to generate after trying all accounts. Last error: {last_error}")

def update_accounts_usage_from_dict(account_manager, account_usage):
    """
    Updates the usage stats for each account in account_usage dict.
    Calls account_manager.update_account_usage for each account_id and tokens_used.
    """
    for account_id, tokens_used in account_usage.items():
        account_manager.update_account_usage(account_id, tokens_used) 