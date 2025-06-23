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

    def get_paid_accounts(self):
        """Get all accounts marked as paid (bypass VPN/IP restrictions)"""
        paid_accounts = []
        for account in self.accounts:
            if (account.get('is_paid', False) and 
                not account.get('flagged', False) and 
                not account.get('inactive', False)):
                api_key = os.getenv(f"ELEVENLABS_API_KEY_{account['id']}")
                email = os.getenv(f"ELEVENLABS_EMAIL_{account['id']}")
                
                if api_key and email:
                    paid_accounts.append({
                        'id': account['id'],
                        'email': email,
                        'api_key': api_key,
                        'credits_used': account['credits_used'],
                        'credits_remaining': account['credits_remaining'],
                        'last_used': account['last_used'],
                        'plan_type': account.get('plan_type', 'starter'),
                        'is_paid': True
                    })
        return paid_accounts

    def get_free_accounts(self):
        """Get all accounts marked as free (subject to VPN/IP restrictions)"""
        free_accounts = []
        for account in self.accounts:
            if (not account.get('is_paid', False) and 
                not account.get('flagged', False) and 
                not account.get('inactive', False)):
                api_key = os.getenv(f"ELEVENLABS_API_KEY_{account['id']}")
                email = os.getenv(f"ELEVENLABS_EMAIL_{account['id']}")
                
                if api_key and email:
                    free_accounts.append({
                        'id': account['id'],
                        'email': email,
                        'api_key': api_key,
                        'credits_used': account['credits_used'],
                        'credits_remaining': account['credits_remaining'],
                        'last_used': account['last_used'],
                        'is_paid': False
                    })
        return free_accounts

    def set_account_as_paid(self, account_id, plan_type='starter'):
        """Mark an account as paid to prioritize it and bypass IP restrictions"""
        for account in self.accounts:
            if account['id'] == account_id:
                account['is_paid'] = True
                account['plan_type'] = plan_type
                account['upgraded_date'] = datetime.now().isoformat()
                self._save_accounts()
                print(f"[SUCCESS] Account {account_id} marked as paid ({plan_type} plan)")
                return
        raise ValueError(f"Account id {account_id} not found.")

    def set_account_as_free(self, account_id):
        """Mark an account as free (subject to IP restrictions)"""
        for account in self.accounts:
            if account['id'] == account_id:
                account['is_paid'] = False
                account.pop('plan_type', None)
                account.pop('upgraded_date', None)
                self._save_accounts()
                print(f"[INFO] Account {account_id} marked as free")
                return
        raise ValueError(f"Account id {account_id} not found.")

    def get_next_account_with_priority(self):
        """Get next account with paid accounts taking priority over free accounts"""
        # First, try to get a paid account
        paid_accounts = self.get_paid_accounts()
        if paid_accounts:
            # Sort by credits remaining (descending) and last used (ascending)
            paid_accounts.sort(key=lambda x: (-x['credits_remaining'], x['last_used']))
            best_paid = paid_accounts[0]
            print(f"[PRIORITY] Using paid account {best_paid['id']} ({best_paid['plan_type']} plan)")
            return best_paid
        
        # If no paid accounts available, fall back to free accounts
        print("[FALLBACK] No paid accounts available, trying free accounts...")
        free_accounts = self.get_free_accounts()
        if free_accounts:
            # Sort by credits remaining (descending) and last used (ascending)
            free_accounts.sort(key=lambda x: (-x['credits_remaining'], x['last_used']))
            best_free = free_accounts[0]
            print(f"[WARNING] Using free account {best_free['id']} - may be subject to IP restrictions")
            return best_free
        
        # If no valid accounts found
        raise ValueError("No valid ElevenLabs accounts available. Please check your API keys and email configuration, or upgrade accounts to paid plans.")

    def get_next_account(self):
        """Get the next valid account with proper validation and error handling."""
        attempts = 0
        max_attempts = self.total_accounts
        
        while attempts < max_attempts:
            next_index = (self.last_account_index + 1) % self.total_accounts
            self.last_account_index = next_index
            account = self.accounts[next_index]
            
            api_key = os.getenv(f"ELEVENLABS_API_KEY_{account['id']}")
            email = os.getenv(f"ELEVENLABS_EMAIL_{account['id']}")
            
            # Skip invalid accounts but continue rotation
            if not api_key or not email:
                print(f"[WARNING] Skipping invalid account {account['id']} - missing API key or email")
                attempts += 1
                continue
            
            # Skip flagged or inactive accounts
            if account.get('flagged', False):
                print(f"[WARNING] Skipping flagged account {account['id']} - reason: {account.get('flag_reason', 'Unknown')}")
                attempts += 1
                continue
            
            if account.get('inactive', False):
                print(f"[WARNING] Skipping inactive account {account['id']} - reason: {account.get('inactive_reason', 'Unknown')}")
                attempts += 1
                continue
            
            # Return valid account
            return {
                'id': account['id'],
                'email': email,
                'api_key': api_key,
                'credits_used': account['credits_used'],
                'credits_remaining': account['credits_remaining'],
                'last_used': account['last_used'],
                'is_paid': account.get('is_paid', False),
                'plan_type': account.get('plan_type', 'free')
            }
        
        # If we get here, no valid accounts found
        raise ValueError("No valid ElevenLabs accounts available. Please check your API keys and email configuration.")

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
    
    def flag_account_as_suspicious(self, account_id, reason=""):
        """Flag an account as having suspicious activity detected by ElevenLabs"""
        for account in self.accounts:
            if account['id'] == account_id:
                account['flagged'] = True
                account['flag_reason'] = reason
                account['flag_date'] = datetime.now().isoformat()
                self._save_accounts()
                print(f"[WARNING] Account {account_id} flagged for suspicious activity: {reason}")
                return
        raise ValueError(f"Account id {account_id} not found.")

    def unflag_account(self, account_id):
        """Remove suspicious activity flag from an account"""
        for account in self.accounts:
            if account['id'] == account_id:
                account['flagged'] = False
                account.pop('flag_reason', None)
                account.pop('flag_date', None)
                self._save_accounts()
                print(f"[INFO] Account {account_id} unflagged")
                return
        raise ValueError(f"Account id {account_id} not found.")

    def reactivate_account(self, account_id):
        """Remove inactive status from an account"""
        for account in self.accounts:
            if account['id'] == account_id:
                account['inactive'] = False
                account.pop('inactive_reason', None)
                self._save_accounts()
                print(f"[INFO] Account {account_id} reactivated")
                return
        raise ValueError(f"Account id {account_id} not found.")

    def get_account_status_summary(self):
        """Get a comprehensive summary of all account statuses"""
        paid_accounts = self.get_paid_accounts()
        free_accounts = self.get_free_accounts()
        flagged_accounts = [acc for acc in self.accounts if acc.get('flagged', False)]
        inactive_accounts = [acc for acc in self.accounts if acc.get('inactive', False)]
        
        total_paid_credits = sum(acc['credits_remaining'] for acc in paid_accounts)
        total_free_credits = sum(acc['credits_remaining'] for acc in free_accounts)
        
        summary = {
            'total_accounts': self.total_accounts,
            'paid_accounts': len(paid_accounts),
            'free_accounts': len(free_accounts),
            'flagged_accounts': len(flagged_accounts),
            'inactive_accounts': len(inactive_accounts),
            'total_paid_credits': total_paid_credits,
            'total_free_credits': total_free_credits,
            'total_credits': total_paid_credits + total_free_credits,
            'paid_account_details': paid_accounts,
            'free_account_details': free_accounts,
            'flagged_account_details': flagged_accounts,
            'inactive_account_details': inactive_accounts
        }
        
        return summary

    def print_account_status(self):
        """Print a formatted account status summary"""
        summary = self.get_account_status_summary()
        
        print("\n" + "="*60)
        print("🔍 ELEVENLABS ACCOUNT STATUS SUMMARY")
        print("="*60)
        print(f"📊 Total Accounts: {summary['total_accounts']}")
        print(f"💳 Paid Accounts: {summary['paid_accounts']} ({summary['total_paid_credits']:,} credits)")
        print(f"🆓 Free Accounts: {summary['free_accounts']} ({summary['total_free_credits']:,} credits)")
        print(f"🚫 Flagged Accounts: {summary['flagged_accounts']}")
        print(f"💤 Inactive Accounts: {summary['inactive_accounts']}")
        print(f"💎 Total Available Credits: {summary['total_credits']:,}")
        
        if summary['paid_account_details']:
            print("\n💳 PAID ACCOUNTS:")
            for acc in summary['paid_account_details']:
                print(f"   Account {acc['id']}: {acc['email']} ({acc['plan_type']}) - {acc['credits_remaining']:,} credits")
        
        if summary['free_account_details']:
            print("\n🆓 FREE ACCOUNTS:")
            for acc in summary['free_account_details']:
                print(f"   Account {acc['id']}: {acc['email']} - {acc['credits_remaining']:,} credits")
        
        if summary['flagged_account_details']:
            print("\n🚫 FLAGGED ACCOUNTS:")
            for acc in summary['flagged_account_details']:
                print(f"   Account {acc['id']}: {acc['email']} - {acc.get('flag_reason', 'Unknown reason')}")
        
        if summary['inactive_account_details']:
            print("\n💤 INACTIVE ACCOUNTS:")
            for acc in summary['inactive_account_details']:
                print(f"   Account {acc['id']}: {acc.get('email', 'No email')} - {acc.get('inactive_reason', 'Unknown reason')}")
        
        print("="*60)

    def get_account_api_key(self, account_id):
        return os.getenv(f"ELEVENLABS_API_KEY_{account_id}")

    def get_account_email(self, account_id):
        return os.getenv(f"ELEVENLABS_EMAIL_{account_id}")
    
    def validate_accounts(self, skip_api_test=False):
        """Validate all accounts and mark invalid ones as inactive"""
        valid_accounts = []
        invalid_accounts = []
        
        print(f"[DEBUG] Starting validation of {len(self.accounts)} accounts...")
        
        for i, account in enumerate(self.accounts):
            account_id = account['id']
            print(f"[DEBUG] Validating account {account_id} ({i+1}/{len(self.accounts)})...")
            
            api_key = os.getenv(f"ELEVENLABS_API_KEY_{account_id}")
            email = os.getenv(f"ELEVENLABS_EMAIL_{account_id}")
            
            print(f"[DEBUG] Account {account_id}: API key {'found' if api_key else 'MISSING'}, Email {'found' if email else 'MISSING'}")
            
            if not api_key or not email:
                print(f"[WARNING] Account {account_id} missing API key or email - marking as inactive")
                account['inactive'] = True
                account['inactive_reason'] = "Missing API key or email in .env file"
                invalid_accounts.append(account_id)
                valid_accounts.append(account)  # Keep the account but mark as inactive
                continue
            
            # Clear any existing inactive status if API key/email are now present
            if account.get('inactive', False) and account.get('inactive_reason') == "Missing API key or email in .env file":
                account['inactive'] = False
                account.pop('inactive_reason', None)
                print(f"[INFO] Account {account_id} reactivated - API key and email now present")
            
            if skip_api_test:
                print(f"[DEBUG] Skipping API test for account {account_id}")
                valid_accounts.append(account)
                continue
            
            # Test the API key with a simple request
            try:
                print(f"[DEBUG] Testing API key for account {account_id}...")
                headers = {'xi-api-key': api_key}
                response = requests.get('https://api.elevenlabs.io/v1/user', headers=headers, timeout=15)
                print(f"[DEBUG] Account {account_id} API response: {response.status_code}")
                
                if response.status_code == 200:
                    user_data = response.json()
                    plan_info = user_data.get('subscription', {})
                    
                    # Auto-detect if account is paid based on API response
                    if plan_info.get('tier') != 'free':
                        account['is_paid'] = True
                        account['plan_type'] = plan_info.get('tier', 'starter')
                        print(f"[SUCCESS] Account {account_id} ({email}) validated - PAID ({account['plan_type']})")
                    else:
                        account['is_paid'] = account.get('is_paid', False)  # Keep existing setting
                        print(f"[SUCCESS] Account {account_id} ({email}) validated - FREE")
                    
                    valid_accounts.append(account)
                elif response.status_code == 401:
                    print(f"[WARNING] Account {account_id} has invalid API key - marking as inactive")
                    account['inactive'] = True
                    account['inactive_reason'] = "Invalid API key"
                    invalid_accounts.append(account_id)
                    valid_accounts.append(account)  # Keep the account but mark as inactive
                else:
                    print(f"[WARNING] Account {account_id} returned status {response.status_code} - keeping but may have issues")
                    valid_accounts.append(account)
                    
            except Exception as e:
                print(f"[WARNING] Could not validate account {account_id}: {e} - keeping in rotation")
                print(f"[DEBUG] Full error details: {type(e).__name__}: {str(e)}")
                valid_accounts.append(account)
                
            # Add a small delay between API calls to avoid rate limiting
            if i < len(self.accounts) - 1:  # Don't sleep after the last account
                import time
                time.sleep(0.5)
        
        # Update accounts list (now includes inactive accounts)
        self.accounts = valid_accounts
        self.total_accounts = len(self.accounts)
        self._save_accounts()
        
        active_accounts = len([acc for acc in valid_accounts if not acc.get('inactive', False)])
        if invalid_accounts:
            print(f"[INFO] Marked {len(invalid_accounts)} accounts as inactive. {active_accounts} active accounts, {self.total_accounts} total accounts.")
        
        return len(valid_accounts), invalid_accounts

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
    Uses paid account prioritization to bypass IP restrictions.
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
            try:
                # Use prioritized account selection (paid accounts first)
                account = account_manager.get_next_account_with_priority()
                api_key = account['api_key']
                account_id = account['id']
                is_paid = account.get('is_paid', False)
                plan_type = account.get('plan_type', 'free')
                
                if is_paid:
                    print(f'[PRIORITY] Using paid account {account_id} ({account["email"]}) - {plan_type} plan for chunk {idx+1}')
                else:
                    print(f'[FALLBACK] Using free account {account_id} ({account["email"]}) for chunk {idx+1}, attempt {attempts+1}/{max_attempts}')
            except ValueError as e:
                print(f'[ERROR] Failed to get next account: {e}')
                break
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
                # Parse error details
                try:
                    error_json = response.json()
                    error_message = error_json.get('detail', '').lower()
                    error_text = str(error_json)
                except Exception:
                    error_message = ''
                    error_text = response.text
                
                # Categorize errors for account switching
                should_switch_account = False
                error_reason = ""
                
                if response.status_code in [401, 403]:
                    should_switch_account = True
                    error_reason = "Authentication/Authorization error (invalid API key)"
                    
                    # Check for suspicious activity flag
                    if 'unusual_activity' in error_message or 'detected_unusual_activity' in error_message:
                        if is_paid:
                            print(f"[CRITICAL] Paid account {account_id} flagged for unusual activity - this should not happen!")
                        account_manager.flag_account_as_suspicious(account_id, "Unusual activity detected by ElevenLabs")
                    
                    # Check for VPN/Proxy detection on free accounts
                    if ('proxy' in error_message or 'vpn' in error_message) and not is_paid:
                        print(f"[IP-BAN] Free account {account_id} blocked due to VPN/Proxy detection")
                        account_manager.flag_account_as_suspicious(account_id, "VPN/Proxy detected - upgrade to paid plan recommended")
                elif response.status_code == 402 or 'insufficient' in error_message or 'credit' in error_message or 'quota' in error_message:
                    should_switch_account = True
                    error_reason = "Credit/Payment/Quota error"
                elif response.status_code == 429:
                    should_switch_account = True
                    error_reason = "Rate limit error"
                elif response.status_code in [500, 502, 503]:
                    should_switch_account = True
                    error_reason = "Server error (temporary)"
                elif response.status_code == 400:
                    should_switch_account = False
                    error_reason = "Bad request (likely voice ID or request format issue)"
                else:
                    should_switch_account = True
                    error_reason = f"Unexpected error ({response.status_code})"
                
                if should_switch_account:
                    print(f'[WARNING] {error_reason} for account {account_id} on chunk {idx+1}, switching account.')
                    print(f'[DEBUG] Error details: {response.status_code} - {error_text[:200]}...')
                    # Fix: Convert account_id (1-based) to array index (0-based) correctly
                    account_manager.last_account_index = (account_id - 1) % account_manager.total_accounts
                    attempts += 1
                else:
                    print(f'[ERROR] {error_reason} for chunk {idx+1}, stopping attempts.')
                    print(f'[ERROR] Full error: {response.status_code} {error_text}')
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