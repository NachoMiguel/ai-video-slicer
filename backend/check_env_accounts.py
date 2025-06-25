#!/usr/bin/env python3
"""
Quick script to check if all 14 ElevenLabs accounts are configured in .env file
"""

import os
from dotenv import load_dotenv

def check_env_accounts():
    """Check if all 14 accounts have API keys and emails in .env file"""
    safe_print("[?] Checking .env file for ElevenLabs accounts...")
    safe_print("=" * 50)
    
    # Load environment
    load_dotenv()
    
    missing_keys = []
    missing_emails = []
    found_accounts = []
    
    for i in range(1, 15):  # Check accounts 1-14
        api_key = os.getenv(f"ELEVENLABS_API_KEY_{i}")
        email = os.getenv(f"ELEVENLABS_EMAIL_{i}")
        
        status = "[SUCCESS]"
        issues = []
        
        if not api_key:
            missing_keys.append(i)
            issues.append("No API key")
            status = "[ERROR]"
        
        if not email:
            missing_emails.append(i)
            issues.append("No email")
            status = "[ERROR]"
        
        if api_key and email:
            found_accounts.append(i)
            safe_print(f"   {status} Account {i}: {email[:20]}{'...' if len(email) > 20 else ''} | API: {api_key[:8]}...{api_key[-4:]}")
        else:
            safe_print(f"   {status} Account {i}: {' & '.join(issues) if issues else 'OK'}")
    
    safe_print("=" * 50)
    safe_print(f"[?] SUMMARY:")
    safe_print(f"   [SUCCESS] Properly configured: {len(found_accounts)} accounts")
    safe_print(f"   [ERROR] Missing API keys: {len(missing_keys)} accounts")
    safe_print(f"   [ERROR] Missing emails: {len(missing_emails)} accounts")
    
    if missing_keys:
        safe_print(f"\n[?] Missing API keys for accounts: {missing_keys}")
        
    if missing_emails:
        safe_print(f"\n[?] Missing emails for accounts: {missing_emails}")
    
    if len(found_accounts) == 14:
        safe_print(f"\n[DONE] All 14 accounts are properly configured!")
        return True
    else:
        safe_print(f"\n[WARNING]  Only {len(found_accounts)}/14 accounts are configured")
        return False

if __name__ == "__main__":
    check_env_accounts() 