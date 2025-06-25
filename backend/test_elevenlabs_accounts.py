#!/usr/bin/env python3
"""
Test script to validate all ElevenLabs accounts and check rotation logic.
Run this after adding your 14 accounts to the .env file.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from elevenlabs_account_manager import ElevenLabsAccountManager

def test_account_validation():
    """Test all 14 ElevenLabs accounts"""
    safe_print("[?] Testing ElevenLabs Account Validation...")
    safe_print("=" * 50)
    
    # Load environment
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(env_path)
    
    # Initialize account manager
    manager = ElevenLabsAccountManager()
    safe_print(f"[?] Total accounts configured: {manager.total_accounts}")
    
    # Test each account
    valid_count = 0
    invalid_count = 0
    
    for i in range(1, 15):  # Test accounts 1-14
        api_key = os.getenv(f"ELEVENLABS_API_KEY_{i}")
        email = os.getenv(f"ELEVENLABS_EMAIL_{i}")
        
        safe_print(f"\n[?] Testing Account {i}:")
        
        if not api_key:
            safe_print(f"   [ERROR] Missing ELEVENLABS_API_KEY_{i}")
            invalid_count += 1
            continue
            
        if not email:
            safe_print(f"   [ERROR] Missing ELEVENLABS_EMAIL_{i}")
            invalid_count += 1
            continue
            
        safe_print(f"   [?] Email: {email}")
        safe_print(f"   [?] API Key: {api_key[:8]}...{api_key[-4:]}")
        
        # Test API key validity
        try:
            import requests
            headers = {'xi-api-key': api_key}
            response = requests.get('https://api.elevenlabs.io/v1/user', headers=headers, timeout=10)
            
            if response.status_code == 200:
                user_data = response.json()
                safe_print(f"   [SUCCESS] Valid! Credits remaining: {user_data.get('subscription', {}).get('character_limit', 'Unknown')}")
                valid_count += 1
            elif response.status_code == 401:
                safe_print(f"   [ERROR] Invalid API key")
                invalid_count += 1
            else:
                safe_print(f"   [WARNING]  Unknown response: {response.status_code}")
                invalid_count += 1
                
        except Exception as e:
            safe_print(f"   [ERROR] Error testing: {e}")
            invalid_count += 1
    
    safe_print(f"\n[?] SUMMARY:")
    safe_print(f"   [SUCCESS] Valid accounts: {valid_count}")
    safe_print(f"   [ERROR] Invalid accounts: {invalid_count}")
    safe_print(f"   [?] Success rate: {valid_count/14*100:.1f}%")
    
    return valid_count, invalid_count

def test_account_rotation():
    """Test account rotation logic"""
    safe_print("\n[ROTATE] Testing Account Rotation Logic...")
    safe_print("=" * 50)
    
    manager = ElevenLabsAccountManager()
    
    safe_print(f"Starting index: {manager.last_account_index}")
    
    # Test getting next 5 accounts
    for i in range(5):
        try:
            account = manager.get_next_account()
            safe_print(f"Rotation {i+1}: Account {account['id']} ({account['email']})")
        except Exception as e:
            safe_print(f"Error on rotation {i+1}: {e}")
            break
    
    safe_print(f"Final index: {manager.last_account_index}")

def main():
    """Main test function"""
    safe_print("[?] ElevenLabs Account Testing Suite")
    safe_print("=" * 50)
    
    try:
        valid_count, invalid_count = test_account_validation()
        
        if valid_count > 0:
            test_account_rotation()
        else:
            safe_print("\n[ERROR] No valid accounts found. Please check your .env configuration.")
            
    except Exception as e:
        safe_print(f"\n[CRASH] Test failed with error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main()) 