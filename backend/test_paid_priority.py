#!/usr/bin/env python3
"""
Test script to verify paid account prioritization is working correctly.
"""

import os
import sys
from dotenv import load_dotenv

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from elevenlabs_account_manager import ElevenLabsAccountManager
try:
    from unicode_safe_logger import safe_print
except ImportError:
    from backend.unicode_safe_logger import safe_print

def test_prioritization():
    """Test that paid accounts are prioritized over free accounts"""
    safe_print("[?] Testing Paid Account Prioritization...")
    safe_print("=" * 50)
    
    # Initialize account manager
    manager = ElevenLabsAccountManager()
    
    # Show current status
    safe_print("[?] Current Account Status:")
    summary = manager.get_account_status_summary()
    safe_print(f"   Total Accounts: {summary['total_accounts']}")
    safe_print(f"   Paid Accounts: {summary['paid_accounts']}")
    safe_print(f"   Free Accounts: {summary['free_accounts']}")
    
    # Test account selection
    safe_print("\n[TARGET] Testing Account Selection (5 attempts):")
    for i in range(5):
        try:
            account = manager.get_next_account_with_priority()
            account_type = "PAID" if account.get('is_paid', False) else "FREE"
            plan_type = account.get('plan_type', 'free')
            safe_print(f"   Attempt {i+1}: Account {account['id']} - {account_type} ({plan_type})")
        except Exception as e:
            safe_print(f"   Attempt {i+1}: ERROR - {e}")
    
    # Show recommendations if no paid accounts
    if summary['paid_accounts'] == 0:
        safe_print("\n[?] RECOMMENDATION:")
        safe_print("   No paid accounts detected. To bypass IP restrictions:")
        safe_print("   1. Choose your best account to upgrade")
        safe_print("   2. Run: python manage_elevenlabs_accounts.py upgrade <account_id>")
        safe_print("   3. Actually purchase the plan at https://elevenlabs.io/pricing")
        safe_print("   4. Run this test again to verify prioritization")

def main():
    """Main function"""
    try:
        test_prioritization()
    except Exception as e:
        safe_print(f"[ERROR] Test failed: {e}")

if __name__ == "__main__":
    main() 