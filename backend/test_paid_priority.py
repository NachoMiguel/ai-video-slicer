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

def test_prioritization():
    """Test that paid accounts are prioritized over free accounts"""
    print("🧪 Testing Paid Account Prioritization...")
    print("=" * 50)
    
    # Initialize account manager
    manager = ElevenLabsAccountManager()
    
    # Show current status
    print("📊 Current Account Status:")
    summary = manager.get_account_status_summary()
    print(f"   Total Accounts: {summary['total_accounts']}")
    print(f"   Paid Accounts: {summary['paid_accounts']}")
    print(f"   Free Accounts: {summary['free_accounts']}")
    
    # Test account selection
    print("\n🎯 Testing Account Selection (5 attempts):")
    for i in range(5):
        try:
            account = manager.get_next_account_with_priority()
            account_type = "PAID" if account.get('is_paid', False) else "FREE"
            plan_type = account.get('plan_type', 'free')
            print(f"   Attempt {i+1}: Account {account['id']} - {account_type} ({plan_type})")
        except Exception as e:
            print(f"   Attempt {i+1}: ERROR - {e}")
    
    # Show recommendations if no paid accounts
    if summary['paid_accounts'] == 0:
        print("\n💡 RECOMMENDATION:")
        print("   No paid accounts detected. To bypass IP restrictions:")
        print("   1. Choose your best account to upgrade")
        print("   2. Run: python manage_elevenlabs_accounts.py upgrade <account_id>")
        print("   3. Actually purchase the plan at https://elevenlabs.io/pricing")
        print("   4. Run this test again to verify prioritization")

def main():
    """Main function"""
    try:
        test_prioritization()
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main() 