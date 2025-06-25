#!/usr/bin/env python3
"""Quick test to verify account configuration"""

from elevenlabs_account_manager import ElevenLabsAccountManager

def main():
    safe_print("[?] Testing Account Configuration...")
    
    try:
        manager = ElevenLabsAccountManager('elevenlabs_accounts.json')
        summary = manager.get_account_status_summary()
        
        safe_print(f"[SUCCESS] ACCOUNT SUMMARY:")
        safe_print(f"   Total Accounts: {summary['total_accounts']}")
        safe_print(f"   Paid Accounts: {summary['paid_accounts']}")
        safe_print(f"   Free Accounts: {summary['free_accounts']}")
        safe_print(f"   Total Credits: {summary['total_credits']:,}")
        
        if summary['paid_accounts'] > 0:
            safe_print(f"\n[?] PAID ACCOUNTS:")
            for acc in summary['paid_account_details']:
                safe_print(f"   Account {acc['id']}: {acc['email']} ({acc['plan_type']}) - {acc['credits_remaining']:,} credits")
        
        if summary['free_accounts'] > 0:
            safe_print(f"\n[?] FREE ACCOUNTS:")
            for acc in summary['free_account_details']:
                safe_print(f"   Account {acc['id']}: {acc['email']} - {acc['credits_remaining']:,} credits")
        
        # Test prioritization
        safe_print(f"\n[TARGET] TESTING PRIORITIZATION:")
        for i in range(3):
            try:
                account = manager.get_next_account_with_priority()
                account_type = "PAID" if account.get('is_paid', False) else "FREE"
                plan = account.get('plan_type', 'free')
                safe_print(f"   Test {i+1}: Account {account['id']} - {account_type} ({plan})")
            except Exception as e:
                safe_print(f"   Test {i+1}: ERROR - {e}")
                break
        
        safe_print(f"\n[DONE] Configuration Test Complete!")
        
    except Exception as e:
        safe_print(f"[ERROR] Error: {e}")

if __name__ == "__main__":
    main() 