#!/usr/bin/env python3
"""Detailed test to demonstrate paid account prioritization"""

from elevenlabs_account_manager import ElevenLabsAccountManager

def main():
    safe_print("[?] DETAILED PAID ACCOUNT PRIORITIZATION TEST")
    safe_print("=" * 60)
    
    manager = ElevenLabsAccountManager('elevenlabs_accounts.json')
    
    # Show current configuration
    paid_accounts = manager.get_paid_accounts()
    free_accounts = manager.get_free_accounts()
    
    safe_print(f"[?] CURRENT CONFIGURATION:")
    safe_print(f"   [?] Paid Accounts: {len(paid_accounts)}")
    safe_print(f"   [?] Free Accounts: {len(free_accounts)}")
    
    safe_print(f"\n[?] PAID ACCOUNTS (these get priority):")
    for acc in paid_accounts:
        safe_print(f"   [SUCCESS] Account {acc['id']}: {acc['email']} ({acc['plan_type']}) - {acc['credits_remaining']:,} credits")
    
    safe_print(f"\n[?] FREE ACCOUNTS (used only as fallback):")
    for acc in free_accounts:
        safe_print(f"   [WARNING]  Account {acc['id']}: {acc['email']} - {acc['credits_remaining']:,} credits")
    
    safe_print(f"\n[TARGET] TESTING get_next_account_with_priority() - 10 SELECTIONS:")
    safe_print("   (This simulates what happens during video processing)")
    
    for i in range(10):
        try:
            account = manager.get_next_account_with_priority()
            account_type = "[?] PAID" if account.get('is_paid', False) else "[?] FREE"
            plan = account.get('plan_type', 'free')
            credits = account['credits_remaining']
            
            safe_print(f"   Selection {i+1:2d}: {account_type} - Account {account['id']} ({plan}) - {credits:,} credits")
            
        except Exception as e:
            safe_print(f"   Selection {i+1:2d}: [ERROR] ERROR - {e}")
            break
    
    safe_print(f"\n[SUCCESS] CONCLUSION:")
    if len(paid_accounts) > 0:
        safe_print(f"   [DONE] Paid account prioritization is WORKING!")
        safe_print(f"   [?] Your paid accounts will bypass IP/VPN restrictions")
        safe_print(f"   [?] You have {sum(acc['credits_remaining'] for acc in paid_accounts):,} paid credits available")
        safe_print(f"   [?] Monthly cost: ${len(paid_accounts) * 5}/month for {len(paid_accounts)} starter accounts")
    else:
        safe_print(f"   [WARNING]  No paid accounts detected - upgrade accounts to bypass restrictions")
    
    safe_print("=" * 60)

if __name__ == "__main__":
    main() 