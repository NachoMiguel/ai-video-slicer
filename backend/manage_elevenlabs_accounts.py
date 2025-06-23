#!/usr/bin/env python3
"""
ElevenLabs Account Management Script
===================================

This script helps you manage your ElevenLabs accounts, including:
- Viewing account status and credit balances
- Marking accounts as paid to bypass IP restrictions
- Getting upgrade recommendations
- Managing flagged accounts

Usage:
    python manage_elevenlabs_accounts.py status          # View all account statuses
    python manage_elevenlabs_accounts.py upgrade 1       # Mark account 1 as paid (starter)
    python manage_elevenlabs_accounts.py upgrade 2 creator  # Mark account 2 as paid (creator)
    python manage_elevenlabs_accounts.py unflag 3        # Remove flag from account 3
    python manage_elevenlabs_accounts.py recommend       # Get upgrade recommendations
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from elevenlabs_account_manager import ElevenLabsAccountManager

def show_usage():
    """Display usage information"""
    print(__doc__)

def status_command(manager, skip_validation=False):
    """Show detailed account status"""
    if not skip_validation:
        print("🔍 Validating accounts (this may take a moment)...")
        try:
            valid_count, invalid_accounts = manager.validate_accounts()
            print(f"✅ Validation complete: {valid_count} valid accounts")
            if invalid_accounts:
                print(f"❌ Marked {len(invalid_accounts)} accounts as inactive: {invalid_accounts}")
        except Exception as e:
            print(f"⚠️  Validation failed: {e}")
    else:
        print("⚡ Skipping API validation (showing cached status)")
    
    # Show detailed status
    manager.print_account_status()
    
    # Show recommendations
    recommend_command(manager)

def upgrade_command(manager, account_id, plan_type='starter'):
    """Upgrade an account to paid status"""
    try:
        account_id = int(account_id)
        
        # Validate plan type
        valid_plans = ['starter', 'creator', 'pro', 'scale', 'business']
        if plan_type.lower() not in valid_plans:
            print(f"❌ Invalid plan type '{plan_type}'. Valid options: {', '.join(valid_plans)}")
            return
        
        # Check if account exists
        account_found = False
        for account in manager.accounts:
            if account['id'] == account_id:
                account_found = True
                break
        
        if not account_found:
            print(f"❌ Account {account_id} not found")
            return
        
        # Mark as paid
        manager.set_account_as_paid(account_id, plan_type.lower())
        print(f"🎉 Account {account_id} successfully marked as paid ({plan_type} plan)")
        print(f"💡 Remember to actually upgrade this account at: https://elevenlabs.io/pricing")
        print(f"💰 {plan_type.title()} plan cost: ${get_plan_cost(plan_type)}/month")
        
    except ValueError as e:
        print(f"❌ Error: {e}")

def downgrade_command(manager, account_id):
    """Mark an account as free"""
    try:
        account_id = int(account_id)
        manager.set_account_as_free(account_id)
        print(f"📉 Account {account_id} marked as free")
        
    except ValueError as e:
        print(f"❌ Error: {e}")

def unflag_command(manager, account_id):
    """Remove flag from an account"""
    try:
        account_id = int(account_id)
        manager.unflag_account(account_id)
        print(f"🔓 Account {account_id} unflagged")
        
    except ValueError as e:
        print(f"❌ Error: {e}")

def reactivate_command(manager, account_id):
    """Reactivate an inactive account"""
    try:
        account_id = int(account_id)
        manager.reactivate_account(account_id)
        print(f"🔄 Account {account_id} reactivated")
        print(f"💡 Make sure you have added the API key and email for this account to your .env file")
        
    except ValueError as e:
        print(f"❌ Error: {e}")

def recommend_command(manager):
    """Provide upgrade recommendations"""
    summary = manager.get_account_status_summary()
    
    print("\n" + "="*60)
    print("💡 UPGRADE RECOMMENDATIONS")
    print("="*60)
    
    if summary['paid_accounts'] > 0:
        print(f"✅ You have {summary['paid_accounts']} paid account(s) - great for bypassing IP restrictions!")
    else:
        print("⚠️  You have no paid accounts - consider upgrading to avoid IP/VPN issues")
    
    # Find best candidates for upgrade
    free_accounts = summary['free_account_details']
    if free_accounts:
        # Sort by credits remaining
        free_accounts.sort(key=lambda x: x['credits_remaining'], reverse=True)
        
        print(f"\n🎯 RECOMMENDED ACCOUNTS TO UPGRADE:")
        for i, acc in enumerate(free_accounts[:3]):  # Show top 3
            print(f"   {i+1}. Account {acc['id']}: {acc['email']} - {acc['credits_remaining']:,} credits remaining")
        
        print(f"\n💰 COST ANALYSIS:")
        print(f"   Starter Plan: $5/month (first month $1) - 30k credits/month")
        print(f"   Creator Plan: $22/month (first month $11) - 100k credits/month")
        print(f"   Pro Plan: $99/month - 500k credits/month")
        
        print(f"\n🚀 QUICK START:")
        print(f"   1. Upgrade your best account: python manage_elevenlabs_accounts.py upgrade {free_accounts[0]['id']}")
        print(f"   2. Go to https://elevenlabs.io/pricing and actually purchase the plan")
        print(f"   3. Test with a small script to verify IP restrictions are lifted")
        
    if summary['flagged_accounts'] > 0:
        print(f"\n🚫 FLAGGED ACCOUNTS ({summary['flagged_accounts']}):")
        for acc in summary['flagged_account_details']:
            reason = acc.get('flag_reason', 'Unknown')
            print(f"   Account {acc['id']}: {reason}")
            if 'vpn' in reason.lower() or 'proxy' in reason.lower():
                print(f"      💡 Solution: Upgrade account {acc['id']} to paid plan")
                print(f"      🔧 Command: python manage_elevenlabs_accounts.py upgrade {acc['id']}")
    
    print("="*60)

def get_plan_cost(plan_type):
    """Get the monthly cost for a plan type"""
    costs = {
        'starter': 5,
        'creator': 22,
        'pro': 99,
        'scale': 330,
        'business': 1320
    }
    return costs.get(plan_type.lower(), 'Unknown')

def test_account_command(manager, account_id):
    """Test a specific account"""
    try:
        account_id = int(account_id)
        
        # Find the account
        account_found = None
        for account in manager.accounts:
            if account['id'] == account_id:
                account_found = account
                break
        
        if not account_found:
            print(f"❌ Account {account_id} not found")
            return
        
        api_key = os.getenv(f"ELEVENLABS_API_KEY_{account_id}")
        email = os.getenv(f"ELEVENLABS_EMAIL_{account_id}")
        
        if not api_key or not email:
            print(f"❌ Account {account_id} missing API key or email in .env file")
            return
        
        print(f"🧪 Testing account {account_id} ({email})...")
        
        # Test API key
        import requests
        headers = {'xi-api-key': api_key}
        response = requests.get('https://api.elevenlabs.io/v1/user', headers=headers, timeout=10)
        
        if response.status_code == 200:
            user_data = response.json()
            plan_info = user_data.get('subscription', {})
            tier = plan_info.get('tier', 'free')
            
            print(f"✅ Account {account_id} is working!")
            print(f"   📧 Email: {email}")
            print(f"   💳 Plan: {tier}")
            print(f"   🔢 Credits: {plan_info.get('character_limit', 'Unknown')}")
            
            if tier != 'free':
                print(f"   🎉 This is a PAID account - bypasses IP restrictions!")
            else:
                print(f"   ⚠️  This is a FREE account - subject to IP restrictions")
                
        else:
            print(f"❌ Account {account_id} failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Error: {response.text}")
                
    except ValueError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        show_usage()
        return
    
    command = sys.argv[1].lower()
    
    # Initialize account manager
    try:
        # Try to find the accounts file in current directory or parent directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        accounts_file = os.path.join(current_dir, 'elevenlabs_accounts.json')
        
        if not os.path.exists(accounts_file):
            # Try parent directory
            parent_dir = os.path.dirname(current_dir)
            accounts_file = os.path.join(parent_dir, 'backend', 'elevenlabs_accounts.json')
        
        if os.path.exists(accounts_file):
            manager = ElevenLabsAccountManager(json_path=accounts_file)
        else:
            manager = ElevenLabsAccountManager()  # Use default path
            
    except Exception as e:
        print(f"❌ Failed to initialize account manager: {e}")
        print(f"💡 Make sure you're running from the ai-video-slicer/ root directory")
        print(f"💡 Or that elevenlabs_accounts.json exists in backend/")
        return
    
    # Handle commands
    if command == 'status':
        skip_validation = '--skip-validation' in sys.argv
        status_command(manager, skip_validation)
        
    elif command == 'upgrade':
        if len(sys.argv) < 3:
            print("❌ Usage: python manage_elevenlabs_accounts.py upgrade <account_id> [plan_type]")
            return
        account_id = sys.argv[2]
        plan_type = sys.argv[3] if len(sys.argv) > 3 else 'starter'
        upgrade_command(manager, account_id, plan_type)
        
    elif command == 'downgrade':
        if len(sys.argv) < 3:
            print("❌ Usage: python manage_elevenlabs_accounts.py downgrade <account_id>")
            return
        account_id = sys.argv[2]
        downgrade_command(manager, account_id)
        
    elif command == 'unflag':
        if len(sys.argv) < 3:
            print("❌ Usage: python manage_elevenlabs_accounts.py unflag <account_id>")
            return
        account_id = sys.argv[2]
        unflag_command(manager, account_id)
        
    elif command == 'reactivate':
        if len(sys.argv) < 3:
            print("❌ Usage: python manage_elevenlabs_accounts.py reactivate <account_id>")
            return
        account_id = sys.argv[2]
        reactivate_command(manager, account_id)
        
    elif command == 'recommend':
        recommend_command(manager)
        
    elif command == 'test':
        if len(sys.argv) < 3:
            print("❌ Usage: python manage_elevenlabs_accounts.py test <account_id>")
            return
        account_id = sys.argv[2]
        test_account_command(manager, account_id)
        
    else:
        print(f"❌ Unknown command: {command}")
        show_usage()

if __name__ == "__main__":
    main() 