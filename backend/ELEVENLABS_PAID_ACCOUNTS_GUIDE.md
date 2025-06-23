# ElevenLabs Paid Account Prioritization System

## 🚫 Problem: IP/VPN Restrictions

ElevenLabs has implemented strict abuse detection that blocks free accounts when:
- Using VPN/Proxy connections
- Multiple requests from same IP address
- Suspicious activity patterns detected

**Error Message:**
```
"Unusual activity detected. Free Tier usage disabled. If you are using a proxy/VPN you might need to purchase a Paid Plan to not trigger our abuse detectors."
```

## ✅ Solution: Paid Account Prioritization

This system automatically prioritizes paid ElevenLabs accounts, which bypass IP/VPN restrictions entirely.

### Key Benefits:
- **🔓 Bypasses IP restrictions**: Paid accounts work with VPN/proxy
- **💰 Cost-effective**: Starting at $5/month ($1 first month)
- **🎯 Smart fallback**: Uses free accounts only when needed
- **📊 Easy management**: Simple commands to upgrade accounts
- **🛡️ Reliable**: No more "unusual activity" errors

## 🚀 Quick Start Guide

### Step 1: Check Current Status
```bash
cd backend
python manage_elevenlabs_accounts.py status
```

This shows:
- Current account statuses
- Credit balances
- Upgrade recommendations
- Flagged accounts

### Step 2: Choose Account to Upgrade
Based on the status output, pick your best account (most credits remaining).

### Step 3: Mark Account as Paid (in system)
```bash
# Mark account 1 as paid (starter plan)
python manage_elevenlabs_accounts.py upgrade 1

# Or specify plan type
python manage_elevenlabs_accounts.py upgrade 1 creator
```

### Step 4: Actually Purchase the Plan
1. Go to [ElevenLabs Pricing](https://elevenlabs.io/pricing)
2. Login to the account you marked as paid
3. Purchase the plan (Starter recommended: $5/month, $1 first month)

### Step 5: Test the System
```bash
# Test prioritization is working
python test_paid_priority.py

# Test specific account
python manage_elevenlabs_accounts.py test 1
```

## 📋 Available Commands

### Account Management
```bash
# View all account statuses and recommendations
python manage_elevenlabs_accounts.py status

# Mark account as paid
python manage_elevenlabs_accounts.py upgrade <account_id> [plan_type]

# Mark account as free
python manage_elevenlabs_accounts.py downgrade <account_id>

# Remove suspicious activity flag
python manage_elevenlabs_accounts.py unflag <account_id>

# Test specific account
python manage_elevenlabs_accounts.py test <account_id>

# Get upgrade recommendations
python manage_elevenlabs_accounts.py recommend
```

### Testing
```bash
# Test paid account prioritization
python test_paid_priority.py

# Test all accounts (existing script)
python test_elevenlabs_accounts.py
```

## 💰 Cost Analysis

| Plan | Monthly Cost | First Month | Credits/Month | Best For |
|------|-------------|-------------|---------------|----------|
| Starter | $5 | $1 | 30,000 | Testing & small projects |
| Creator | $22 | $11 | 100,000 | Regular usage |
| Pro | $99 | $99 | 500,000 | Heavy usage |

**Recommendation**: Start with **Starter Plan** for 1-2 accounts.

## 🔧 How It Works

### 1. Account Prioritization
The system now uses `get_next_account_with_priority()` which:
1. **First**: Tries paid accounts (sorted by credits remaining)
2. **Fallback**: Uses free accounts only if no paid accounts available

### 2. Enhanced Error Handling
- Detects VPN/proxy-related errors
- Automatically flags problematic accounts
- Provides specific recommendations for flagged accounts

### 3. Smart Account Selection
- Paid accounts bypass IP restrictions completely
- Free accounts used as backup only
- Accounts with most credits prioritized

## 📊 System Integration

The prioritization is automatically used in:
- `synthesize_chunks_with_account_switching()` - Main TTS function
- Video processing pipeline
- All ElevenLabs API calls

### Before (Free Accounts Only):
```
❌ Account 1: VPN detected - blocked
❌ Account 2: VPN detected - blocked  
❌ Account 3: VPN detected - blocked
🚫 All accounts blocked - processing fails
```

### After (With Paid Accounts):
```
✅ Account 1: PAID - bypasses VPN restrictions
✅ Processing continues successfully
💡 Free accounts available as backup
```

## 🛠️ Technical Details

### New Account Properties
Each account now tracks:
```json
{
  "id": 1,
  "email": "user@example.com",
  "is_paid": true,
  "plan_type": "starter",
  "upgraded_date": "2025-01-20T10:30:00",
  "flagged": false,
  "flag_reason": null
}
```

### Enhanced Functions
- `get_paid_accounts()` - Get all paid accounts
- `get_free_accounts()` - Get all free accounts  
- `set_account_as_paid()` - Mark account as paid
- `get_next_account_with_priority()` - Smart account selection
- `get_account_status_summary()` - Comprehensive status
- `print_account_status()` - Formatted status display

## 🚨 Troubleshooting

### "No valid accounts available"
**Solution**: Upgrade at least one account to paid
```bash
python manage_elevenlabs_accounts.py upgrade 1
```

### "Account still getting VPN errors"
**Possible causes**:
1. Account not actually upgraded on ElevenLabs website
2. Account not marked as paid in system
3. Using flagged account

**Solution**:
```bash
# Check account status
python manage_elevenlabs_accounts.py test 1

# Mark as paid if needed
python manage_elevenlabs_accounts.py upgrade 1

# Unflag if needed
python manage_elevenlabs_accounts.py unflag 1
```

### "Paid account being flagged"
This shouldn't happen with genuine paid accounts. If it does:
1. Contact ElevenLabs support
2. Check if account was properly upgraded
3. Verify API key is correct

## 📈 Monitoring & Maintenance

### Regular Checks
```bash
# Weekly account health check
python manage_elevenlabs_accounts.py status

# Test prioritization is working
python test_paid_priority.py
```

### Account Rotation
The system automatically:
- Prioritizes accounts with more credits
- Tracks usage across accounts  
- Flags problematic accounts
- Provides upgrade recommendations

## 💡 Best Practices

1. **Start Small**: Upgrade 1-2 accounts initially
2. **Monitor Usage**: Check status regularly
3. **Keep Backups**: Maintain some free accounts as backup
4. **Test Changes**: Use test scripts after modifications
5. **Track Costs**: Monitor monthly spending vs. usage

## 🔗 Related Links

- [ElevenLabs Pricing](https://elevenlabs.io/pricing)
- [ElevenLabs API Documentation](https://elevenlabs.io/docs)
- [IP Ban Bypass Guide](https://github.com/helenoixuc/elvn-ipb)

## 📞 Support

If you encounter issues:
1. Run `python manage_elevenlabs_accounts.py status` for diagnostics
2. Check this guide for common solutions
3. Test with `python test_paid_priority.py`
4. Review error logs for specific error messages

---

**🎉 Success Indicator**: When you see `[PRIORITY] Using paid account X` in your logs, the system is working correctly and bypassing IP restrictions! 