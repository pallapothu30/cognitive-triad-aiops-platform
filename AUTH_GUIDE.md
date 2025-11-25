# Cognitive Triad - Authentication Guide

## How Authentication Works

This application uses **Emergent Managed Google OAuth** for secure authentication.

### Authentication Flow:

1. **User clicks "Get Started with Google"** on landing page
2. **Redirects to Emergent Auth**: `https://auth.emergentagent.com/?redirect=<dashboard_url>`
3. **User signs in with Google** (managed by Emergent)
4. **Callback to dashboard** with `session_id` in URL fragment
5. **Exchange session_id** for session_token via backend
6. **Store session_token** in httpOnly cookie (7-day expiry)
7. **User accesses dashboard** with authenticated session

## Common Errors & Solutions

### "Invalid state parameter"
**Cause:** OAuth flow was interrupted or expired
**Solution:** 
- Go back to landing page and try login again
- Clear browser cookies and cache
- Ensure you complete the Google sign-in without closing the window

### "Authentication failed: User data not found or expired"
**Cause:** session_id expired or was already used
**Solution:**
- Session IDs are single-use and expire quickly
- Return to landing page and initiate fresh login

### "Missing X-Session-ID header"
**Cause:** Direct access to session endpoint without proper callback
**Solution:**
- Don't bookmark the callback URL
- Always start from landing page

## Testing Authentication

### Manual Testing:
1. Visit: https://cognitive-triad.preview.emergentagent.com
2. Click "Get Started with Google"
3. Complete Google OAuth
4. You should land on dashboard with stats visible

### For Development:
Use the test script in `/app/auth_testing.md` to create test users directly in MongoDB.

## Session Management

- **Session Duration:** 7 days
- **Storage:** httpOnly cookies (secure)
- **Logout:** Deletes session from database and clears cookie
- **Auto-refresh:** Dashboard stats update after each action

## Security Features

✅ HttpOnly cookies prevent XSS attacks
✅ SameSite=None with Secure flag for CORS
✅ Session tokens stored with timezone-aware expiry
✅ Single-use session_id prevents replay attacks
✅ 7-day auto-logout for security
