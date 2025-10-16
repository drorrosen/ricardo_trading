# Deployment Checklist for Heroku 🚀

## ✅ All Fixes Applied to `app_modular.py`

### 1. CoinGecko API 365-Day Limit Fix ✅
**Location**: Lines 47-65

```python
# Line 47: Changed default from 365 to 180 days
def load_real_data(self, strong_asset: str, weak_asset: str, days: int = 180):

# Lines 57-60: Added automatic capping
if days > 365:
    print(f"⚠️ Requested {days} days exceeds CoinGecko free limit. Using 365 days instead.")
    days = 365
```

**Result**: API calls will never exceed 365 days, preventing the 401 error.

---

### 2. Updated Default Settings ✅
**Location**: Lines 299-307

```python
DEFAULT_SETTINGS = {
    'cointegration_window': 30,
    'correlation_window': 30,
    'beta_window': 30,
    'data_points': 180,      # Changed from 500 to 180 (6 months)
    'min_data_points': 30,   # Changed from 50 to 30 (1 month)
    'max_data_points': 365   # Changed from 2000 to 365 (free tier limit)
}
```

**Result**: Safe defaults that work with free tier.

---

### 3. Updated UI Input Limits ✅
**Location**: Lines 1647-1654

```python
data_points_override = st.number_input(
    "Days of History",
    min_value=30,      # Changed from DEFAULT_SETTINGS['min_data_points']
    max_value=365,     # Changed from DEFAULT_SETTINGS['max_data_points']
    value=180,         # Safe default
    step=30,
    help="⚠️ CoinGecko free tier max: 365 days. Recommended: 180 days (6 months)"
)
```

**Result**: Users can't accidentally request more than 365 days.

---

### 4. Updated Preset Buttons ✅
**Location**: Lines 1660-1671

```python
# BEFORE (OLD CODE):
"180 (6mo)" → 180
"360 (1yr)" → 360
"720 (2yr)" → 720  ❌ EXCEEDS LIMIT!
"1000" → 1000      ❌ EXCEEDS LIMIT!

# AFTER (NEW CODE):
"30 days" → 30
"90 days" → 90
"180 days" → 180
"365 days (Max)" → 365  ✅ WITHIN LIMIT
```

**Result**: All presets are within free tier limits.

---

### 5. Added API Limit Warning in Sidebar ✅
**Location**: Lines 1223-1230

```python
<div class="dashboard-info-item">
    <strong>API Limit:</strong><br/>
    <span class="dashboard-info-value">Max 365 days</span>
</div>

st.warning("ℹ️ CoinGecko free tier: Max 365 days of data per request")
```

**Result**: Users are clearly informed about the limitation.

---

## 📋 Pre-Deployment Verification

### Test Locally First:
```bash
cd d:\PycharmProjects\ricardo
streamlit run app_modular.py
```

### Verify These Work:
- [ ] 30 days preset → Should load successfully
- [ ] 90 days preset → Should load successfully
- [ ] 180 days preset → Should load successfully ✅ (Recommended)
- [ ] 365 days preset → Should load successfully (Max)
- [ ] Manual input > 365 → Should automatically cap at 365
- [ ] ETH/BCH pair → Should work
- [ ] BTC/ADA pair → Should work
- [ ] Any pair with default 180 days → Should work

---

## 🚀 Heroku Deployment Steps

### 1. Ensure Required Files Exist:
- [x] `app_modular.py` (updated with fixes)
- [ ] `requirements.txt` (check if exists)
- [ ] `Procfile` (needed for Heroku)
- [ ] `runtime.txt` (optional, specifies Python version)

### 2. Check Requirements.txt:
Should include:
```txt
streamlit
pandas
numpy
plotly
scipy
requests
openpyxl
xlsxwriter
```

### 3. Create/Update Procfile:
```
web: streamlit run app_modular.py --server.port=$PORT --server.address=0.0.0.0
```

### 4. Git Commands:
```bash
git add app_modular.py
git commit -m "Fix: CoinGecko API 365-day limit + UX improvements"
git push heroku main
```

---

## ⚠️ Important Notes

### Why the Error Happened:
**Old Code** was requesting data without limits:
- Default: 365 days
- Max allowed: 5000 days
- User could select 720, 1000+ days
- **Result**: API returned 401 error

### Why It's Fixed Now:
**New Code** has multiple safeguards:
1. ✅ Default: 180 days (safe)
2. ✅ UI Max: 365 days (can't input more)
3. ✅ Automatic capping: If somehow > 365, caps to 365
4. ✅ All presets ≤ 365 days
5. ✅ Clear warnings about limits

---

## 🧪 Test After Deployment

Once deployed to Heroku, test:

1. **Basic Test**: Analyze ETH/BCH with 180 days → Should work ✅
2. **Max Test**: Try 365 days → Should work ✅  
3. **Quick Test**: Try 30 days → Should work fast ✅

If all pass, deployment successful! 🎉

---

## 📊 Expected Behavior

### On Your Chart:
When you analyze a pair, you'll see:
- **Default**: 180 days of price history
- **Speed**: Faster load times (less data)
- **Quality**: Still statistically significant (6 months is plenty)
- **Reliability**: No more 401 errors

### Data Granularity:
- **30 days**: Hourly data (very detailed)
- **90 days**: Hourly data (good detail)
- **180 days**: Daily data (recommended) ⭐
- **365 days**: Daily data (maximum available)

---

## 🔍 If Error Still Occurs

Check these:

1. **Are you using the NEW code?**
   - Check line 47: Should say `days: int = 180`
   - Check lines 57-60: Should have capping logic

2. **Did you restart Streamlit?**
   - Stop the app (Ctrl+C)
   - Run again: `streamlit run app_modular.py`

3. **Is the input actually ≤ 365?**
   - Look at the number input field
   - Should show "Days of History" with max 365

4. **Check the console output:**
   - Look for: `"Loading data from CoinGecko API for X/Y (ZZZ days)"`
   - ZZZ should be ≤ 365

---

## ✨ Summary

All fixes are in place in `d:\PycharmProjects\ricardo\app_modular.py`:

| Fix | Status | Location |
|-----|--------|----------|
| Default to 180 days | ✅ | Line 47 |
| Cap at 365 days | ✅ | Lines 57-60 |
| UI max 365 | ✅ | Line 1650 |
| Safe presets | ✅ | Lines 1660-1671 |
| API warning | ✅ | Line 1230 |
| Updated defaults | ✅ | Lines 304-306 |

**Ready to deploy to Heroku!** 🚀

