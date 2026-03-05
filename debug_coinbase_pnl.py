"""
debug_coinbase_pnl.py - v2

Checks pagination and dumps everything we can find.
"""

import os
from collections import defaultdict
from coinbase.rest import RESTClient

API_KEY    = os.environ.get('COINBASE_API_KEY')
API_SECRET = os.environ.get('COINBASE_API_SECRET')

if not API_KEY or not API_SECRET:
    raise SystemExit("Set COINBASE_API_KEY and COINBASE_API_SECRET env vars first.")

client = RESTClient(api_key=API_KEY, api_secret=API_SECRET)

# ============================================================================
# 1. Pagination inspection - what fields does the response actually have?
# ============================================================================
print("="*60)
print("PAGINATION RESPONSE FIELDS")
print("="*60)
response = client.list_orders(order_status='FILLED', product_type='FUTURE', limit=10)
print("Response attributes:")
for attr in dir(response):
    if not attr.startswith('_'):
        try:
            val = getattr(response, attr)
            if not callable(val):
                print(f"  {attr}: {val}")
        except:
            pass

# ============================================================================
# 2. Fetch ALL pages using every possible pagination field
# ============================================================================
print("\n" + "="*60)
print("FETCHING ALL PAGES")
print("="*60)

all_orders = []
cursor = None
page = 0

while True:
    kwargs = dict(order_status='FILLED', product_type='FUTURE', limit=100)
    if cursor:
        kwargs['cursor'] = cursor

    response = client.list_orders(**kwargs)
    orders = response.orders if hasattr(response, 'orders') else []
    all_orders.extend(orders)
    page += 1

    # Print ALL pagination-related fields on this response
    has_next = getattr(response, 'has_next_page', None)
    new_cursor = getattr(response, 'cursor', None)
    print(f"  Page {page}: {len(orders)} orders | has_next_page={has_next} | cursor={new_cursor}")

    # Stop only if clearly no more pages
    if not orders or page > 100:
        break
    if has_next is False:
        break
    if has_next is None and not new_cursor:
        break
    if new_cursor == cursor:  # cursor didn't advance
        break
    cursor = new_cursor
    if not cursor:
        break

print(f"\nTotal orders fetched: {len(all_orders)}")
print(f"Date range: {getattr(all_orders[-1], 'created_time', '?')[:10]} "
      f"to {getattr(all_orders[0], 'created_time', '?')[:10]}")

# ============================================================================
# 3. PnL calculation
# ============================================================================
total_sell = total_buy = total_fees = 0.0
for o in all_orders:
    fv   = float(getattr(o, 'filled_value', 0) or 0)
    fees = float(getattr(o, 'total_fees', 0) or 0)
    side = getattr(o, 'side', '')
    if side == 'SELL':
        total_sell += fv
    elif side == 'BUY':
        total_buy += fv
    total_fees += fees

print(f"\n{'='*50}")
print(f"  SELL value:  ${total_sell:>12,.2f}")
print(f"  BUY  value:  ${total_buy:>12,.2f}")
print(f"  Gross PnL:   ${total_sell - total_buy:>+12,.2f}")
print(f"  Fees:        ${total_fees:>12,.2f}")
print(f"  Net PnL:     ${total_sell - total_buy - total_fees:>+12,.2f}")
print(f"{'='*50}")

# ============================================================================
# 4. Check for unmatched (open) positions in the order history
# ============================================================================
print(f"\n{'='*50}")
print("UNMATCHED SIDES BY PRODUCT (open position indicator)")
print(f"{'='*50}")
by_product = defaultdict(lambda: {'sell_size': 0.0, 'buy_size': 0.0})
for o in all_orders:
    pid  = getattr(o, 'product_id', '?')
    size = float(getattr(o, 'filled_size', 0) or 0)
    side = getattr(o, 'side', '')
    if side == 'SELL':
        by_product[pid]['sell_size'] += size
    elif side == 'BUY':
        by_product[pid]['buy_size'] += size

for pid, d in sorted(by_product.items()):
    diff = d['sell_size'] - d['buy_size']
    status = "BALANCED" if abs(diff) < 0.001 else f"UNMATCHED: {diff:+.4f}"
    print(f"  {pid:<28} sell={d['sell_size']:>8.2f}  buy={d['buy_size']:>8.2f}  {status}")

# ============================================================================
# 5. Account balance
# ============================================================================
print(f"\n{'='*50}")
print("CURRENT ACCOUNT BALANCE")
print(f"{'='*50}")
try:
    accounts = client.get_accounts().accounts
    for acc in accounts:
        currency = getattr(acc, 'currency', '')
        if currency in ['USD', 'USDC']:
            bal = getattr(acc, 'available_balance', {})
            val = bal.get('value', '?') if isinstance(bal, dict) else getattr(bal, 'value', '?')
            print(f"  {currency}: ${float(val):,.2f}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\nDone.")