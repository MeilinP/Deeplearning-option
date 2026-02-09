"""
Quick test script to verify Polygon API is working correctly
"""

from data_utils import PolygonOptionsDataLoader
from datetime import datetime, timedelta

print("=" * 60)
print("Testing Polygon.io API Connection")
print("=" * 60)

# Initialize loader
loader = PolygonOptionsDataLoader()
print("✓ API client initialized")

# Test 1: Fetch a few contracts to verify API works
print("\nTest 1: Fetching 5 SPY option contracts...")
try:
    from polygon import RESTClient
    import os
    
    api_key = os.getenv("POLYGON_API_KEY")
    client = RESTClient(api_key)
    
    contracts = client.list_options_contracts(
        underlying_ticker="SPY",
        limit=5
    )
    
    count = 0
    for contract in contracts:
        count += 1
        print(f"  {count}. {contract.ticker} - Expires: {contract.expiration_date}")
        
    print(f"✓ Successfully fetched {count} contracts")
    
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Test 2: Fetch contracts with date range
print("\nTest 2: Fetching SPY options expiring in next 60 days...")
try:
    today = datetime.now()
    end_date = today + timedelta(days=60)
    
    contracts = client.list_options_contracts(
        underlying_ticker="SPY",
        expiration_date_lte=end_date.strftime("%Y-%m-%d"),
        limit=10
    )
    
    count = 0
    for contract in contracts:
        count += 1
        if count <= 3:
            print(f"  {count}. {contract.ticker} - Strike: ${contract.strike_price}, Expires: {contract.expiration_date}")
        
    print(f"✓ Found {count} contracts expiring within 60 days")
    
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

print("\n" + "=" * 60)
print("All tests passed! Polygon API is working correctly.")
print("=" * 60)
print("\nNote: The Polygon Starter plan provides 2 years of historical data.")
print("When fetching options, the date range refers to option EXPIRATION dates,")
print("not trade dates. For backtesting, use a range like:")
print("  - Start: 6-12 months ago")
print("  - End: Today or near future")
