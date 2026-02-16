#!/usr/bin/env python3
"""Test all mock data generators."""
import sys
sys.path.insert(0, '/home/charsree/.workspace/stock-chat-assistant')

from backend.data.mock_generators import get_ohlcv, get_options_chain, get_fundamentals, get_sentiment, get_macro_data

print('Testing price data...')
df = get_ohlcv('TSLA', 'daily', 30)
print(f'  Generated {len(df)} daily bars, price range: ${df.close.min():.2f} - ${df.close.max():.2f}')

print('Testing options data...')
chain = get_options_chain('TSLA', 250)
print(f'  Generated {len(chain)} options contracts')

print('Testing fundamentals...')
fund = get_fundamentals('TSLA')
print(f'  Generated {len(fund["quarterly"])} quarters, PE: {fund["valuation"]["pe_ratio"]}')

print('Testing sentiment...')
sent = get_sentiment('TSLA')
print(f'  Generated {len(sent["news"])} news articles')

print('Testing macro data...')
macro = get_macro_data()
print(f'  Generated {len(macro["vix"])} VIX data points')

print('\nAll mock generators working!')
