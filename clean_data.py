import pandas as pd
import numpy as np
import os

def clean_marketing_file(path, channel_name):
    # Essential: Check if file exists to avoid crashes
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return pd.DataFrame()  # Return empty DataFrame instead of crashing
    
    df = pd.read_csv(path)
    
    # Essential: Check if file is empty
    if df.empty:
        print(f"Warning: Empty file: {path}")
        return df
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Parse date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Normalize campaign/state strings
    for c in ['campaign','tactic','state']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower().replace('nan','')
    # Numeric conversions
    for col in ['impression','clicks','spend','attributed_revenue']:  # Fixed: 'impression' not 'impressions'
        if col in df.columns:
            # Only convert if not already numeric to avoid unnecessary processing
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = (df[col].astype(str)
                           .str.replace(',','')
                           .replace(['','nan','None'], '0')
                           .astype(float, errors='ignore').fillna(0.0))
    
    df['channel'] = channel_name
    return df

# Essential: Error handling for file loading
files = {'facebook': 'data/Facebook.csv', 'google': 'data/Google.csv', 'tiktok': 'data/TikTok.csv'}
dfs = []

for ch, file_path in files.items():
    df = clean_marketing_file(file_path, ch)
    if not df.empty:  # Only add non-empty DataFrames
        dfs.append(df)

if dfs:  # Essential: Check if we have data before concatenating
    marketing = pd.concat(dfs, ignore_index=True)
    print(f"✅ Successfully loaded {len(marketing)} rows from {len(dfs)} files")
    
    # Basic dedupe and aggregate to date/channel/campaign
    marketing = marketing.drop_duplicates()
    group_cols = ['date','channel','campaign']
    num_cols = ['impression','clicks','spend','attributed_revenue']  # Fixed: 'impression' not 'impressions'
    marketing_agg = marketing.groupby(group_cols, dropna=False)[num_cols].sum().reset_index()
    
    # Load business data
    business = pd.read_csv('data/business.csv')  # Fixed: Added 'data/' prefix
    business.columns = business.columns.str.strip().str.lower().str.replace(' ', '_')
    business['date'] = pd.to_datetime(business['date'], errors='coerce')
    
    # Numeric conversions for business columns
    for col in ['orders','new_orders','new_customers','total_revenue','gross_profit','cogs']:
        if col in business.columns:
            if not pd.api.types.is_numeric_dtype(business[col]):  # Only convert if needed
                business[col] = (business[col].astype(str)
                               .str.replace(',','').replace(['','nan','None'], '0')
                               .astype(float, errors='ignore').fillna(0.0))
    
    # Derived metrics on aggregated marketing
    marketing_agg['ctr'] = marketing_agg.apply(lambda r: (r['clicks'] / r['impression']) if r['impression']>0 else np.nan, axis=1)
    marketing_agg['cpc'] = marketing_agg.apply(lambda r: (r['spend'] / r['clicks']) if r['clicks']>0 else np.nan, axis=1)
    marketing_agg['roas'] = marketing_agg.apply(lambda r: (r['attributed_revenue'] / r['spend']) if r['spend']>0 else np.nan, axis=1)
    
    # Quick sanity: totals
    platform_spend = marketing_agg.groupby('channel')['spend'].sum().to_dict()
    total_marketing_spend = marketing_agg['spend'].sum()
    total_business_revenue = business['total_revenue'].sum()
    
    print("Spend by platform:", platform_spend)
    print("Total marketing spend:", total_marketing_spend)
    print("Total business revenue:", total_business_revenue)
    
    # Data consistency checks
    print("\n=== DATA CONSISTENCY CHECKS ===")
    
    # Check date ranges match
    marketing_date_range = (marketing_agg['date'].min(), marketing_agg['date'].max())
    business_date_range = (business['date'].min(), business['date'].max())
    
    print(f"Marketing date range: {marketing_date_range[0]} to {marketing_date_range[1]}")
    print(f"Business date range: {business_date_range[0]} to {business_date_range[1]}")
    
    if marketing_date_range != business_date_range:
        print("⚠️ Warning: Date ranges don't match between marketing and business data")
    else:
        print("✅ Date ranges match")
    
    # 1️⃣ Save campaign-level data (drilldown view)
    marketing_agg.to_csv("marketing_by_campaign.csv", index=False)
    print("✅ Saved marketing_by_campaign.csv")

    # 2️⃣ Create channel-level daily aggregate
    channel_agg = marketing_agg.groupby(['date','channel'], dropna=False)[
        ['impression','clicks','spend','attributed_revenue']
    ].sum().reset_index()

    # Recompute metrics for channel-level with proper NaN handling
    channel_agg['ctr'] = np.where(channel_agg['impression'] > 0, 
                                 channel_agg['clicks'] / channel_agg['impression'], 
                                 np.nan)
    channel_agg['cpc'] = np.where(channel_agg['clicks'] > 0, 
                                 channel_agg['spend'] / channel_agg['clicks'], 
                                 np.nan)
    channel_agg['roas'] = np.where(channel_agg['spend'] > 0, 
                                  channel_agg['attributed_revenue'] / channel_agg['spend'], 
                                  np.nan)

    # 3️⃣ Merge with business data
    master_daily = pd.merge(channel_agg, business, on='date', how='left')

    # 4️⃣ Derived metrics combining marketing + business with proper NaN handling
    master_daily['cac'] = np.where(master_daily['new_customers'] > 0, 
                                  master_daily['spend'] / master_daily['new_customers'], 
                                  np.nan)
    master_daily['cpa'] = np.where(master_daily['orders'] > 0, 
                                  master_daily['spend'] / master_daily['orders'], 
                                  np.nan)
    master_daily['aov'] = np.where(master_daily['orders'] > 0, 
                                  master_daily['total_revenue'] / master_daily['orders'], 
                                  np.nan)
    master_daily['gross_margin_pct'] = np.where(master_daily['total_revenue'] > 0, 
                                               master_daily['gross_profit'] / master_daily['total_revenue'], 
                                               np.nan)

    # 5️⃣ Save final master_daily file
    master_daily.to_csv("master_daily.csv", index=False)
    print(f"✅ Saved master_daily.csv with {len(master_daily)} rows")
    
else:
    print("❌ No data files were loaded successfully")
    marketing = pd.DataFrame()
    business = pd.DataFrame()

