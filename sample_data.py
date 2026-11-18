"""Synthetic 30-row sales dataset (no Customer ID) with the same columns as the original project sample."""
import numpy as np, pandas as pd

def make(seed: int = 7, n: int = 30, days: int = 150) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    prods = {"Laptop": ("Electronics", 55000), "Phone": ("Electronics", 22000), "Headphones": ("Electronics", 3000),
             "Desk": ("Furniture", 9000), "Chair": ("Furniture", 6500), "Notebook": ("Stationery", 120)}
    names = r.choice(list(prods), n)
    df = pd.DataFrame({
        "OrderDate": pd.to_datetime("2026-01-01") + pd.to_timedelta(np.sort(r.integers(0, days, n)), unit="D"),
        "City": r.choice(["Mumbai", "Pune", "Delhi", "Bengaluru", "Chennai"], n),
        "Category": [prods[p][0] for p in names], "Product": names,
        "UnitPrice_INR": [prods[p][1] for p in names],
        "Quantity": r.integers(1, 6, n),
        "DiscountRate": r.choice([0, 0.05, 0.1, 0.15], n),
        "SalesChannel": r.choice(["Online", "Store"], n),
    })
    df["GrossSales_INR"] = df.UnitPrice_INR * df.Quantity
    df["DiscountAmount_INR"] = (df.GrossSales_INR * df.DiscountRate).round(2)
    df["NetRevenue_INR"] = df.GrossSales_INR - df.DiscountAmount_INR
    return df

if __name__ == "__main__":
    make().to_csv("sample_sales.csv", index=False)
    make(seed=3, n=400, days=730).to_csv("sample_sales_2years.csv", index=False)
    print("wrote sample_sales.csv (30 rows) and sample_sales_2years.csv (400 rows, for forecasting)")
