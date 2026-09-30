"""The small repool / liquidation-status files: what happened AFTER a buyout."""
import pandas as pd, os, warnings
from config import DATA
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200, "display.max_columns", 40)

FILES = [
    DATA / "Repool_EBO_Rocktop_March_21.csv",
    DATA / "RT_EBO_Repool_LoanMod.xlsx",
    r"F:\Ekhoni_Lagbe_Na\LiqStatus_count_Rocktop_March21.csv",
    r"F:\Ekhoni_Lagbe_Na\Cape Analytics\EBO007 Cape Analytics Order 20200109_cape_augmented.csv",
]
for f in FILES:
    print("\n" + "=" * 92)
    print(os.path.basename(f), f"({os.path.getsize(f)/1e3:.0f} KB)" if os.path.exists(f) else "MISSING")
    print("=" * 92)
    if not os.path.exists(f):
        continue
    try:
        if f.lower().endswith(".xlsx"):
            xl = pd.ExcelFile(f)
            print("  sheets:", xl.sheet_names)
            for s in xl.sheet_names[:4]:
                d = xl.parse(s)
                print(f"\n  --- {s}: {d.shape} ---")
                print("  cols:", list(d.columns)[:20])
                print(d.head(12).to_string())
        else:
            d = pd.read_csv(f, low_memory=False, nrows=200000)
            print(f"  shape (first 200k): {d.shape}")
            print("  cols:", list(d.columns)[:30])
            print(d.head(12).to_string())
            for c in d.columns[:20]:
                if d[c].dtype == object and d[c].nunique() < 15:
                    print(f"    {c}: {d[c].value_counts(dropna=False).head(8).to_dict()}")
    except Exception as e:
        print("  ERROR:", type(e).__name__, str(e)[:200])
