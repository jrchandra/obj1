import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_xlsx", required=True)
    ap.add_argument("--systems", default="chatgpt,gemini,google_translate,microsoft_translate")
    ap.add_argument("--group_cols", default="direction,domain,sentence_type_fine")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    systems = [c.strip() for c in args.systems.split(",") if c.strip()]
    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    group_cols = [c for c in group_cols if c in df.columns]

    out_sheets = {}
    for syscol in systems:
        metric_cols = [c for c in df.columns if c.startswith(syscol + "__")]
        if not metric_cols:
            continue
        use_cols = group_cols + metric_cols
        tmp = df[use_cols].copy()
        agg = tmp.groupby(group_cols).mean(numeric_only=True).reset_index()
        out_sheets[syscol] = agg

    with pd.ExcelWriter(args.out_xlsx, engine="openpyxl") as w:
        for name, sdf in out_sheets.items():
            sdf.to_excel(w, index=False, sheet_name=name[:31])

    print(f"[OK] Wrote: {args.out_xlsx}")

if __name__ == "__main__":
    main()
