import argparse
import os
import pandas as pd


REQUIRED = ["seg_id", "system_masked", "error_present", "error_types", "severity"]


def load_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infiles", nargs="+", required=True, help="Annotator .xlsx files")
    ap.add_argument("--outfile", required=True, help="Merged long table .csv")
    args = ap.parse_args()

    frames = []
    for f in args.infiles:
        df = load_xlsx(f)
        for c in REQUIRED:
            if c not in df.columns:
                raise ValueError(f"{f} missing required column: {c}. Found: {list(df.columns)}")
        df = df[df["seg_id"].astype(str) != "GUIDE"].copy()
        df["annotator"] = os.path.splitext(os.path.basename(f))[0]
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    # Basic normalization
    merged["error_present"] = merged["error_present"].astype(str).str.strip().str.upper()
    merged["severity"] = merged["severity"].astype(str).str.strip().str.upper()
    merged["error_types"] = merged["error_types"].astype(str).str.strip().str.upper()

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    merged.to_csv(args.outfile, index=False, encoding="utf-8")
    print(f"[OK] Wrote merged annotations: {args.outfile}")
    print("[INFO] Annotators:", sorted(merged["annotator"].unique().tolist()))


if __name__ == "__main__":
    main()
