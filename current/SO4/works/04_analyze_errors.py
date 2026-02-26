import argparse
import pandas as pd


def explode_types(s: str):
    if s is None:
        return []
    s = str(s).strip().upper()
    if s in ("", "NAN", "NONE"):
        return []
    return [x.strip() for x in s.split(";") if x.strip()]


def majority_vote(series):
    # returns most common non-empty value
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    if len(s) == 0:
        return ""
    return s.value_counts().idxmax()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="Merged annotations CSV")
    ap.add_argument("--outfile_prefix", required=True, help="Prefix for output files")
    args = ap.parse_args()

    df = pd.read_csv(args.infile, encoding="utf-8")
    df["key"] = df["seg_id"].astype(str) + "||" + df["system_masked"].astype(str)

    # Aggregate to a single "final label" per key by majority vote (simple adjudication proxy)
    agg = df.groupby("key").agg({
        "seg_id": "first",
        "domain": "first",
        "sentence_type": "first",
        "direction": "first",
        "system_masked": "first",
        "system_true": "first",
        "error_present": majority_vote,
        "severity": majority_vote,
        "error_types": majority_vote
    }).reset_index(drop=True)

    # Basic rates
    agg["is_error"] = agg["error_present"].astype(str).str.upper().eq("Y")

    # Overall error rate by system
    by_sys = agg.groupby("system_true").agg(
        n=("is_error", "size"),
        error_rate=("is_error", "mean")
    ).reset_index().sort_values("error_rate", ascending=False)

    # By direction/domain/sentence_type
    by_dir = agg.groupby("direction").agg(n=("is_error", "size"), error_rate=("is_error", "mean")).reset_index()
    by_domain = agg.groupby("domain").agg(n=("is_error", "size"), error_rate=("is_error", "mean")).reset_index()
    by_sent = agg.groupby("sentence_type").agg(n=("is_error", "size"), error_rate=("is_error", "mean")).reset_index()

    # Explode error types for distribution
    type_rows = []
    for _, r in agg.iterrows():
        if not r["is_error"]:
            continue
        for t in explode_types(r["error_types"]):
            type_rows.append({
                "system_true": r["system_true"],
                "system_masked": r["system_masked"],
                "direction": r["direction"],
                "domain": r["domain"],
                "sentence_type": r["sentence_type"],
                "severity": r["severity"],
                "error_type": t
            })
    tdf = pd.DataFrame(type_rows)

    # If no errors annotated, still write empty outputs
    if len(tdf) == 0:
        tdf = pd.DataFrame(columns=["system_true","system_masked","direction","domain","sentence_type","severity","error_type"])

    type_by_sys = tdf.groupby(["system_true","error_type"]).size().reset_index(name="count")
    type_by_dir = tdf.groupby(["direction","error_type"]).size().reset_index(name="count")
    type_by_domain = tdf.groupby(["domain","error_type"]).size().reset_index(name="count")
    type_by_sev = tdf.groupby(["severity","error_type"]).size().reset_index(name="count")

    # Save
    by_sys.to_csv(args.outfile_prefix + "__error_rate_by_system.csv", index=False, encoding="utf-8")
    by_dir.to_csv(args.outfile_prefix + "__error_rate_by_direction.csv", index=False, encoding="utf-8")
    by_domain.to_csv(args.outfile_prefix + "__error_rate_by_domain.csv", index=False, encoding="utf-8")
    by_sent.to_csv(args.outfile_prefix + "__error_rate_by_sentence_type.csv", index=False, encoding="utf-8")

    type_by_sys.to_csv(args.outfile_prefix + "__error_types_by_system.csv", index=False, encoding="utf-8")
    type_by_dir.to_csv(args.outfile_prefix + "__error_types_by_direction.csv", index=False, encoding="utf-8")
    type_by_domain.to_csv(args.outfile_prefix + "__error_types_by_domain.csv", index=False, encoding="utf-8")
    type_by_sev.to_csv(args.outfile_prefix + "__error_types_by_severity.csv", index=False, encoding="utf-8")

    # Quick markdown summary (easy to paste into thesis)
    with open(args.outfile_prefix + "__summary.md", "w", encoding="utf-8") as f:
        f.write("# SO4 Error Analysis Summary\n\n")
        f.write("## Error rate by system\n\n")
        f.write(by_sys.to_markdown(index=False))
        f.write("\n\n## Error rate by direction\n\n")
        f.write(by_dir.to_markdown(index=False))
        f.write("\n\n## Error rate by domain\n\n")
        f.write(by_domain.to_markdown(index=False))
        f.write("\n\n## Error rate by sentence type\n\n")
        f.write(by_sent.to_markdown(index=False))
        f.write("\n")

    print("[OK] Wrote analysis outputs with prefix:", args.outfile_prefix)


if __name__ == "__main__":
    main()
