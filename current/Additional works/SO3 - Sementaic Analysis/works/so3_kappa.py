import argparse
import pandas as pd
from sklearn.metrics import cohen_kappa_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann1_csv", required=True)
    ap.add_argument("--ann2_csv", required=True)
    ap.add_argument("--id_col", default="item_id")
    ap.add_argument(
    "--score_cols",
    default="cand_A__semantic_fidelity_0_4,cand_A__cultural_adequacy_0_4,cand_A__fluency_0_4,"
            "cand_B__semantic_fidelity_0_4,cand_B__cultural_adequacy_0_4,cand_B__fluency_0_4,"
            "cand_C__semantic_fidelity_0_4,cand_C__cultural_adequacy_0_4,cand_C__fluency_0_4,"
            "cand_D__semantic_fidelity_0_4,cand_D__cultural_adequacy_0_4,cand_D__fluency_0_4"
)

    ap.add_argument("--out_csv", required=True, help="Merged file with disagreements flagged")
    args = ap.parse_args()

    a1 = pd.read_csv(args.ann1_csv)
    a2 = pd.read_csv(args.ann2_csv)
    cols = [c.strip() for c in args.score_cols.split(",") if c.strip()]

    for c in [args.id_col] + cols:
        if c not in a1.columns or c not in a2.columns:
            raise ValueError(f"Missing required column in one of the files: {c}")

    m = a1[[args.id_col] + cols].merge(
        a2[[args.id_col] + cols],
        on=args.id_col,
        suffixes=("__ann1", "__ann2"),
        how="inner"
    )

    results = []
    for c in cols:
        s1 = m[f"{c}__ann1"]
        s2 = m[f"{c}__ann2"]
        # drop blanks
        valid = s1.notna() & s2.notna() & (s1.astype(str) != "") & (s2.astype(str) != "")
        if valid.sum() == 0:
            k = None
        else:
            k = cohen_kappa_score(s1[valid], s2[valid])
        results.append((c, valid.sum(), k))

        m[f"{c}__disagree_flag"] = ((s1.astype(str) != s2.astype(str)) & valid).astype(int)

    m.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote merged disagreements: {args.out_csv}")
    print("\nKAPPA SUMMARY")
    for c, n, k in results:
        print(f"- {c}: n={n}, kappa={k}")

if __name__ == "__main__":
    main()
