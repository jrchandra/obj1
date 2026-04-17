import argparse
import itertools
import numpy as np
import pandas as pd


def cohen_kappa(a, b, labels):
    # a,b are arrays of same length containing labels
    a = np.array(a)
    b = np.array(b)
    mask = (~pd.isna(a)) & (~pd.isna(b))
    a = a[mask]
    b = b[mask]
    if len(a) == 0:
        return np.nan

    lab2i = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=float)
    for x, y in zip(a, b):
        if x not in lab2i or y not in lab2i:
            continue
        cm[lab2i[x], lab2i[y]] += 1

    total = cm.sum()
    if total == 0:
        return np.nan

    po = np.trace(cm) / total
    pe = (cm.sum(axis=1) / total @ (cm.sum(axis=0) / total))
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def weighted_kappa(a, b, labels, weight_type="quadratic"):
    # ordinal labels in increasing order
    a = np.array(a)
    b = np.array(b)
    mask = (~pd.isna(a)) & (~pd.isna(b))
    a = a[mask]
    b = b[mask]
    if len(a) == 0:
        return np.nan

    lab2i = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)

    O = np.zeros((n, n), dtype=float)
    for x, y in zip(a, b):
        if x not in lab2i or y not in lab2i:
            continue
        O[lab2i[x], lab2i[y]] += 1

    total = O.sum()
    if total == 0:
        return np.nan

    # Expected matrix
    row = O.sum(axis=1) / total
    col = O.sum(axis=0) / total
    E = np.outer(row, col) * total

    # Weights
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            d = abs(i - j)
            if weight_type == "linear":
                W[i, j] = d / (n - 1) if n > 1 else 0
            else:  # quadratic
                W[i, j] = (d ** 2) / ((n - 1) ** 2) if n > 1 else 0

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return np.nan
    return 1 - (num / den)


def explode_types(s: str):
    if s is None:
        return []
    s = str(s).strip().upper()
    if s in ("", "NAN", "NONE"):
        return []
    return [x.strip() for x in s.split(";") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="Merged annotations CSV")
    ap.add_argument("--outfile", required=True, help="Agreement report CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.infile, encoding="utf-8")
    df["key"] = df["seg_id"].astype(str) + "||" + df["system_masked"].astype(str)

    annotators = sorted(df["annotator"].unique().tolist())
    if len(annotators) < 2:
        raise ValueError("Need at least 2 annotators.")

    # Pivot for error_present and severity
    piv_err = df.pivot_table(index="key", columns="annotator", values="error_present", aggfunc="first")
    piv_sev = df.pivot_table(index="key", columns="annotator", values="severity", aggfunc="first")
    piv_types = df.pivot_table(index="key", columns="annotator", values="error_types", aggfunc="first")

    # Define label sets
    err_labels = ["N", "Y"]
    sev_labels = ["MINOR", "MAJOR", "CRITICAL"]

    # For types: compute one-vs-rest kappa per type
    ALL_TYPES = ["LEXICAL", "SYNTACTIC", "PRAGMATIC", "CULTURAL"]

    rows = []
    for a, b in itertools.combinations(annotators, 2):
        k_err = cohen_kappa(piv_err[a], piv_err[b], err_labels)
        k_sev = weighted_kappa(piv_sev[a], piv_sev[b], sev_labels, weight_type="quadratic")

        rows.append({
            "pair": f"{a} vs {b}",
            "metric": "cohen_kappa_error_present",
            "value": k_err
        })
        rows.append({
            "pair": f"{a} vs {b}",
            "metric": "weighted_kappa_severity_quadratic",
            "value": k_sev
        })

        # type-wise kappas
        for t in ALL_TYPES:
            a_bin = piv_types[a].apply(lambda x: "Y" if t in explode_types(x) else "N")
            b_bin = piv_types[b].apply(lambda x: "Y" if t in explode_types(x) else "N")
            k_t = cohen_kappa(a_bin, b_bin, ["N", "Y"])
            rows.append({
                "pair": f"{a} vs {b}",
                "metric": f"cohen_kappa_type_{t}",
                "value": k_t
            })

    out = pd.DataFrame(rows)
    out.to_csv(args.outfile, index=False, encoding="utf-8")
    print(f"[OK] Wrote agreement report: {args.outfile}")


if __name__ == "__main__":
    main()
