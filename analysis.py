# analysis.py

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import chi2_contingency, kruskal
from scikit_posthocs import posthoc_dunn

# --- 0) Configuration & Setup ---


def setup_logging(log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"Logging initialized. Log file at: {log_file}")


def load_config(config_path):
    logging.info(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def snake_case_string(s):
    if not isinstance(s, str):
        return s
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s


def snake_case_columns(df):
    df.columns = [snake_case_string(col) for col in df.columns]
    return df


# --- 2) Load Data ---


def load_data(args):
    logging.info(f"Loading data from local file: {args.input}")
    input_path = Path(args.input)
    if not input_path.exists():
        logging.error(f"Input file not found at: {input_path}")
        sys.exit(1)
    try:
        if input_path.suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(input_path, engine="openpyxl")
        elif input_path.suffix == ".csv":
            df = pd.read_csv(input_path)
        elif input_path.suffix == ".ods":
            df = pd.read_excel(input_path, engine="odf")
        else:
            logging.error(f"Unsupported file type: {input_path.suffix}")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to load data file: {e}")
        sys.exit(1)
    df = snake_case_columns(df)
    logging.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


# --- 3) Validate Schema & Types ---


def validate_and_clean(df, config):
    logging.info("Starting schema validation and type coercion.")
    group_col_sc = snake_case_string(config["group_columns"]["year"])
    if group_col_sc not in df.columns:
        logging.error(f"Grouping column '{group_col_sc}' not found in data. Exiting.")
        sys.exit(1)

    df[group_col_sc] = df[group_col_sc].astype(str).str.extract(r"(\d+)", expand=False)
    df[group_col_sc] = pd.to_numeric(df[group_col_sc], errors="coerce")
    df[group_col_sc] = df[group_col_sc].astype("category")
    logging.info(
        f"Value counts for grouping column '{group_col_sc}' (after cleaning):\n{df[group_col_sc].value_counts(dropna=False).to_string()}"
    )

    all_domain_items_sc = [
        snake_case_string(item)
        for domain in config.get("domains", {}).values()
        for item in domain.get("items", [])
    ]
    nominal_items_sc = [
        snake_case_string(s) for s in config.get("nominal_evidence", {}).values()
    ]
    yes_vals = config.get("nominal_yes_values", ["Yes"])
    no_vals = config.get("nominal_no_values", ["No"])
    for item_sc in nominal_items_sc:
        if item_sc in df.columns:
            df[item_sc] = df[item_sc].apply(
                lambda x: 1 if x in yes_vals else (0 if x in no_vals else np.nan)
            )

    valid_likert = config.get("likert_valid_values", [1, 2, 3, 4, 5])
    for item_sc in all_domain_items_sc:
        if item_sc in df.columns:
            df[item_sc] = pd.to_numeric(df[item_sc], errors="coerce")
            df.loc[~df[item_sc].isin(valid_likert), item_sc] = np.nan
    logging.info("Validation and cleaning complete.")
    return df


# --- 4) De-duplication & QA ---


def deduplicate_and_qa(df, config, outdir):
    logging.info("Performing de-duplication and QA.")
    initial_rows = len(df)
    df = df.drop_duplicates()
    logging.info(f"Removed {initial_rows - len(df)} duplicate rows.")
    year_col = snake_case_string(config["group_columns"]["year"])
    if year_col in df.columns and df[year_col].notna().any():
        coverage_report = (
            df.groupby(year_col, observed=False).size().reset_index(name="count")
        )
        (outdir / "tables" / "T0_coverage.csv").write_text(
            coverage_report.to_csv(index=False)
        )
        t1 = coverage_report.copy()
        t1["percentage"] = (100 * t1["count"] / t1["count"].sum()).round(2)
        (outdir / "tables" / "T1_SampleProfile.csv").write_text(t1.to_csv(index=False))
        logging.info("T0 (Coverage) and T1 (Sample Profile) saved.")
    return df


# --- 5 & 6) Scoring ---


def reverse_score_items(df, config):
    logging.info("Applying reverse scoring.")
    for domain_props in config.get("domains", {}).values():
        for item in domain_props.get("reverse_scored", []):
            item_sc = snake_case_string(item)
            if item_sc in df.columns:
                df[item_sc] = 6 - df[item_sc]
    return df


def compute_composites(df, config):
    logging.info("Computing composite scores.")
    min_coverage = config.get("composite_min_coverage", 0.7)
    for domain_name, props in config.get("domains", {}).items():
        item_cols_exist = [
            c
            for c in [snake_case_string(s) for s in props.get("items", [])]
            if c in df.columns
        ]
        if not item_cols_exist:
            logging.warning(
                f"Skipping composite for '{domain_name}': No item columns found."
            )
            continue
        coverage = df[item_cols_exist].notna().sum(axis=1) / len(item_cols_exist)
        domain_scores = (
            df[item_cols_exist].copy().T.fillna(df[item_cols_exist].mean(axis=1)).T
        )
        composite_col = f"score_{snake_case_string(domain_name)}"
        df[composite_col] = domain_scores.mean(axis=1)
        df.loc[coverage < min_coverage, composite_col] = np.nan
        logging.info(f"Computed composite '{composite_col}'.")
    nominal_cols_exist = [
        c
        for c in [
            snake_case_string(s) for s in config.get("nominal_evidence", {}).values()
        ]
        if c in df.columns
    ]
    if nominal_cols_exist:
        df["evidence_score"] = df[nominal_cols_exist].sum(axis=1)
        logging.info("Computed 'evidence_score'.")
    return df


# --- 7) Descriptive Statistics ---


def generate_descriptive_tables(df, config, outdir):
    logging.info("Generating descriptive statistics tables.")
    tables_dir = outdir / "tables"
    all_tables = {}
    year_col = snake_case_string(config["group_columns"]["year"])

    def iqr(x):
        return x.quantile(0.75) - x.quantile(0.25)

    desc_funcs = ["count", "mean", "median", "std", "min", "max", iqr]
    composite_cols = [c for c in df.columns if c.startswith("score_")]
    if composite_cols:
        all_tables["T2_DomainDescriptivesOverall"] = (
            df[composite_cols].agg(desc_funcs).T
        )
        if year_col in df.columns and df[year_col].notna().any():
            df_long = df.melt(
                id_vars=[year_col],
                value_vars=composite_cols,
                var_name="domain",
                value_name="score",
            )
            all_tables["T3_DomainDescriptivesByYear"] = df_long.groupby(
                [year_col, "domain"], observed=False
            )["score"].agg(desc_funcs)
    nominal_cols = [
        c
        for c in [
            snake_case_string(v) for v in config.get("nominal_evidence", {}).values()
        ]
        if c in df.columns
    ]
    if nominal_cols and year_col in df.columns and df[year_col].notna().any():
        t4_list = []
        for col in nominal_cols:
            res = (
                df.groupby(year_col, observed=False)[col]
                .agg(["sum", "mean"])
                .rename(columns={"sum": "count_yes", "mean": "percentage_yes"})
            )
            res["percentage_yes"] *= 100
            res["item"] = col
            t4_list.append(res.reset_index())
        if t4_list:
            all_tables["T4_EvidencePerItem"] = pd.concat(t4_list)
    if "evidence_score" in df.columns:
        all_tables["T5_EvidenceScoreDescriptives"] = (
            df["evidence_score"].agg(desc_funcs).to_frame(name="evidence_score").T
        )

    if all_tables:
        with pd.ExcelWriter(
            tables_dir / "ALL_TABLES.xlsx", engine="openpyxl"
        ) as writer:
            for name, table in all_tables.items():
                table.to_csv(tables_dir / f"{name}.csv", index=True)
                table.to_excel(writer, sheet_name=name, index=True)
        logging.info(
            f"Descriptive tables compiled into {tables_dir / 'ALL_TABLES.xlsx'}"
        )


# --- 8 & 9) Inferential Tests & Visualizations ---


def run_analysis_and_visuals(df, config, outdir):
    logging.info("Running comparative tests and generating visualizations.")
    tables_dir, figs_dir = outdir / "tables", outdir / "figures"
    alpha = config.get("alpha", 0.05)
    year_col = snake_case_string(config["group_columns"]["year"])

    if not (year_col in df.columns and df[year_col].notna().any()):
        logging.warning(
            "Skipping group comparisons: grouping column is missing or empty."
        )
        return

    kw_results, dunn_results, chi2_results = [], [], []

    score_vars = [c for c in df.columns if c.startswith("score_")]
    for i, var in enumerate(score_vars):
        if df[var].notna().sum() > 1:
            h, p = kruskal(
                *[g.dropna() for _, g in df.groupby(year_col, observed=False)[var]]
            )
            n = df[var].notna().sum()
            k = df[year_col].nunique()
            eps_sq = (h - k + 1) / (n - k) if (n - k) > 0 else 0
            kw_results.append(
                {
                    "variable": var,
                    "H": h,
                    "df": k - 1,
                    "p": p,
                    "epsilon_squared": eps_sq,
                }
            )

            # --- FIX FOR T6b ---
            if p < alpha:
                dunn_raw = posthoc_dunn(
                    df, val_col=var, group_col=year_col, p_adjust="holm"
                )
                # Take the lower triangle of the matrix to get unique, non-self pairs
                dunn_tidy = (
                    dunn_raw.where(np.tril(np.ones(dunn_raw.shape), k=-1).astype(bool))
                    .stack()
                    .reset_index()
                )
                dunn_tidy.columns = ["group1", "group2", "p_adj"]
                dunn_tidy["variable"] = var
                dunn_results.append(dunn_tidy)

            plt.figure(figsize=(10, 6))
            sns.boxplot(
                x=year_col, y=var, data=df, order=sorted(df[year_col].cat.categories)
            )
            plt.title(f"Distribution of {var.replace('_', ' ').title()} by Year Level")
            plt.xlabel("Year Level")
            plt.ylabel("Score")
            plt.savefig(figs_dir / f"F{i + 1}_{var}_boxplot.png", bbox_inches="tight")
            plt.close()

    nominal_cols = [
        c
        for c in [
            snake_case_string(v) for v in config.get("nominal_evidence", {}).values()
        ]
        if c in df.columns
    ]
    if nominal_cols:
        for item in nominal_cols:
            if df[item].nunique() > 1:
                ct = pd.crosstab(df[year_col], df[item])
                if ct.shape[0] > 1 and ct.shape[1] > 1:
                    chi2, p, dof, _ = chi2_contingency(ct)
                    n = ct.sum().sum()
                    phi2 = chi2 / n
                    r, k = ct.shape
                    v = (
                        np.sqrt(phi2 / min(k - 1, r - 1))
                        if min(k - 1, r - 1) > 0
                        else 0
                    )
                    chi2_results.append(
                        {"item": item, "chi2": chi2, "df": dof, "p": p, "CramerV": v}
                    )

        df_pct = df.groupby(year_col, observed=False)[nominal_cols].mean().mul(100)
        if not df_pct.empty:
            df_pct.plot(kind="bar", stacked=False, figsize=(12, 7))
            plt.title("% 'Yes' for Evidence Items by Year Level")
            plt.ylabel("Percentage (%)")
            plt.xticks(rotation=0)
            plt.legend(
                title="Evidence Item", bbox_to_anchor=(1.05, 1), loc="upper left"
            )
            plt.savefig(figs_dir / "F5_Evidence_StackedBar.png", bbox_inches="tight")
            plt.close()

    if kw_results:
        pd.DataFrame(kw_results).to_csv(tables_dir / "T6_KW_Tests.csv", index=False)
    if dunn_results:
        pd.concat(dunn_results).to_csv(tables_dir / "T6b_Dunn_PostHoc.csv", index=False)
    if chi2_results:
        pd.DataFrame(chi2_results).to_csv(
            tables_dir / "T7_ChiSquare_EvidenceItems.csv", index=False
        )
    logging.info("Comparative analysis and visualizations are complete.")


# --- 11) Final Summary ---


def write_run_summary(df, config, outdir):
    logging.info("Writing run summary.")
    year_col = snake_case_string(config["group_columns"]["year"])
    summary_path = outdir / "run_summary.txt"
    alpha = config.get("alpha", 0.05)
    with open(summary_path, "w") as f:
        f.write(
            f"--- Analysis Run Summary: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n"
        )
        f.write(f"\nTotal Respondents (after de-duplication): {len(df)}\n")
        if year_col in df.columns and df[year_col].notna().any():
            f.write(
                "Respondents by Year Level:\n"
                + df[year_col].value_counts().sort_index().to_string()
                + "\n"
            )
        f.write(f"\n--- Significant Findings (alpha = {alpha}) ---\n")
        for test_name, file_path in [
            ("Kruskal-Wallis", "T6_KW_Tests.csv"),
            ("Chi-Square", "T7_ChiSquare_EvidenceItems.csv"),
        ]:
            path = outdir / "tables" / file_path
            if path.exists():
                try:
                    sig = pd.read_csv(path).query(f"p < {alpha}")
                    f.write(f"\nSignificant results for {test_name} tests:\n")
                    f.write(
                        sig.to_string(index=False) if not sig.empty else "  - None.\n"
                    )
                except Exception as e:
                    f.write(f"\nCould not read results from {file_path}: {e}\n")
    logging.info(f"Run summary saved to: {summary_path}")


# --- 1) Main Execution Block ---


def main():
    parser = argparse.ArgumentParser(description="End-to-end survey analysis script.")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input data file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config.yaml file."
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Directory to save outputs."
    )
    args = parser.parse_args()

    output_dir = Path(args.outdir)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir / "logs")
    logging.info("--- Starting Analysis Run ---")

    config = load_config(args.config)

    df = load_data(args)
    df = validate_and_clean(df, config)
    df = deduplicate_and_qa(df, config, output_dir)
    df = reverse_score_items(df, config)
    df_clean = compute_composites(df, config)

    (Path("data") / "interim").mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(Path("data") / "interim" / "data_clean.csv", index=False)
    logging.info(f"Cleaned data saved to data/interim/data_clean.csv")

    generate_descriptive_tables(df_clean, config, output_dir)
    run_analysis_and_visuals(df_clean, config, output_dir)
    write_run_summary(df_clean, config, output_dir)

    logging.info("--- Analysis Run Complete ---")


if __name__ == "__main__":
    main()
