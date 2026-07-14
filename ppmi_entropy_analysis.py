#!/usr/bin/env python3
"""
ppmi_entropy_analysis.py — Shannon Entropy Decomposition of PPMI MDS-UPDRS Data
=================================================================================
Computes Shannon entropy from MDS-UPDRS item-level score distributions
stratified by genetic subgroup (GBA1 vs sporadic). Runs permutation testing
and generates publication-ready figures.

Hypotheses (Rodriguez, 2026):
  H1/H2: GBA1-PD shows domain-specific UPDRS entropy trajectories that
         diverge from idiopathic PD, invisible to total UPDRS comparison

Usage:
  python ppmi_entropy_analysis.py                    # default: data/ folder
  python ppmi_entropy_analysis.py --data-dir /path/to/ppmi_csvs/
  python ppmi_entropy_analysis.py --analysis h1      # GBA1 vs sporadic

Options:
  --data-dir DIR     Folder containing PPMI CSV files (default: data/)
  --output DIR       Output directory (default: results/)
  --analysis MODE    h1 = GBA1 vs sporadic (default: h1)
  --n-perms N        Number of permutations (default: 5000)
  --min-n N          Minimum observations per item per visit (default: 15)
  --seed N           Random seed (default: 42)

Output (all in results/):
  entropy_trajectories_{analysis}.json    Raw entropy per group per Part
  divergence_permtest_{analysis}.json     Permutation test results
  summary_{analysis}.txt                  Human-readable results summary
  fig1_entropy_trajectories_{analysis}.png
  fig2_divergence_summary_{analysis}.png

Requirements:
  pip install numpy scipy pandas matplotlib

Data:
  PPMI (https://www.ppmi-info.org/)
  All analysis performed locally.

Author: Anderson M. Rodriguez
ORCID:  0009-0007-5179-9341
"""

import numpy as np
import pandas as pd
import json
import sys
import os
import time
import argparse
import warnings
from glob import glob

warnings.filterwarnings("ignore")

# NumPy 2.0+ moved trapz to trapezoid
try:
    from numpy import trapezoid as np_trapz
except ImportError:
    from numpy import trapz as np_trapz

# ============================================================
# CONFIGURATION
# ============================================================

# MDS-UPDRS domains: items detected by column prefix
# Items scored 0-4 ordinal. Total columns (ending TOT/RTOT) excluded.
DOMAINS = {
    "Part_I": {
        "description": "Non-Motor Experiences of Daily Living",
        "prefix": "NP1",
        "files": ["MDS-UPDRS_Part_I_Patient_Questionnaire", "MDS-UPDRS_Part_I_"],
    },
    "Part_II": {
        "description": "Motor Experiences of Daily Living",
        "prefix": "NP2",
        "files": ["MDS_UPDRS_Part_II__Patient_Questionnaire"],
    },
    "Part_III": {
        "description": "Motor Examination",
        "prefix": "NP3",
        "files": ["MDS-UPDRS_Part_III"],
    },
    "Part_IV": {
        "description": "Motor Complications",
        "prefix": "NP4",
        "files": ["MDS-UPDRS_Part_IV__Motor_Complications"],
    },
}

# Canonical Part ordering for every display/iteration site. Sorting keys such as
# "Part_I_IAD" vs "Part_III_IAD" lexicographically scrambles them (the underscore
# sorts after the letters), so all output ordering routes through this list.
PART_ORDER = ["Part_I", "Part_II", "Part_III", "Part_IV"]

DOMAIN_COLORS = {
    "Part_I":   "#c0392b",
    "Part_II":  "#2471a3",
    "Part_III": "#27ae60",
    "Part_IV":  "#7d3c98",
}

GROUP_STYLES = {
    "group_a": {"linestyle": "-",  "linewidth": 2.2, "markersize": 5, "alpha": 0.95},
    "group_b": {"linestyle": "--", "linewidth": 2.2, "markersize": 5, "alpha": 0.75},
}

# Visit code to approximate months from baseline
# Derived from PPMI protocol; refined by actual INFODT when available
VISIT_MONTH_MAP = {
    "SC": -1, "BL": 0, "V01": 3, "V02": 6, "V03": 9, "V04": 12,
    "V05": 18, "V06": 24, "V07": 30, "V08": 36, "V09": 42,
    "V10": 48, "V11": 54, "V12": 60, "V13": 66, "V14": 72,
    "V15": 84, "V16": 96, "V17": 108, "ST": -2,
}


# ============================================================
# UTILITIES
# ============================================================
def shannon_entropy_bits(counts):
    """Shannon entropy in bits from a count vector."""
    counts = np.array(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return np.nan
    probs = counts / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def find_file(data_dir, patterns):
    """Find a CSV file matching any of the given name patterns."""
    for pat in patterns:
        # Exact match
        for ext in [".csv", ".CSV"]:
            candidates = glob(os.path.join(data_dir, f"*{pat}*{ext}"))
            if candidates:
                return sorted(candidates)[0]
    return None


def detect_item_columns(df, prefix):
    """Detect UPDRS item columns by prefix, excluding totals."""
    items = []
    exclude = {"TOT", "RTOT", "TOTAL", "SUM", "COMMENT", "CMMNT"}
    for c in df.columns:
        c_upper = c.upper().strip().strip('"')
        prefix_upper = prefix.upper()
        if c_upper.startswith(prefix_upper):
            # Exclude totals and non-item columns
            suffix = c_upper[len(prefix_upper):]
            if not any(ex in suffix for ex in exclude):
                # Check if the column contains numeric ordinal data
                vals = pd.to_numeric(df[c], errors="coerce").dropna()
                # Filter out 101 ("unable to rate") before checking range
                vals = vals[vals != 101]
                if len(vals) > 0 and vals.min() >= 0 and vals.max() <= 4:
                    items.append(c)
    return sorted(items)


# ============================================================
# DATA LOADING
# ============================================================
def load_ppmi_data(data_dir):
    """
    Load and merge all required PPMI data files.
    Returns a dictionary of DataFrames keyed by domain,
    plus participant metadata.
    """
    print(f"\n  Loading PPMI data from: {data_dir}/")

    # --- Participant Status ---
    ps_file = find_file(data_dir, ["Participant_Status"])
    if ps_file is None:
        print("  ERROR: Cannot find Participant_Status file.")
        sys.exit(1)
    ps = pd.read_csv(ps_file, low_memory=False)
    print(f"  Participant_Status: {len(ps)} rows")

    # Standardize column names
    ps.columns = [c.strip() for c in ps.columns]

    # --- Genetic Status ---
    gen_file = find_file(data_dir, [
        "Consensus_APOE_Genotype_and_Pathogenic_Variants",
        "iu_genetic_consensus",
        "Genetic_Data",
        "Consensus_Genotype",
        "PPMI_PD_Variants_Genetic_Status",
    ])
    gen_status = None
    if gen_file:
        gen_status = pd.read_csv(gen_file, low_memory=False)
        gen_status.columns = [c.strip() for c in gen_status.columns]
        print(f"  Genetic Status: {len(gen_status)} rows from {os.path.basename(gen_file)}")

    # --- Participant Genetic Status (simpler lookup) ---
    pgs_file = find_file(data_dir, [
        "Participant_Genetic_Status",
        "PPMI_PD_Variants_Genetic_Status",
        "PD_Variants",
    ])
    pgs = None
    if pgs_file:
        pgs = pd.read_csv(pgs_file, low_memory=False)
        pgs.columns = [c.strip() for c in pgs.columns]
        print(f"  Participant Genetic Status: {len(pgs)} rows")

    # --- MDS-UPDRS Parts ---
    updrs_data = {}
    for dname, dinfo in DOMAINS.items():
        dfs = []
        for fpattern in dinfo["files"]:
            fpath = find_file(data_dir, [fpattern])
            if fpath:
                df = pd.read_csv(fpath, low_memory=False)
                df.columns = [c.strip() for c in df.columns]
                items = detect_item_columns(df, dinfo["prefix"])
                if items:
                    # Keep only essential columns + items
                    keep = ["PATNO", "EVENT_ID", "INFODT"]
                    if "PDSTATE" in df.columns:
                        keep.append("PDSTATE")
                    keep += items
                    keep = [c for c in keep if c in df.columns]
                    dfs.append(df[keep].copy())
                    print(f"  {dname}: {len(items)} items from {os.path.basename(fpath)}")

        if dfs:
            # Part I spans two files (clinician + patient questionnaire). They are
            # row-concatenated; entropy is computed per item column by pooling across
            # rows, so the disjoint-column NaN fill from concat does not affect results.
            combined = pd.concat(dfs, ignore_index=True)
            updrs_data[dname] = combined

    if not updrs_data:
        print("  ERROR: No MDS-UPDRS data files found.")
        sys.exit(1)

    return {
        "participant_status": ps,
        "genetic_status": gen_status,
        "participant_genetic_status": pgs,
        "updrs": updrs_data,
    }


def build_participant_table(data):
    """
    Build a master participant table with cohort and genetic subgroup.
    """
    ps = data["participant_status"].copy()

    # Detect column names for cohort and subgroup
    cohort_col = None
    for c in ["COHORT_DEFINITION", "COHORT", "cohort_definition"]:
        if c in ps.columns:
            cohort_col = c
            break
    if cohort_col is None:
        print("  WARNING: Cannot find cohort column. Columns:", list(ps.columns))

    # Detect genetic subgroup — may be in Participant_Status or needs merge
    subgroup_col = None
    for c in ["GENETIC_SUBGROUP", "genetic_subgroup", "GENOTYPE", "SUBGROUP"]:
        if c in ps.columns:
            subgroup_col = c
            break

    # If no subgroup in participant_status, try genetic status files
    if subgroup_col is None and data["participant_genetic_status"] is not None:
        pgs = data["participant_genetic_status"]
        # Look for GBA-specific columns
        for c in pgs.columns:
            if "GBA" in c.upper():
                print(f"  Found GBA column in Participant Genetic Status: {c}")

    master = ps[["PATNO"]].copy()
    if cohort_col:
        master["COHORT"] = ps[cohort_col].values
    if subgroup_col:
        master["GENETIC_SUBGROUP"] = ps[subgroup_col].values

    # Detect enrollment status
    status_col = None
    for c in ["ENROLL_STATUS", "enroll_status"]:
        if c in ps.columns:
            status_col = c
            break
    if status_col:
        master["ENROLL_STATUS"] = ps[status_col].values

    print(f"\n  Master participant table: {len(master)} participants")
    if cohort_col:
        print(f"  Cohorts: {master['COHORT'].value_counts().to_dict()}")
    if subgroup_col:
        print(f"  Genetic subgroups: {master['GENETIC_SUBGROUP'].value_counts().to_dict()}")

    return master


def compute_months_from_baseline(df):
    """
    Compute months from baseline for each participant.
    Uses INFODT dates when available, falls back to EVENT_ID mapping.
    """
    df = df.copy()

    # Try date-based computation
    if "INFODT" in df.columns:
        df["INFODT"] = pd.to_datetime(df["INFODT"], errors="coerce", format="mixed")
        # Per-patient baseline date
        bl_dates = df[df["EVENT_ID"] == "BL"].groupby("PATNO")["INFODT"].first()
        df = df.merge(bl_dates.rename("BL_DATE"), on="PATNO", how="left")

        has_dates = df["INFODT"].notna() & df["BL_DATE"].notna()
        df.loc[has_dates, "Month"] = (
            (df.loc[has_dates, "INFODT"] - df.loc[has_dates, "BL_DATE"]).dt.days / 30.44
        ).round().astype(int)

    # Fill remaining from visit code map
    if "Month" not in df.columns:
        df["Month"] = np.nan
    missing = df["Month"].isna()
    if missing.any() and "EVENT_ID" in df.columns:
        df.loc[missing, "Month"] = df.loc[missing, "EVENT_ID"].map(VISIT_MONTH_MAP)

    df = df.dropna(subset=["Month"])
    df["Month"] = df["Month"].astype(int)

    # Drop screening visits and negatives
    df = df[df["Month"] >= 0]

    # Snap to 6-month visit windows to aggregate observations
    # across scheduling variability (±weeks around protocol visits)
    df["Month"] = ((df["Month"] / 6).round() * 6).astype(int)

    if "BL_DATE" in df.columns:
        df = df.drop(columns=["BL_DATE"])

    return df


# ============================================================
# GROUP ASSIGNMENT
# ============================================================
def assign_groups_h1(master, data):
    """H1/H2: GBA1-PD vs Sporadic PD."""
    if "GENETIC_SUBGROUP" in master.columns:
        gba = master[
            (master["COHORT"].str.contains("Parkinson", case=False, na=False)) &
            (master["GENETIC_SUBGROUP"].str.upper() == "GBA")
        ]["PATNO"].unique()
        sporadic = master[
            (master["COHORT"].str.contains("Parkinson", case=False, na=False)) &
            (master["GENETIC_SUBGROUP"].str.upper().isin(["SRDC", "SPORADIC"]))
        ]["PATNO"].unique()
    else:
        # Try from genetic data file
        gen = data.get("genetic_status")
        if gen is None:
            print("  ERROR: Cannot determine GBA1 status. No genetic subgroup data found.")
            return None, None, None, None
        # Look for GBA variant column
        gba_col = None
        for c in gen.columns:
            if "GBA" in c.upper() and "VARIANT" in c.upper():
                gba_col = c
                break
        if gba_col is None:
            for c in gen.columns:
                if "GBA" in c.upper():
                    gba_col = c
                    break
        if gba_col is None:
            print("  ERROR: Cannot find GBA column in genetic data.")
            print(f"  Available columns: {list(gen.columns)}")
            return None, None, None, None

        # Identify carriers: non-zero, non-null GBA value
        gba_carriers = gen[
            gen[gba_col].notna() &
            (gen[gba_col] != "") &
            (gen[gba_col].astype(str).str.strip() != "0")
        ]["PATNO"].unique()

        pd_patients = master[
            master["COHORT"].str.contains("Parkinson", case=False, na=False)
        ]["PATNO"].unique()

        gba = np.intersect1d(gba_carriers, pd_patients)

        # Sporadic = PD patients with NO pathogenic variants in any gene
        # Check all variant columns: GBA, LRRK2, SNCA, VPS35, PRKN, PARK7, PINK1
        variant_genes = ["GBA", "LRRK2", "SNCA", "VPS35", "PRKN", "PARK7", "PINK1"]
        any_variant_carriers = set()
        for gene in variant_genes:
            if gene in gen.columns:
                carriers = gen[
                    gen[gene].notna() &
                    (gen[gene] != "") &
                    (gen[gene].astype(str).str.strip() != "0")
                ]["PATNO"].unique()
                any_variant_carriers.update(carriers)

        tested = gen[gen["PATNO"].isin(pd_patients)]["PATNO"].unique()
        sporadic = np.setdiff1d(tested, list(any_variant_carriers))
        sporadic = np.intersect1d(sporadic, pd_patients)

    print(f"\n  H1/H2 Groups:")
    print(f"    GBA1-PD:     {len(gba)} participants")
    print(f"    Sporadic PD: {len(sporadic)} participants")

    return gba, sporadic, "GBA1-PD", "Sporadic PD"


# ============================================================
# ENTROPY COMPUTATION
# ============================================================
def compute_entropy_by_group(updrs_data, group_a_ids, group_b_ids,
                             group_a_label, group_b_label, min_n=15):
    """
    Compute per-Part Shannon entropy trajectories for two groups.

    For each UPDRS Part, at each visit/month, compute the average
    Shannon entropy across all items in that Part. Items are scored
    0-4 ordinal; entropy is computed from the distribution of scores
    across the population at that timepoint.

    Returns:
      group_traj: {group_label: {part_name: np.array}}
      common_months: sorted list of months present in both groups × all Parts
      group_info: {group_label: {"n_patients": int}}
    """
    groups = {
        group_a_label: set(group_a_ids),
        group_b_label: set(group_b_ids),
    }

    group_traj = {g: {} for g in groups}
    group_info = {g: {"n_patients": len(ids)} for g, ids in groups.items()}

    for dname, dinfo in DOMAINS.items():
        if dname not in updrs_data:
            print(f"    Skipping {dname}: no data loaded")
            continue

        df = updrs_data[dname].copy()
        items = detect_item_columns(df, dinfo["prefix"])
        if not items:
            print(f"    Skipping {dname}: no valid item columns detected")
            continue

        # Convert to numeric
        for col in items:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Handle "unable to rate" coded as 101
            df.loc[df[col] > 4, col] = np.nan

        # Compute months
        df = compute_months_from_baseline(df)

        all_months = sorted(df["Month"].unique())

        for glabel, gids in groups.items():
            gdf = df[df["PATNO"].isin(gids)]
            entropies = {}

            for m in all_months:
                month_data = gdf[gdf["Month"] == m]
                item_H = []

                for col in items:
                    vals = month_data[col].dropna().values
                    if len(vals) < min_n:
                        continue
                    counts = np.zeros(5)
                    for s in vals:
                        idx = int(round(s))
                        if 0 <= idx <= 4:
                            counts[idx] += 1
                    item_H.append(shannon_entropy_bits(counts))

                # Require at least half the items to have sufficient data
                if len(item_H) >= max(1, len(items) // 2):
                    entropies[m] = np.mean(item_H)

            group_traj[glabel][dname] = entropies

    # Find common months across all groups × all domains
    month_sets = []
    for glabel in group_traj:
        for dname in group_traj[glabel]:
            month_sets.append(set(group_traj[glabel][dname].keys()))

    if not month_sets:
        return group_traj, [], group_info

    common = sorted(set.intersection(*month_sets))

    # Convert to arrays
    for glabel in group_traj:
        for dname in list(group_traj[glabel].keys()):
            ent = group_traj[glabel][dname]
            group_traj[glabel][dname] = np.array([ent[m] for m in common])

    return group_traj, [int(m) for m in common], group_info


# ============================================================
# DIVERGENCE METRICS
# ============================================================
def compute_divergence(traj_a, traj_b, months, domains):
    """Compute divergence metrics between two sets of trajectories."""
    t = np.array(months, dtype=float)
    metrics = {}

    for d in domains:
        if d not in traj_a or d not in traj_b:
            continue
        if len(traj_a[d]) == 0 or len(traj_b[d]) == 0:
            continue

        diff = traj_a[d] - traj_b[d]
        abs_diff = np.abs(diff)

        metrics[f"{d}_IAD"] = float(np_trapz(abs_diff, t))
        metrics[f"{d}_mean_sep"] = float(np.mean(diff))

        # Peak timing
        peak_a = int(months[np.argmax(traj_a[d])])
        peak_b = int(months[np.argmax(traj_b[d])])
        metrics[f"{d}_peak_A"] = peak_a
        metrics[f"{d}_peak_B"] = peak_b
        metrics[f"{d}_peak_diff"] = abs(peak_a - peak_b)

    iad_vals = [v for k, v in metrics.items() if k.endswith("_IAD")]
    metrics["omnibus_IAD"] = float(np.mean(iad_vals)) if iad_vals else 0
    metrics["total_IAD"] = float(np.sum(iad_vals)) if iad_vals else 0

    return metrics


# ============================================================
# PERMUTATION TEST
# ============================================================
def run_permutation_test(updrs_data, group_a_ids, group_b_ids,
                         group_a_label, group_b_label,
                         n_perms, min_n, seed):
    """
    Shuffle group labels at the participant level, recompute entropy
    trajectories, and measure divergence under the null.
    """
    rng = np.random.RandomState(seed)
    all_ids = np.concatenate([group_a_ids, group_b_ids])
    n_a = len(group_a_ids)

    print(f"\n  Computing observed trajectories...")
    obs_traj, months, info = compute_entropy_by_group(
        updrs_data, group_a_ids, group_b_ids,
        group_a_label, group_b_label, min_n)

    common_domains = (set(obs_traj[group_a_label].keys()) &
                      set(obs_traj[group_b_label].keys()))
    domains = [d for d in PART_ORDER if d in common_domains]

    if not domains or not months:
        print("  ERROR: No overlapping domains/months between groups.")
        return {}, {}, obs_traj, months, domains, info

    obs = compute_divergence(
        obs_traj[group_a_label], obs_traj[group_b_label], months, domains)

    # Permutation null
    null = {k: [] for k in obs}
    t0 = time.time()

    print(f"  Running permutation test ({n_perms} permutations)...")
    for i in range(n_perms):
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            rate = (i + 1) / el
            rem = (n_perms - i - 1) / rate
            print(f"    permutation {i+1}/{n_perms}  "
                  f"({el:.0f}s elapsed, ~{rem:.0f}s remaining)")

        shuf = rng.permutation(all_ids)
        perm_a = shuf[:n_a]
        perm_b = shuf[n_a:]

        try:
            pt, pm, _ = compute_entropy_by_group(
                updrs_data, perm_a, perm_b,
                group_a_label, group_b_label, min_n)

            perm_domains = sorted(
                set(pt[group_a_label].keys()) &
                set(pt[group_b_label].keys())
            )
            if perm_domains and pm:
                m = compute_divergence(
                    pt[group_a_label], pt[group_b_label], pm, perm_domains)
                for k in null:
                    if k in m:
                        null[k].append(m[k])
        except Exception:
            continue

    # P-values
    results = {}
    for k in sorted(obs.keys()):
        nv = np.array(null.get(k, []))
        o = obs[k]
        if len(nv) == 0:
            p = np.nan; nm = np.nan; ns = np.nan
        else:
            # Add-one (b+1)/(m+1) Monte-Carlo p-value: never exactly zero, and a
            # valid upper bound at the resolution of the realized null sample.
            if "IAD" in k or "peak_diff" in k:
                b = int(np.sum(nv >= o))
            else:
                b = int(np.sum(np.abs(nv) >= abs(o)))
            p = float((b + 1) / (len(nv) + 1))
            nm = float(np.mean(nv))
            ns = float(np.std(nv))

        results[k] = {
            "observed": float(o), "null_mean": nm,
            "null_std": ns, "p_value": p,
            "n_perms": len(nv),
        }

    return results, obs, obs_traj, months, domains, info


# ============================================================
# FIGURES
# ============================================================
def generate_figures(results, obs, traj, months, domains, info,
                     group_a_label, group_b_label, analysis_tag, outdir):
    """Generate publication-ready figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
        "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })

    t = np.array(months)
    n_a = info[group_a_label]["n_patients"]
    n_b = info[group_b_label]["n_patients"]

    # ── Figure 1: Multi-panel entropy trajectories ──────────────
    n_panels = len(domains)
    ncols = 2
    nrows = (n_panels + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows))
    if n_panels == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, dname in enumerate(domains):
        ax = axes[i]
        color = DOMAIN_COLORS.get(dname, "gray")
        label_d = dname.replace("_", " ")

        traj_a = traj[group_a_label][dname]
        traj_b = traj[group_b_label][dname]

        ax.plot(t, traj_a, "o-", color=color,
                label=f"{group_a_label} (n={n_a})",
                **GROUP_STYLES["group_a"])
        ax.plot(t, traj_b, "s--", color=color,
                label=f"{group_b_label} (n={n_b})",
                **GROUP_STYLES["group_b"])

        ax.fill_between(t, traj_a, traj_b, color=color, alpha=0.08)
        ax.set_xlabel("Months from Baseline")
        ax.set_ylabel("Shannon Entropy (bits)")
        ax.set_title(f"MDS-UPDRS {label_d}")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.25)

        iad_key = f"{dname}_IAD"
        if iad_key in results:
            r = results[iad_key]
            p = r["p_value"]
            ax.text(0.97, 0.03, f"IAD = {r['observed']:.3f}\np = {p:.4f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=9, fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="gray", alpha=0.85))

    # Hide unused panels
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Shannon Entropy Trajectories: {group_a_label} vs {group_b_label}\n"
        f"PPMI MDS-UPDRS Item-Level Distributions",
        fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fname = f"fig1_entropy_trajectories_{analysis_tag}.png"
    fig.savefig(os.path.join(outdir, fname))
    plt.close(fig)
    print(f"  {fname}")

    # IAD keys in canonical Part order (used by the bar chart below).
    iad_keys = [f"{d}_IAD" for d in PART_ORDER if f"{d}_IAD" in results]

    # ── Figure 2: IAD bar chart ──────────────────────────────────
    if iad_keys:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        bar_data = []
        for k in iad_keys:
            r = results[k]
            dname = k.replace("_IAD", "")
            bar_data.append((dname, r["observed"], r["null_mean"],
                            r["null_std"], r["p_value"]))

        names = [b[0].replace("_", "\n") for b in bar_data]
        observed = [b[1] for b in bar_data]
        null_mu = [b[2] for b in bar_data]
        null_sd = [b[3] for b in bar_data]
        pvals = [b[4] for b in bar_data]
        colors = [DOMAIN_COLORS.get(b[0], "gray") for b in bar_data]

        x = np.arange(len(names))
        w = 0.35

        ax.bar(x - w / 2, observed, w, color=colors, alpha=0.85,
               label="Observed IAD", edgecolor="black", linewidth=0.5)
        ax.bar(x + w / 2, null_mu, w, color="lightgray", alpha=0.8,
               label="Null mean IAD", edgecolor="black", linewidth=0.5,
               yerr=null_sd, capsize=4)

        for i, p in enumerate(pvals):
            y_pos = max(observed[i], null_mu[i] + null_sd[i]) + 0.05
            ax.text(i, y_pos, f"p = {p:.4f}", ha="center", va="bottom",
                    fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=11)
        ax.set_ylabel("Integrated Absolute Divergence", fontsize=12)
        ax.set_title(
            f"Entropy Divergence: {group_a_label} vs {group_b_label}\n"
            "Observed vs Permutation Null", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.2, axis="y")
        fig.tight_layout()
        fname = f"fig2_divergence_summary_{analysis_tag}.png"
        fig.savefig(os.path.join(outdir, fname))
        plt.close(fig)
        print(f"  {fname}")


# ============================================================
# SUMMARY REPORT
# ============================================================
def write_summary(results, obs, info, group_a_label, group_b_label,
                  months, n_perms, analysis_tag, outdir):
    """Write a human-readable results summary."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"PPMI ENTROPY DIVERGENCE ANALYSIS — {analysis_tag.upper()}")
    lines.append("=" * 70)
    lines.append(f"  Group A: '{group_a_label}' (n={info[group_a_label]['n_patients']})")
    lines.append(f"  Group B: '{group_b_label}' (n={info[group_b_label]['n_patients']})")
    lines.append(f"  Time points: {len(months)} ({months[0]}-{months[-1]} months)")
    lines.append(f"  Permutations: {n_perms}")
    lines.append("")

    lines.append("DOMAIN-LEVEL RESULTS:")
    lines.append("-" * 50)

    for d in PART_ORDER:
        k = f"{d}_IAD"
        if k not in results:
            continue
        r = results[k]
        dname = k.replace("_IAD", "").replace("_", " ")
        p = r["p_value"]
        sig = " ***" if p <= 0.001 else " **" if p <= 0.01 else " *" if p <= 0.05 else ""
        p_disp = f"p = {r['p_value']:.4f}"
        lines.append(f"  MDS-UPDRS {dname}:")
        lines.append(f"    IAD observed:  {r['observed']:.4f}")
        lines.append(f"    IAD null:      {r['null_mean']:.4f} +/- {r['null_std']:.4f}")
        lines.append(f"    {p_disp}{sig}")

        peak_key = f"{dname.replace(' ', '_')}_peak_diff"
        if peak_key in results and results[peak_key]["observed"] > 0:
            lines.append(f"    Peak timing diff: {results[peak_key]['observed']:.0f} months")
        lines.append("")

    r = results.get("total_IAD", {})
    p = r.get("p_value", np.nan)
    sig = " ***" if p <= 0.001 else " **" if p <= 0.01 else " *" if p <= 0.05 else ""
    lines.append("OMNIBUS:")
    lines.append("-" * 50)
    lines.append(f"  Total IAD:     {r.get('observed', 0):.4f}")
    lines.append(f"  Total null:    {r.get('null_mean', 0):.4f} +/- {r.get('null_std', 0):.4f}")
    lines.append(f"  p = {p:.4f}{sig}")

    if r.get("null_std", 0) > 0:
        sigma = (r["observed"] - r["null_mean"]) / r["null_std"]
        lines.append(f"  sigma from null: {sigma:.1f}")
    lines.append("")

    text = "\n".join(lines)
    fname = f"summary_{analysis_tag}.txt"
    with open(os.path.join(outdir, fname), "w") as f:
        f.write(text)
    print(f"\n{text}")
    return text


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Shannon entropy decomposition of PPMI MDS-UPDRS data")
    parser.add_argument("--data-dir", default="data",
                        help="Folder containing PPMI CSV files")
    parser.add_argument("--output", default="results",
                        help="Output directory")
    parser.add_argument("--analysis", default="h1",
                        choices=["h1"],
                        help="Which analysis to run")
    parser.add_argument("--n-perms", type=int, default=5000,
                        help="Number of permutations")
    parser.add_argument("--min-n", type=int, default=15,
                        help="Minimum observations per item per visit")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    t_start = time.time()
    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("PPMI ENTROPY DECOMPOSITION — MDS-UPDRS")
    print("  Shannon entropy · Permutation tested · Model-independent")
    print("  Rodriguez (2026)")
    print("=" * 70)

    # Load all data
    data = load_ppmi_data(args.data_dir)
    master = build_participant_table(data)

    analyses = [("h1", assign_groups_h1)]

    for tag, assign_fn in analyses:
        print(f"\n{'=' * 70}")
        print(f"ANALYSIS: {tag.upper()}")
        print(f"{'=' * 70}")

        group_a, group_b, label_a, label_b = assign_fn(master, data)

        if group_a is None or len(group_a) == 0 or len(group_b) == 0:
            print(f"  Skipping {tag}: insufficient group assignments.")
            continue

        results, obs, traj, months, domains, info = run_permutation_test(
            data["updrs"], group_a, group_b, label_a, label_b,
            args.n_perms, args.min_n, args.seed)

        if not results:
            print(f"  No results for {tag}.")
            continue

        # Save JSON
        with open(os.path.join(args.output, f"divergence_permtest_{tag}.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)

        traj_export = {}
        for grp in traj:
            traj_export[grp] = {
                "months": months,
                "n_patients": info[grp]["n_patients"],
                "domains": {d: traj[grp][d].tolist() for d in traj[grp]
                           if isinstance(traj[grp][d], np.ndarray)},
            }
        with open(os.path.join(args.output, f"entropy_trajectories_{tag}.json"), "w") as f:
            json.dump(traj_export, f, indent=2, default=float)

        # Figures
        print(f"\n  Generating figures...")
        try:
            generate_figures(results, obs, traj, months, domains, info,
                            label_a, label_b, tag, args.output)
        except Exception as e:
            print(f"  Figure generation error: {e}")
            import traceback; traceback.print_exc()

        # Summary
        write_summary(results, obs, info, label_a, label_b,
                     months, args.n_perms, tag, args.output)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"COMPLETE. Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"All outputs in: {args.output}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
