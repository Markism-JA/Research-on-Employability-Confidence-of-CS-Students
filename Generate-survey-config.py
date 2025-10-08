import pandas as pd
import yaml

# === INPUTS ===
ods_file = "data/raw/Perceived Employability Confidence of CS Students (Responses) From 2nd to 4th year.ods"
input_yaml = "config_ranges.yaml"
output_yaml = "config_expanded.yaml"

# === LOAD RAW HEADERS ===
df = pd.read_excel(ods_file, engine="odf")
headers = df.columns.tolist()

# === LOAD CONFIG WITH RANGES ===
with open(input_yaml, "r") as f:
    config = yaml.safe_load(f)

# === BUILD ANALYSIS-READY CONFIG ===
analysis_config = {}

# ID columns (optional)
analysis_config["id_columns"] = ["timestamp"]

# Group columns (optional)
analysis_config["group_columns"] = {"year": "year_level", "section": "section"}

# Nominal evidence mapping
nominal_items = ["portfolio", "hackathon", "certifications", "linked_in"]
index_nominal = config["employability_evidence"]["index_range"]
nominal_headers = headers[index_nominal[0] : index_nominal[1] + 1]
analysis_config["nominal_evidence"] = dict(zip(nominal_items, nominal_headers))
analysis_config["nominal_yes_values"] = ["Yes", "Y", "1", 1, True]
analysis_config["nominal_no_values"] = ["No", "N", "0", 0, False]

# Domains
domain_names = ["technical", "soft", "job_readiness", "industry_trend"]
domains_dict = {}
for domain in domain_names:
    rng = config["domains"][domain]["index_range"]
    items = headers[rng[0] : rng[1] + 1]
    domains_dict[domain] = {
        "items": items,
        "reverse_scored": [],
    }  # adjust reverse if needed

analysis_config["domains"] = domains_dict

# Likert & composite settings
analysis_config["likert_valid_values"] = [1, 2, 3, 4, 5]
analysis_config["composite_min_coverage"] = 0.7
analysis_config["alpha"] = 0.05
analysis_config["bootstrap_ci"] = False

# === SAVE FINAL CONFIG ===
with open(output_yaml, "w") as f:
    yaml.dump(analysis_config, f, sort_keys=False)

print(f"Expanded analysis config saved to {output_yaml}")
