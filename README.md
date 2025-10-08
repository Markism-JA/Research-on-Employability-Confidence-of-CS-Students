# Perceived Employability Confidence of CS Students: Data Analysis

This repository contains the data analysis script and configuration for my research paper, "Perceived Employability Confidence Among Computer Science Undergraduates in STI College Caloocan." The project analyzes survey data to understand student confidence across various domains and their accumulation of tangible career assets.

The analysis follows a quantitative, descriptive-comparative design. The main Python script (`Generate-survey-config.py`) automates the entire process, from data loading and cleaning to statistical testing and the generation of all tables and figures used in the final paper.

## Project Goal

The primary goal of this research is to answer the following questions:
1.  What are the self-perceived confidence levels of CS students in technical skills, soft skills, job readiness, and industry trend awareness?
2.  Do these confidence levels differ significantly across academic year levels (2nd, 3rd, and 4th year)?
3.  What proportion of students are building tangible employability evidence (e.g., portfolios, hackathon experience)?
4.  Is there a statistically significant association between a student's year level and their likelihood of having this evidence?

## Key Features

-   **Automated End-to-End Analysis:** A single script handles the entire data analysis pipeline.
-   **Configurable:** The analysis is controlled by a YAML file (`survey_config.yaml`), allowing for easy adaptation to different survey questions or domains without changing the code.
-   **Statistical Rigor:** Implements appropriate non-parametric tests (Kruskal-Wallis H, Dunn's post-hoc, Chi-Square) and calculates meaningful effect sizes (Epsilon-squared, Cramér's V).
-   **Automated Output Generation:** Automatically generates all descriptive and inferential statistics tables (in `.csv` and a compiled `.xlsx` format) and all visualizations (`.png` figures) required for the research paper.

## Project Structure

The repository is organized as follows:

```
project/
├── config/
│   └── survey_config.yaml    # Mappings for domains, columns, options
├── data/
│   ├── raw/                  # Original survey data export (e.g., responses.ods)
│   └── interim/              # Cleaned data with composite scores (data_clean.csv)
├── outputs/
│   ├── tables/               # All generated tables (T1–T7)
│   └── figures/              # All generated figures (F1–F6)
├── .gitignore                
├── Generate-survey-config.py               # The main data analysis script
├── README.md                 # This file
└── requirements.txt          # A list of all required Python libraries
```

## Methodology in a Nutshell

The `analysis.py` script performs the following steps:

1.  **Load Data:** Reads the raw survey data from an ODS, XLSX, or CSV file.
2.  **Clean & Validate:**
    -   Converts all column names to a standardized `snake_case` format.
    -   Parses text-based year levels (e.g., "2nd Year") into numeric categories.
    -   Maps "Yes/No" answers for evidence items to binary (1/0).
    -   Validates Likert scale responses, converting invalid entries to `NaN`.
3.  **Calculate Composite Scores:**
    -   For each domain defined in `survey_config.yaml`, it calculates a composite score (mean of Likert items).
    -   It handles missing data by applying a minimum coverage rule (e.g., 70%) and imputing with the respondent's mean for that domain.
    -   It calculates a collective `Evidence Score` (0-4) by summing the binary evidence items.
4.  **Generate Descriptive Statistics:**
    -   Calculates frequencies and percentages for the sample profile (`T1`).
    -   Calculates mean, median, std, etc., for all composite scores (`T2`, `T3`).
    -   Calculates percentages for all evidence items (`T4`).
5.  **Perform Inferential Statistics:**
    -   Runs a **Kruskal-Wallis H test** to compare domain scores across year levels (`T6`).
    -   If significant, runs a **Dunn's post-hoc test** to find which specific pairs are different (`T6b`).
    -   Runs a **Chi-Square test of independence** to check for associations between year level and evidence items (`T7`).
    -   Calculates **Epsilon-squared** and **Cramér's V** as effect sizes.
6.  **Generate Visualizations:**
    -   Creates **boxplots** to show the distribution of scores by year level (`F1-F5`).
    -   Creates a **grouped bar chart** to show the percentage of "Yes" responses for evidence items by year level (`F6`).

## How to Run the Analysis

To reproduce the analysis, follow these steps:

### 1. Prerequisites

-   Python 3.9+
-   A virtual environment (recommended)

### 2. Setup

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### 3. Place Your Data

Place your raw survey data file (e.g., `responses.ods`) inside the `data/raw/` directory.

### 4. Configure the Analysis

Review the `config/survey_config.yaml` file to ensure the column names and Likert values match your survey data. This is where you define the items for each domain.

### 5. Run the Script

Execute the `Generate-survey-config.py` script from the root of the project directory, pointing to your input file and config.

```bash
python Generate-survey-config.py \
  --input "data/raw/your_data_file.ods" \
  --config "config/survey_config.yaml" \
  --outdir "outputs"```

After the script finishes, all generated tables and figures will be available in the `outputs/` directory. A log of the run will be saved in `outputs/logs/run.log`.

## Dependencies

The analysis relies on the following major Python libraries:

-   `pandas`: For data manipulation and analysis.
-   `numpy`: For numerical operations.
-   `scipy`: For core statistical tests (Kruskal-Wallis, Chi-Square).
-   `scikit-posthocs`: For Dunn's post-hoc test.
-   `seaborn` & `matplotlib`: For data visualization.
-   `PyYAML`: For reading the configuration file.
-   `odfpy` & `openpyxl`: For reading `.ods` and `.xlsx` files.

A complete list is available in `requirements.txt`.
