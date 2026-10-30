from pathlib import Path
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="AO-BRAIN | Evidence Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2.5rem;}
    .hero {padding: 1.15rem 1.35rem; border: 1px solid rgba(120,120,120,.25);
           border-radius: 16px; margin-bottom: 1rem;}
    .hero-kicker {font-size: .78rem; letter-spacing: .12em; text-transform: uppercase;
                  opacity: .72; font-weight: 650;}
    .hero-title {font-size: 2.1rem; font-weight: 750; margin: .1rem 0 .35rem 0;}
    .hero-subtitle {font-size: 1rem; opacity: .85; line-height: 1.5; max-width: 1000px;}
    .small-note {font-size: .86rem; opacity: .78;}
    div[data-testid="stMetric"] {border: 1px solid rgba(120,120,120,.22);
                                  border-radius: 12px; padding: .75rem .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# Keep a central list of the columns used by this app. Missing optional columns
# are filled with empty values so a partially completed workbook won't crash it.
STUDY_COLUMNS = [
    "Study_ID", "Full_Title", "Authors", "Year", "DOI_URL", "Evidence_Category",
    "Study_Design", "Research_Objective", "Population_Description", "Total_N",
    "EEG_State", "EEG_Recording_Conditions", "Clinical_Measures", "Pain_Duration",
    "Diagnostic_Phenotypes", "Phenotype_Definition", "Main_Limitations",
    "Peer_Review_Status",
]

FINDING_COLUMNS = [
    "Finding_ID", "Study_ID", "Evidence_Category", "EEG_State", "Task_or_Condition",
    "EEG_Band", "Frequency_Hz", "Brain_Region", "EEG_Metric", "Clinical_Outcome",
    "Clinical_Measure", "Group_or_Contrast", "Analysis_Type", "Statistical_Test",
    "Result_Direction", "Effect_Estimate", "P_Value", "Confidence_Interval",
    "Other_stadistics", "Finding_Summary", "Authors_Interpretation",
    "Limitations_or_Caveats",
]

REQUIRED_STUDY_COLUMNS = ["Study_ID"]
REQUIRED_FINDING_COLUMNS = ["Finding_ID", "Study_ID"]


def clean_value(value):
    """Convert a cell to readable text and normalize repeated whitespace."""
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = re.sub(r"\s+", " ", str(value)).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def clean_frame(frame, expected_columns):
    frame = frame.copy()
    for column in expected_columns:
        if column not in frame.columns:
            frame[column] = ""
    for column in frame.columns:
        frame[column] = frame[column].map(clean_value)
    return frame


@st.cache_data(show_spinner=False)
def load_workbook(file_path, file_modified_time):
    """Load the two workbook sheets. The timestamp refreshes Streamlit's cache."""
    studies_raw = pd.read_excel(file_path, sheet_name="Studies")
    findings_raw = pd.read_excel(file_path, sheet_name="Findings")

    missing_studies = [c for c in REQUIRED_STUDY_COLUMNS if c not in studies_raw.columns]
    missing_findings = [c for c in REQUIRED_FINDING_COLUMNS if c not in findings_raw.columns]
    if missing_studies or missing_findings:
        details = []
        if missing_studies:
            details.append("Studies: " + ", ".join(missing_studies))
        if missing_findings:
            details.append("Findings: " + ", ".join(missing_findings))
        raise ValueError("Missing required Excel columns — " + " | ".join(details))

    studies = clean_frame(studies_raw, STUDY_COLUMNS)
    findings = clean_frame(findings_raw, FINDING_COLUMNS)
    return studies, findings


def classify_outcome(value):
    text = clean_value(value).lower()
    has_pain = "pain" in text
    has_function = "function" in text or "functional" in text
    if has_pain and has_function:
        return "Pain and physical function"
    if has_pain:
        return "Pain"
    if has_function:
        return "Physical function"
    return "Other / unspecified"


def classify_direction(value):
    """Group equivalent result labels without changing the original Excel value."""
    text = re.sub(r"\s+", " ", clean_value(value).lower()).strip()
    if not text:
        return "Not reported"
    if "no significant" in text or "not significant" in text or "non-significant" in text:
        return "No significant association"
    if "mixed" in text:
        return "Mixed findings"
    if "negative" in text and "association" in text:
        return "Negative association"
    if "positive" in text and "association" in text:
        return "Positive association"
    if text == "negative" or text.startswith("negative "):
        return "Negative (label unspecified)"
    if text == "positive" or text.startswith("positive "):
        return "Positive (label unspecified)"
    if "correlation" in text or text == "correlation" or text == "correlation only":
        return "Correlation; direction unspecified"
    return clean_value(value)


def format_source_url(value):
    value = clean_value(value)
    if not value:
        return ""
    if value.startswith("https://") or value.startswith("http://"):
        return value
    if value.startswith("www."):
        return "https://" + value
    if value.lower().startswith("doi:"):
        value = value[4:].strip()
    if value.startswith("10."):
        return "https://doi.org/" + value
    return ""


def first_nonblank(row, column, fallback="Not recorded in the database."):
    value = clean_value(row.get(column, ""))
    return value if value else fallback


def unique_options(frame, column):
    if column not in frame.columns:
        return []
    values = [clean_value(v) for v in frame[column].tolist()]
    return sorted({v for v in values if v}, key=str.casefold)


def apply_multiselect(frame, column, selected):
    if selected:
        return frame[frame[column].isin(selected)].copy()
    return frame.copy()


def parse_first_number(value):
    """Read the first numeric value from an effect-estimate cell, if possible."""
    text = clean_value(value).replace(",", "")
    match = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_confidence_interval(value):
    """Extract two endpoints from a confidence-interval field when unambiguous."""
    text = clean_value(value).replace("−", "-").replace("–", "-").replace("—", "-")
    if not text:
        return None
    nums = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text.replace(",", " "))
    if len(nums) < 2:
        return None
    try:
        # CI fields sometimes include the confidence level (e.g. "95% CI: -0.2 to 0.5")
        # or the point estimate before parentheses. The final two values are treated as
        # endpoints, then validated against the effect estimate before plotting.
        lower, upper = float(nums[-2]), float(nums[-1])
        if lower > upper:
            return None
        return lower, upper
    except (ValueError, TypeError):
        return None


def make_study_lookup(studies):
    clean_studies = studies[studies["Study_ID"].astype(str).str.strip() != ""].copy()
    clean_studies = clean_studies.drop_duplicates(subset=["Study_ID"], keep="first")
    return clean_studies.set_index("Study_ID", drop=False).to_dict(orient="index")


def display_field(label, value):
    value = clean_value(value)
    if value:
        st.markdown(f"**{label}**")
        st.write(value)
    else:
        st.markdown(f"**{label}**")
        st.caption("Not recorded in the database.")


def add_finding_context(findings, studies):
    findings = findings.copy()
    study_lookup = make_study_lookup(studies)

    findings["_StudyTitle"] = findings["Study_ID"].map(
        lambda sid: study_lookup.get(sid, {}).get("Full_Title", "")
    )
    findings["_Authors"] = findings["Study_ID"].map(
        lambda sid: study_lookup.get(sid, {}).get("Authors", "")
    )
    findings["_Year"] = findings["Study_ID"].map(
        lambda sid: study_lookup.get(sid, {}).get("Year", "")
    )
    findings["_DOI"] = findings["Study_ID"].map(
        lambda sid: study_lookup.get(sid, {}).get("DOI_URL", "")
    )
    findings["_MainLimitations"] = findings["Study_ID"].map(
        lambda sid: study_lookup.get(sid, {}).get("Main_Limitations", "")
    )
    findings["_ClinicalDomain"] = findings["Clinical_Outcome"].map(classify_outcome)
    findings["_DirectionCategory"] = findings["Result_Direction"].map(classify_direction)
    findings["_BandForChart"] = findings["EEG_Band"].map(lambda x: clean_value(x) or "Not reported")
    findings["_MeasureForChart"] = findings["Clinical_Measure"].map(lambda x: clean_value(x) or "Not reported")
    findings["_MetricForChart"] = findings["EEG_Metric"].map(lambda x: clean_value(x) or "Not reported")
    findings["_RegionForChart"] = findings["Brain_Region"].map(lambda x: clean_value(x) or "Not reported")
    findings["_StateForChart"] = findings["EEG_State"].map(lambda x: clean_value(x) or "Not reported")
    return findings, study_lookup


def render_finding_panel(row, study_lookup):
    finding_id = clean_value(row.get("Finding_ID", "")) or "Finding without ID"
    band = clean_value(row.get("EEG_Band", "")) or "Band not recorded"
    outcome = clean_value(row.get("Clinical_Outcome", "")) or "Outcome not recorded"
    title = clean_value(row.get("_StudyTitle", "")) or "Study title not available"
    study_id = clean_value(row.get("Study_ID", ""))
    label = f"{finding_id}  |  {band}  |  {outcome}  —  {title[:95]}"

    with st.expander(label, expanded=False):
        st.caption(
            f"Study ID: {study_id or 'Not recorded'} · "
            f"Authors: {clean_value(row.get('_Authors', '')) or 'Not recorded'} · "
            f"Year: {clean_value(row.get('_Year', '')) or 'Not recorded'}"
        )

        left, right = st.columns([1, 1])
        with left:
            st.markdown("#### Reported finding")
            st.write(first_nonblank(row, "Finding_Summary"))
            st.markdown("#### Clinical and EEG details")
            st.write(f"**Clinical outcome:** {outcome}")
            st.write(f"**Clinical measure:** {clean_value(row.get('Clinical_Measure', '')) or 'Not recorded'}")
            st.write(f"**EEG band / frequency:** {band} / {clean_value(row.get('Frequency_Hz', '')) or 'Not recorded'}")
            st.write(f"**EEG metric:** {clean_value(row.get('EEG_Metric', '')) or 'Not recorded'}")
            st.write(f"**Brain region:** {clean_value(row.get('Brain_Region', '')) or 'Not recorded'}")
            st.write(f"**EEG state:** {clean_value(row.get('EEG_State', '')) or 'Not recorded'}")
        with right:
            st.markdown("#### Researchers' interpretation")
            st.write(first_nonblank(row, "Authors_Interpretation"))
            st.markdown("#### Limitations and caveats for this finding")
            st.write(first_nonblank(row, "Limitations_or_Caveats"))
            main_limitations = clean_value(row.get("_MainLimitations", ""))
            if main_limitations:
                st.markdown("#### Study-level limitations")
                st.write(main_limitations)

        with st.expander("Analysis details and original statistical values", expanded=False):
            a, b = st.columns(2)
            with a:
                display_field("Group or contrast", row.get("Group_or_Contrast", ""))
                display_field("Analysis type", row.get("Analysis_Type", ""))
                display_field("Statistical test", row.get("Statistical_Test", ""))
                display_field("Result direction (as entered)", row.get("Result_Direction", ""))
            with b:
                display_field("Effect estimate", row.get("Effect_Estimate", ""))
                display_field("P-value", row.get("P_Value", ""))
                display_field("Confidence interval (as entered)", row.get("Confidence_Interval", ""))
                display_field("Other statistics", row.get("Other_stadistics", ""))

        with st.expander("Study context", expanded=False):
            study = study_lookup.get(study_id, {})
            fields_to_show = [
                ("Study design", "Study_Design"),
                ("Research objective", "Research_Objective"),
                ("Population", "Population_Description"),
                ("Total sample size", "Total_N"),
                ("EEG recording conditions", "EEG_Recording_Conditions"),
                ("Clinical measures reported", "Clinical_Measures"),
                ("Diagnostic phenotypes", "Diagnostic_Phenotypes"),
                ("Phenotype definition", "Phenotype_Definition"),
                ("Peer-review status", "Peer_Review_Status"),
                ("Evidence category", "Evidence_Category"),
            ]
            for label_text, column_name in fields_to_show:
                value = study.get(column_name, "")
                if clean_value(value):
                    st.markdown(f"**{label_text}:** {clean_value(value)}")

        source_url = format_source_url(row.get("_DOI", ""))
        if source_url:
            st.markdown(f"[Open study source / DOI]({source_url})")
        elif title != "Study title not available":
            st.caption("No usable DOI/source link is recorded for this study.")


# -----------------------------------------------------------------------------
# Load the workbook
# -----------------------------------------------------------------------------

workbook_path = Path(__file__).resolve().parent / "OA_NeuroEvidence_Template.xlsx"

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Evidence synthesis · Computational research tool</div>
      <div class="hero-title">🧠 AO-BRAIN</div>
      <div class="hero-subtitle">
        Exploring reported relationships between resting-state EEG patterns, pain,
        and physical functioning in knee osteoarthritis.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "AO-BRAIN organizes findings from published studies so users can explore the "
    "reported EEG patterns, clinical measures, researchers' interpretations, and study limitations. "
    "It is an evidence-exploration tool, not a diagnostic or predictive system."
)

if not workbook_path.exists():
    st.error(
        f"The Excel file was not found: `{workbook_path.name}`. "
        "Place `OA_NeuroEvidence_Template.xlsx` in the same folder as this Python file, "
        "then run the app again."
    )
    st.stop()

try:
    modified_time = workbook_path.stat().st_mtime
    studies, findings = load_workbook(str(workbook_path), modified_time)
except Exception as error:
    st.error("The workbook could not be read. Check the sheet names and required columns.")
    st.exception(error)
    st.stop()

findings, study_lookup = add_finding_context(findings, studies)

if findings.empty:
    st.warning("The Findings sheet has no data rows yet.")
    st.stop()

# Warn about workbook integrity without interrupting the app.
known_study_ids = set(studies["Study_ID"].dropna().astype(str).str.strip())
known_study_ids.discard("")
finding_study_ids = set(findings["Study_ID"].astype(str).str.strip())
finding_study_ids.discard("")
unmatched_ids = sorted(finding_study_ids - known_study_ids)

# -----------------------------------------------------------------------------
# Sidebar filters
# -----------------------------------------------------------------------------

st.sidebar.title("Explore the evidence")
st.sidebar.caption("Filters apply across the charts, result panels, and findings table.")

resting_only = st.sidebar.checkbox("Focus on resting-state EEG", value=True)

outcome_options = ["All domains", "Pain", "Physical function", "Pain and physical function", "Other / unspecified"]
selected_domain = st.sidebar.selectbox("Clinical domain", outcome_options, index=0)

band_options = unique_options(findings, "EEG_Band")
selected_bands = st.sidebar.multiselect("EEG band", band_options, default=[])

measure_options = unique_options(findings, "Clinical_Measure")
selected_measures = st.sidebar.multiselect("Clinical measure", measure_options, default=[])

metric_options = unique_options(findings, "EEG_Metric")
selected_metrics = st.sidebar.multiselect("EEG metric", metric_options, default=[])

region_options = unique_options(findings, "Brain_Region")
selected_regions = st.sidebar.multiselect("Brain region", region_options, default=[])

direction_options = sorted(findings["_DirectionCategory"].dropna().unique().tolist(), key=str.casefold)
selected_directions = st.sidebar.multiselect("Reported result", direction_options, default=[])

evidence_options = unique_options(findings, "Evidence_Category")
selected_evidence = st.sidebar.multiselect("Evidence category", evidence_options, default=[])

analysis_options = unique_options(findings, "Analysis_Type")
selected_analysis = st.sidebar.multiselect("Analysis type", analysis_options, default=[])

test_options = unique_options(findings, "Statistical_Test")
selected_tests = st.sidebar.multiselect("Statistical test", test_options, default=[])

search_term = st.sidebar.text_input("Search text", placeholder="e.g., alpha, WOMAC, correlation")

# Apply filters one by one. This keeps each control simple and makes the process
# easy to explain and maintain.
filtered = findings.copy()

if resting_only:
    resting_mask = filtered["EEG_State"].str.contains("rest", case=False, na=False)
    # If the workbook contains state values, restrict to those mentioning rest.
    # If it does not, leave the records visible and tell the user below.
    if resting_mask.any():
        filtered = filtered[resting_mask].copy()
    elif filtered["EEG_State"].str.strip().ne("").any():
        filtered = filtered.iloc[0:0].copy()

if selected_domain != "All domains":
    if selected_domain == "Pain":
        allowed_domains = {"Pain", "Pain and physical function"}
    elif selected_domain == "Physical function":
        allowed_domains = {"Physical function", "Pain and physical function"}
    else:
        allowed_domains = {selected_domain}
    filtered = filtered[filtered["_ClinicalDomain"].isin(allowed_domains)].copy()

filtered = apply_multiselect(filtered, "EEG_Band", selected_bands)
filtered = apply_multiselect(filtered, "Clinical_Measure", selected_measures)
filtered = apply_multiselect(filtered, "EEG_Metric", selected_metrics)
filtered = apply_multiselect(filtered, "Brain_Region", selected_regions)
filtered = apply_multiselect(filtered, "_DirectionCategory", selected_directions)
filtered = apply_multiselect(filtered, "Evidence_Category", selected_evidence)
filtered = apply_multiselect(filtered, "Analysis_Type", selected_analysis)
filtered = apply_multiselect(filtered, "Statistical_Test", selected_tests)

if search_term.strip():
    searchable_columns = [
        "Finding_ID", "Study_ID", "_StudyTitle", "_Authors", "EEG_Band", "Brain_Region",
        "EEG_Metric", "Clinical_Outcome", "Clinical_Measure", "Group_or_Contrast",
        "Analysis_Type", "Statistical_Test", "Result_Direction", "Finding_Summary",
        "Authors_Interpretation", "Limitations_or_Caveats", "_MainLimitations",
    ]
    search_pattern = re.escape(search_term.strip())
    search_mask = pd.Series(False, index=filtered.index)
    for column in searchable_columns:
        if column in filtered.columns:
            search_mask |= filtered[column].astype(str).str.contains(
                search_pattern, case=False, regex=True, na=False
            )
    filtered = filtered[search_mask].copy()

# -----------------------------------------------------------------------------
# Key numbers
# -----------------------------------------------------------------------------

valid_study_ids = filtered["Study_ID"].replace("", pd.NA).dropna()
unique_studies_in_view = int(valid_study_ids.nunique())
unique_bands_in_view = int(filtered.loc[filtered["EEG_Band"].ne(""), "EEG_Band"].nunique())
unique_domains_in_view = int(filtered["_ClinicalDomain"].nunique()) if not filtered.empty else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Studies represented", f"{unique_studies_in_view}")
k2.metric("Finding records", f"{len(filtered)}")
k3.metric("EEG bands represented", f"{unique_bands_in_view}")
k4.metric("Clinical domains represented", f"{unique_domains_in_view}")

if resting_only and filtered.empty:
    st.info(
        "No findings match the current filters. Check the filters, or turn off "
        "'Focus on resting-state EEG' if you want to inspect all recording states."
    )
if unmatched_ids:
    with st.expander("Data quality note: finding records with no matching study record"):
        st.write(", ".join(unmatched_ids))
        st.caption("Check that these Study_ID values match IDs in the Studies sheet.")

st.caption(
    "Counts represent records in the current filtered view. A single study can contribute "
    "multiple findings, so finding counts and study counts are not interchangeable."
)

# -----------------------------------------------------------------------------
# Tabs: keep the previous plots and tables, and add interactive evidence panels.
# -----------------------------------------------------------------------------

tab_overview, tab_patterns, tab_insights, tab_statistics, tab_studies, tab_data = st.tabs(
    [
        "Evidence map",
        "Reported patterns",
        "Research insights",
        "Statistical results",
        "Study library",
        "Data table",
    ]
)

with tab_overview:
    st.subheader("Evidence map")
    st.write(
        "This map shows where the included literature has reported findings. It counts unique "
        "studies for each EEG-band and clinical-domain combination; it does not measure the "
        "strength of a relationship or indicate whether the result was positive or negative."
    )

    if filtered.empty:
        st.info("No data to plot for the current selection.")
    else:
        map_data = filtered.copy()
        map_data = map_data[map_data["Study_ID"].ne("")].drop_duplicates(
            subset=["Study_ID", "_BandForChart", "_ClinicalDomain"]
        )
        if not map_data.empty:
            pivot = map_data.pivot_table(
                index="_BandForChart",
                columns="_ClinicalDomain",
                values="Study_ID",
                aggfunc="nunique",
                fill_value=0,
            )
            if not pivot.empty:
                fig_map = px.imshow(
                    pivot,
                    text_auto=True,
                    aspect="auto",
                    labels={
                        "x": "Clinical domain",
                        "y": "EEG band",
                        "color": "Unique studies",
                    },
                    title="Unique studies by EEG band and clinical domain",
                    color_continuous_scale="Blues",
                )
                fig_map.update_layout(margin=dict(l=15, r=15, t=55, b=15))
                st.plotly_chart(fig_map, use_container_width=True)

        left, right = st.columns(2)
        with left:
            measure_counts = (
                filtered.assign(
                    _MeasureLabel=filtered["Clinical_Measure"].map(lambda x: clean_value(x) or "Not reported")
                )
                .groupby("_MeasureLabel", as_index=False)
                .agg(
                    Unique_studies=("Study_ID", "nunique"),
                    Finding_records=("Finding_ID", "count"),
                )
                .sort_values("Unique_studies", ascending=True)
            )
            if not measure_counts.empty:
                fig_measure = px.bar(
                    measure_counts,
                    x="Unique_studies",
                    y="_MeasureLabel",
                    orientation="h",
                    hover_data=["Finding_records"],
                    title="Studies by clinical measure",
                    labels={"_MeasureLabel": "Clinical measure", "Unique_studies": "Unique studies"},
                )
                st.plotly_chart(fig_measure, use_container_width=True)
        with right:
            band_counts = (
                filtered.assign(
                    _BandLabel=filtered["EEG_Band"].map(lambda x: clean_value(x) or "Not reported")
                )
                .groupby("_BandLabel", as_index=False)
                .agg(
                    Unique_studies=("Study_ID", "nunique"),
                    Finding_records=("Finding_ID", "count"),
                )
                .sort_values("Unique_studies", ascending=True)
            )
            if not band_counts.empty:
                fig_band = px.bar(
                    band_counts,
                    x="Unique_studies",
                    y="_BandLabel",
                    orientation="h",
                    hover_data=["Finding_records"],
                    title="Studies by EEG band",
                    labels={"_BandLabel": "EEG band", "Unique_studies": "Unique studies"},
                )
                st.plotly_chart(fig_band, use_container_width=True)

    st.divider()
    st.subheader("How to interpret this map")
    st.markdown(
        "- A higher count means more studies in this database reported findings for that combination.\n"
        "- It does **not** mean that the relationship is stronger, more reliable, or causal.\n"
        "- A zero means the current database has no matching records for that cell; it does not prove that no relationship exists."
    )

with tab_patterns:
    st.subheader("Reported patterns by clinical domain")
    st.write(
        "These charts summarize the direction labels entered from the studies. The labels retain "
        "the authors' reported meaning where possible; they should be interpreted together with "
        "the EEG metric, clinical scale, contrast, and original study."
    )

    if filtered.empty:
        st.info("No findings match the current filters.")
    else:
        pattern_counts = (
            filtered.groupby(["_ClinicalDomain", "_DirectionCategory"], as_index=False)
            .agg(
                Finding_records=("Finding_ID", "count"),
                Unique_studies=("Study_ID", "nunique"),
            )
        )
        fig_direction = px.bar(
            pattern_counts,
            x="_ClinicalDomain",
            y="Finding_records",
            color="_DirectionCategory",
            barmode="group",
            hover_data=["Unique_studies"],
            title="Reported result labels by clinical domain",
            labels={
                "_ClinicalDomain": "Clinical domain",
                "Finding_records": "Finding records",
                "_DirectionCategory": "Reported result label",
            },
        )
        st.plotly_chart(fig_direction, use_container_width=True)

        analysis_counts = (
            filtered.assign(
                _AnalysisLabel=filtered["Analysis_Type"].map(lambda x: clean_value(x) or "Not reported")
            )
            .groupby("_AnalysisLabel", as_index=False)
            .agg(Finding_records=("Finding_ID", "count"), Unique_studies=("Study_ID", "nunique"))
            .sort_values("Finding_records", ascending=True)
        )
        fig_analysis = px.bar(
            analysis_counts,
            x="Finding_records",
            y="_AnalysisLabel",
            orientation="h",
            hover_data=["Unique_studies"],
            title="Analysis types represented in the selected findings",
            labels={"_AnalysisLabel": "Analysis type", "Finding_records": "Finding records"},
        )
        st.plotly_chart(fig_analysis, use_container_width=True)

        st.markdown("#### Findings with different reported directions")
        st.caption(
            "This is a screening aid, not an automatic contradiction detector. Different directions "
            "may arise from different measures, brain regions, contrasts, adjustments, or methods. "
            "Review the original papers before interpreting such records as inconsistent."
        )
        comparable_directions = filtered[
            filtered["_DirectionCategory"].isin(["Positive association", "Negative association"])
        ].copy()
        comparison_columns = [
            "_BandForChart", "Clinical_Outcome", "_MeasureForChart", "_MetricForChart", "_RegionForChart"
        ]
        if not comparable_directions.empty:
            direction_review = (
                comparable_directions.groupby(comparison_columns, dropna=False)
                .agg(
                    Directions=("_DirectionCategory", lambda s: ", ".join(sorted(set(s)))),
                    Unique_studies=("Study_ID", "nunique"),
                    Finding_IDs=("Finding_ID", lambda s: ", ".join(sorted({clean_value(x) for x in s if clean_value(x)}))),
                )
                .reset_index()
            )
            direction_review = direction_review[
                direction_review["Directions"].str.contains("Positive association", na=False)
                & direction_review["Directions"].str.contains("Negative association", na=False)
                & (direction_review["Unique_studies"] >= 2)
            ]
            if direction_review.empty:
                st.info(
                    "No combinations with both positive and negative association labels were found "
                    "under the current filters. This does not establish that the literature is consistent."
                )
            else:
                direction_review = direction_review.rename(
                    columns={
                        "_BandForChart": "EEG_Band",
                        "Clinical_Outcome": "Clinical_Outcome",
                        "_MeasureForChart": "Clinical_Measure",
                        "_MetricForChart": "EEG_Metric",
                        "_RegionForChart": "Brain_Region",
                    }
                )
                st.dataframe(direction_review, use_container_width=True, hide_index=True)
        else:
            st.info("There are no positive/negative association labels in this selected view to compare.")

with tab_insights:
    st.subheader("Research insights")
    st.write(
        "Explore each finding together with the researchers' own interpretation and the limitations "
        "recorded for that finding and its study. These panels use the existing Excel fields; no new "
        "scientific conclusions are generated automatically."
    )

    st.info(
        "Tip: choose an EEG band and a clinical domain in the sidebar. The panels below will show "
        "only the matching records, with the source study and caveats attached."
    )

    if filtered.empty:
        st.info("No findings match the current filters. Broaden the selection to see evidence panels.")
    else:
        with st.expander("What is included in these panels?", expanded=False):
            st.markdown(
                "- **Reported finding:** `Finding_Summary` from the database.\n"
                "- **Researchers' interpretation:** `Authors_Interpretation` from the database.\n"
                "- **Finding-level caveats:** `Limitations_or_Caveats`.\n"
                "- **Study-level limitations:** `Main_Limitations` from the Studies sheet.\n"
                "- **Source and context:** title, authors, year, design, population, and DOI/source link when available."
            )

        st.caption(f"Showing {len(filtered)} finding panel(s) in the current selection.")
        for _, finding_row in filtered.iterrows():
            render_finding_panel(finding_row, study_lookup)

with tab_statistics:
    st.subheader("Statistical results as reported")
    st.write(
        "The table preserves the statistical values as they were entered in Excel. Confidence intervals "
        "are not silently reconstructed or replaced when their format is unclear."
    )

    stat_columns = [
        "Finding_ID", "Study_ID", "_StudyTitle", "EEG_Band", "EEG_Metric", "Brain_Region",
        "Clinical_Outcome", "Clinical_Measure", "Group_or_Contrast", "Analysis_Type",
        "Statistical_Test", "Result_Direction", "Effect_Estimate", "P_Value",
        "Confidence_Interval", "Other_stadistics",
    ]
    stats_table = filtered[stat_columns].copy() if not filtered.empty else pd.DataFrame(columns=stat_columns)
    stats_table = stats_table.rename(columns={"_StudyTitle": "Study_Title", "Other_stadistics": "Other_statistics"})
    st.dataframe(stats_table, use_container_width=True, hide_index=True)

    st.download_button(
        "Download statistical results as CSV",
        data=stats_table.to_csv(index=False).encode("utf-8-sig"),
        file_name="ao_brain_statistical_results.csv",
        mime="text/csv",
        disabled=stats_table.empty,
    )

    st.markdown("#### Optional effect-estimate plot")
    st.caption(
        "The plot appears only when at least two records have numeric estimates and confidence intervals, "
        "and the selected records share the same band, clinical measure, EEG metric, outcome, test, and contrast. "
        "Even then, verify the original papers before treating estimates as directly comparable."
    )

    if not filtered.empty:
        effect_rows = filtered.copy()
        effect_rows["_EstimateNumeric"] = effect_rows["Effect_Estimate"].map(parse_first_number)
        effect_rows["_CIParsed"] = effect_rows["Confidence_Interval"].map(parse_confidence_interval)
        effect_rows = effect_rows[
            effect_rows["_EstimateNumeric"].notna() & effect_rows["_CIParsed"].notna()
        ].copy()

        if not effect_rows.empty:
            effect_rows["_CILower"] = effect_rows["_CIParsed"].map(lambda value: value[0])
            effect_rows["_CIUpper"] = effect_rows["_CIParsed"].map(lambda value: value[1])
            effect_rows = effect_rows[
                (effect_rows["_CILower"] <= effect_rows["_EstimateNumeric"])
                & (effect_rows["_EstimateNumeric"] <= effect_rows["_CIUpper"])
            ].copy()

        dimensions_for_plot = [
            "EEG_Band", "Clinical_Measure", "EEG_Metric", "Clinical_Outcome",
            "Analysis_Type", "Statistical_Test", "Group_or_Contrast",
        ]
        same_comparison = all(
            effect_rows[column].map(lambda x: clean_value(x) or "Not reported").nunique() == 1
            for column in dimensions_for_plot
        ) if not effect_rows.empty else False

        if len(effect_rows) >= 2 and effect_rows["Study_ID"].nunique() >= 2 and same_comparison:
            effect_rows["_FindingLabel"] = effect_rows.apply(
                lambda row: f"{clean_value(row['Finding_ID']) or 'Finding'} · {clean_value(row['_StudyTitle'])[:60]}",
                axis=1,
            )
            fig_effect = go.Figure()
            fig_effect.add_trace(
                go.Scatter(
                    x=effect_rows["_EstimateNumeric"],
                    y=effect_rows["_FindingLabel"],
                    mode="markers",
                    marker=dict(size=10),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=(effect_rows["_CIUpper"] - effect_rows["_EstimateNumeric"]).tolist(),
                        arrayminus=(effect_rows["_EstimateNumeric"] - effect_rows["_CILower"]).tolist(),
                        visible=True,
                    ),
                    customdata=effect_rows[["Study_ID", "Effect_Estimate", "Confidence_Interval"]].to_numpy(),
                    hovertemplate=(
                        "Finding: %{y}<br>Estimate parsed: %{x}<br>Study ID: %{customdata[0]}"
                        "<br>Original estimate: %{customdata[1]}<br>Original CI: %{customdata[2]}<extra></extra>"
                    ),
                )
            )
            fig_effect.update_layout(
                title="Effect estimates and reported confidence intervals",
                xaxis_title="Parsed effect estimate (original value shown on hover)",
                yaxis_title="Finding",
                height=max(340, 95 * len(effect_rows)),
                margin=dict(l=10, r=20, t=55, b=35),
            )
            st.plotly_chart(fig_effect, use_container_width=True)
        else:
            st.info(
                "No effect-estimate plot was generated for this selection. There may be too few numeric "
                "estimates with readable intervals, or the records may use different measures, tests, or contrasts. "
                "The original values remain available in the table above."
            )
    else:
        st.info("No statistical records match the current filters.")

with tab_studies:
    st.subheader("Study library")
    st.write(
        "Study-level information is shown separately from individual findings because one study can "
        "contribute several distinct findings. The list below follows the current sidebar filters."
    )

    current_ids = set(filtered["Study_ID"].dropna().astype(str)) if not filtered.empty else set()
    study_view = studies[studies["Study_ID"].isin(current_ids)].copy()
    study_view = study_view[study_view["Study_ID"].ne("")].drop_duplicates("Study_ID")

    study_table_columns = [
        "Study_ID", "Full_Title", "Authors", "Year", "Study_Design", "Total_N",
        "Evidence_Category", "Peer_Review_Status", "DOI_URL",
    ]
    study_table = study_view[study_table_columns] if not study_view.empty else pd.DataFrame(columns=study_table_columns)
    st.dataframe(study_table, use_container_width=True, hide_index=True)

    if not study_view.empty:
        study_labels = {
            row["Study_ID"]: f"{row['Study_ID']} — {(row['Full_Title'] or 'Untitled study')[:100]}"
            for _, row in study_view.iterrows()
        }
        selected_study_id = st.selectbox(
            "Choose a study to inspect its full record",
            options=study_view["Study_ID"].tolist(),
            format_func=lambda sid: study_labels.get(sid, sid),
        )
        selected_study = study_view[study_view["Study_ID"] == selected_study_id].iloc[0]

        st.markdown("#### Study details")
        detail_columns = st.columns(2)
        study_fields = [column for column in STUDY_COLUMNS if column in selected_study.index and column != "Study_ID"]
        for index, column_name in enumerate(study_fields):
            with detail_columns[index % 2]:
                display_field(column_name.replace("_", " "), selected_study[column_name])

        source_url = format_source_url(selected_study.get("DOI_URL", ""))
        if source_url:
            st.markdown(f"[Open study source / DOI]({source_url})")

        related_findings = filtered[filtered["Study_ID"] == selected_study_id]
        st.markdown(f"#### Findings from this study in the current view ({len(related_findings)})")
        related_columns = [
            "Finding_ID", "EEG_Band", "Brain_Region", "EEG_Metric", "Clinical_Outcome",
            "Clinical_Measure", "Result_Direction", "Finding_Summary",
        ]
        st.dataframe(related_findings[related_columns], use_container_width=True, hide_index=True)
    else:
        st.info("No study records match the current filters. Broaden the selection in the sidebar.")

with tab_data:
    st.subheader("Findings database")
    st.write(
        "This view keeps the extracted records available for checking and export. Internal helper fields "
        "used for filtering and charts are not included in the downloaded file."
    )
    data_columns = [column for column in FINDING_COLUMNS if column in filtered.columns]
    export_table = filtered[data_columns].copy()
    st.dataframe(export_table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered findings as CSV",
        data=export_table.to_csv(index=False).encode("utf-8-sig"),
        file_name="ao_brain_filtered_findings.csv",
        mime="text/csv",
        disabled=export_table.empty,
    )

    st.markdown("#### Database coverage")
    coverage_rows = []
    for column in [
        "Finding_Summary", "Authors_Interpretation", "Limitations_or_Caveats",
        "Effect_Estimate", "P_Value", "Confidence_Interval", "Group_or_Contrast",
    ]:
        available = filtered[column].map(lambda value: bool(clean_value(value))).sum() if column in filtered.columns else 0
        coverage_rows.append(
            {
                "Field": column,
                "Records with information": int(available),
                "Records in current view": int(len(filtered)),
                "Coverage (%)": round((available / len(filtered) * 100), 1) if len(filtered) else 0,
            }
        )
    coverage_table = pd.DataFrame(coverage_rows)
    st.dataframe(coverage_table, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Scientific scope and limitations
# -----------------------------------------------------------------------------

st.divider()
with st.expander("Scientific scope and limitations of AO-BRAIN", expanded=False):
    st.markdown(
        "**What the app can do:** organize extracted findings, compare reported patterns across "
        "studies, display the authors' interpretations, show documented caveats, and identify areas "
        "where the current database has little or no recorded evidence.\n\n"
        "**What it cannot establish on its own:** a new patient-level correlation, causality, a validated "
        "biomarker, a clinical diagnosis, or a prediction of future pain or functional deterioration. "
        "Those claims would require appropriate underlying data and study designs.\n\n"
        "**Important:** a count of studies is not a measure of evidence quality. Results may differ because "
        "of clinical scales, EEG metrics, brain regions, sample characteristics, statistical methods, or "
        "adjustments. The original studies should be checked before interpreting apparently similar or "
        "different findings as directly comparable."
    )

st.caption(
    "AO-BRAIN · Evidence exploration based on the studies and findings recorded in the supplied workbook. "
    "Check extracted values against the original papers before using them in a report."
)
