# ============================================================
# app.py — Ontario University Recommender (Streamlit)
# ============================================================
# A web interface for the weighted scoring recommender.
# All logic from the notebook Sections 1–6 lives here,
# reorganized into functions and wrapped with Streamlit UI.
#
# To run locally:
#   streamlit run app.py
#
# File dependencies (must be in the same folder):
#   - 5301_training_dataset.csv
#   - GNG5301_Data_Audit_Spreadsheet_updated.xlsx
# ============================================================


import streamlit as st
import pandas as pd
import numpy as np

# ── Page config ──────────────────────────────────────────────
# Must be the very first Streamlit call in the file.
st.set_page_config(
    page_title="Ontario University Recommender",
    page_icon="🎓",
    layout="wide",
)


# ============================================================
# SECTION 1 — DATA LOADING
# Cached so it only runs once per session, not on every
# widget interaction. st.cache_data is Streamlit's built-in
# memoization — identical to loading the files once at the
# top of a script, but safe for multi-user web contexts.
# ============================================================

@st.cache_data
def load_data():
    OUR_10 = [
        "University of Ottawa", "University of Waterloo",
        "University of Toronto", "McMaster University",
        "Queen's University", "Western University",
        "York University", "Carleton University",
        "University of Guelph", "Ontario Tech University",
    ]

    df_csv = pd.read_csv("5301_training_dataset.csv")
    df_csv["school_name"] = df_csv["school_name"].replace({
        "Western - Main Campus": "Western University"
    })
    df_csv = df_csv[df_csv["school_name"].isin(OUR_10)].copy().reset_index(drop=True)

    SCHOOL_COLS = [
        "university_name", "city", "region", "campus_type",
        "col_score", "sustainability_score", "campus_feel_score",
        "equity_score", "coop_placement_rate", "programs_coop_count",
        "intl_student_pct", "residence_capacity", "bilingual",
        "avg_entering_grade", "qsranking_metric", "total_enrollment",
    ]
    df_school = pd.read_excel(
        "GNG5301_Data_Audit_Spreadsheet_updated.xlsx",
        sheet_name="ML Feature Matrix",
        header=2
    )[SCHOOL_COLS].dropna(subset=["university_name"]).reset_index(drop=True)

    df = df_csv.merge(df_school, left_on="school_name",
                      right_on="university_name", how="left")
    return df


# ============================================================
# SECTION 2 — FEATURE ENGINEERING
# Same logic as the notebook, wrapped in a function so it
# can be cached and called once at app startup.
# ============================================================

@st.cache_data
def build_latest(df):
    CATEGORY_MAP = {
        "Engineering":                             "Engineering & Technology",
        "Computer Science":                        "Engineering & Technology",
        "Mathematics":                             "Engineering & Technology",
        "Physical Science":                        "Engineering & Technology",
        "Nursing":                                 "Health & Medical",
        "Pharmacy":                                "Health & Medical",
        "Dentistry":                               "Health & Medical",
        "Optometry":                               "Health & Medical",
        "Medicine & Related Programs":             "Health & Medical",
        "Other Health Professions":                "Health & Medical",
        "Therapy & Rehabilitation":                "Health & Medical",
        "Kinesiology, Recreation & Phys. Educ.":  "Health & Medical",
        "Veterinary Medicine":                     "Health & Medical",
        "Agriculture & Biological Science":        "Science & Environment",
        "Forestry":                                "Science & Environment",
        "Food Science & Nutrition":                "Food Science",
        "Architecture":                            "Architecture",
        "Business & Commerce":                     "Business",
        "Fine & Applied Arts":                     "Arts, Humanities & Social Sciences",
        "Humanities":                              "Arts, Humanities & Social Sciences",
        "Journalism":                              "Arts, Humanities & Social Sciences",
        "Social Science":                          "Arts, Humanities & Social Sciences",
        "Other Arts & Science":                    "Arts, Humanities & Social Sciences",
        "Theology":                                "Arts, Humanities & Social Sciences",
        "Education":                               "Education",
        "Law":                                     "Law",
    }

    def minmax(series):
        s = pd.to_numeric(series, errors="coerce")
        mn, mx = s.min(), s.max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            return pd.Series(0.5, index=s.index)
        return (s - mn) / (mx - mn)

    df2 = df.copy()
    df2["category"] = df2["program_name"].apply(
        lambda p: CATEGORY_MAP.get(str(p).strip(), "Other") if pd.notna(p) else "Other"
    )

    latest = (
        df2.sort_values(["school_name", "program_name", "year"])
           .groupby(["school_name", "program_name"], as_index=False)
           .tail(1)
           .copy()
           .reset_index(drop=True)
    )

    latest["employment_score"]        = minmax(latest["employment_6m"])
    latest["domestic_cost_score"]     = 1 - minmax(latest["domestic_total_cost"])
    latest["intl_cost_score"]         = 1 - minmax(latest["international_total_cost"])
    latest["col_score_norm"]          = 1 - minmax(latest["col_score"])
    latest["sustainability_score_norm"]= minmax(latest["sustainability_score"])
    latest["diversity_score_norm"]    = minmax(latest["equity_score"])
    latest["coop_rate_score"]         = minmax(
        latest["coop_placement_rate"].fillna(latest["coop_placement_rate"].median())
    )

    qs = pd.to_numeric(latest["qsranking_metric"], errors="coerce")
    worst = qs.dropna().max()
    latest["ranking_score"] = 1 - minmax(qs.fillna(worst * 1.25))

    admission_flag  = pd.to_numeric(latest["has_any_admission_feature"],   errors="coerce").fillna(0)
    school_flag     = pd.to_numeric(latest["has_any_school_level_feature"],errors="coerce").fillna(0)
    rank_flag       = qs.notna().astype(float)
    latest["data_quality_score"] = 0.45*admission_flag + 0.35*school_flag + 0.20*rank_flag

    breadth = (
        latest.groupby(["school_name","category"])["program_name"]
              .nunique().reset_index(name="category_program_count")
    )
    latest = latest.merge(breadth, on=["school_name","category"], how="left")
    latest["breadth_score"] = minmax(latest["category_program_count"]).fillna(0.0)

    return latest


# ============================================================
# SECTION 4 — SCORING FUNCTIONS
# Pure functions with no UI — identical logic to the notebook.
# ============================================================

def scale(imp):
    return (imp - 1) / 4

def build_weights(s):
    raw = {
        "employment":     0.05 + 0.25 * scale(s["imp_jobs"]),
        "cost":           0.05 + 0.20 * scale(s["imp_cost"]),
        "ranking":        0.03 + 0.18 * scale(s["imp_rank"]),
        "sustainability": 0.02 + 0.10 * scale(s["imp_sust"]),
        "diversity":      0.02 + 0.10 * scale(s["imp_div"]),
        "coop":           0.03 + 0.15 * scale(s["imp_coop"]),
        "admission":      0.15,
        "data_quality":   0.05,
    }
    total = sum(raw.values())
    return {k: v/total for k, v in raw.items()}

def admission_fit_score(student_avg, program_avg):
    if pd.isna(program_avg): return 0.45
    diff = float(student_avg) - float(program_avg)
    if diff >=  5: return 1.00
    if diff >=  2: return 0.85
    if diff >=  0: return 0.70
    if diff >= -2: return 0.50
    if diff >= -4: return 0.35
    return 0.20

def campus_feel_match(pref, feel):
    if pref is None or pd.isna(feel): return 0.5
    return 1.0 - abs(float(pref) - float(feel)) / 4.0

def score_candidates(latest, student, top_n=3):
    if student["category"] != "No preference":
        cand = latest[latest["category"] == student["category"]].copy()
    else:
        cand = latest.copy()

    if cand.empty:
        return pd.DataFrame()

    cand = cand[
        cand["admission_overall_average"].apply(
            lambda a: True if pd.isna(a)
            else (float(student["avg"]) - float(a)) >= -5
        )
    ].copy().reset_index(drop=True)

    if cand.empty:
        return pd.DataFrame()

    weights = build_weights(student)
    status  = student["student_status"]

    cand["adm_score"]    = cand["admission_overall_average"].apply(
                               lambda a: admission_fit_score(student["avg"], a))
    cand["cost_score"]   = (cand["domestic_cost_score"] if status == "Domestic"
                            else cand["intl_cost_score"]).fillna(0.5)
    cand["feel_score"]   = cand["campus_feel_score"].apply(
                               lambda f: campus_feel_match(student["feel_pref"], f))
    cand["combined_cost"]= 0.70*cand["cost_score"] + 0.30*cand["col_score_norm"].fillna(0.5)

    cand["c_employment"]     = weights["employment"]     * cand["employment_score"].fillna(0.5)
    cand["c_cost"]           = weights["cost"]           * cand["combined_cost"]
    cand["c_ranking"]        = weights["ranking"]        * cand["ranking_score"].fillna(0.1)
    cand["c_sustainability"] = weights["sustainability"] * cand["sustainability_score_norm"].fillna(0.5)
    cand["c_diversity"]      = weights["diversity"]      * cand["diversity_score_norm"].fillna(0.5)
    cand["c_coop"]           = weights["coop"]           * cand["coop_rate_score"].fillna(0.5)
    cand["c_admission"]      = weights["admission"]      * cand["adm_score"]
    cand["c_feel"]           = 0.08                      * cand["feel_score"]

    cand["raw_score"]  = (cand["c_employment"] + cand["c_cost"] + cand["c_ranking"] +
                          cand["c_sustainability"] + cand["c_diversity"] + cand["c_coop"] +
                          cand["c_admission"] + cand["c_feel"])
    cand["confidence"] = (0.80 + 0.12*cand["data_quality_score"].fillna(0)
                               + 0.08*cand["breadth_score"].fillna(0))
    cand["final_score"]= cand["raw_score"] * cand["confidence"]

    return cand.sort_values(
        ["final_score","employment_score","ranking_score"],
        ascending=[False,False,False]
    ).reset_index(drop=True).head(top_n)


# ============================================================
# SECTION 5 — DISPLAY HELPERS
# These render Streamlit components instead of print() calls.
# ============================================================

def admission_label(student_avg, program_avg):
    if pd.isna(program_avg): return "No cutoff data", "⚪"
    diff = float(student_avg) - float(program_avg)
    if diff >=  5: return "Strong fit",    "🟢"
    if diff >=  2: return "Good fit",      "🟢"
    if diff >=  0: return "At the cutoff", "🟡"
    if diff >= -3: return "Slight reach",  "🟡"
    return               "Reach",          "🔴"

def why_matched(row, student):
    LABELS = {
        "c_employment":     "strong post-graduation employment",
        "c_cost":           "affordable tuition and cost of living",
        "c_ranking":        "strong global ranking",
        "c_sustainability": "sustainability initiatives",
        "c_diversity":      "diversity and equity programs",
        "c_coop":           "co-op availability",
        "c_admission":      "strong academic profile match",
    }
    IMP_KEYS = {
        "c_employment": "imp_jobs", "c_cost": "imp_cost",
        "c_ranking": "imp_rank",   "c_sustainability": "imp_sust",
        "c_diversity": "imp_div",  "c_coop": "imp_coop",
        "c_admission": None,
    }
    scored = {
        label: row.get(col, 0)
        for col, label in LABELS.items()
        if IMP_KEYS[col] is None or student.get(IMP_KEYS[col], 0) >= 3
    }
    top2 = sorted(scored, key=scored.get, reverse=True)[:2]
    if len(top2) == 1:
        return f"Driven by **{top2[0]}**."
    return f"Driven by **{top2[0]}** and **{top2[1]}**."

def render_card(rank, row, student):
    """Renders one recommendation card using Streamlit components."""
    medals    = ["🥇", "🥈", "🥉"]
    school    = row["school_name"]
    program   = row["program_name"]
    category  = row["category"]
    emp       = row.get("employment_6m", None)
    adm       = row.get("admission_overall_average", None)
    status    = student["student_status"]
    cost_col  = "domestic_total_cost" if status == "Domestic" else "international_total_cost"
    cost_val  = row.get(cost_col, None)
    qs        = row.get("qs_wur_2026_rank", None)

    fit_label, fit_icon = admission_label(student["avg"], adm)
    why = why_matched(row, student)

    with st.container(border=True):
        st.markdown(f"### {medals[rank]}  {school}")
        st.markdown(f"**{program}** &nbsp;|&nbsp; *{category}*")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💵 Annual Cost",
                    f"${cost_val:,.0f}" if pd.notna(cost_val) else "N/A",
                    f"{status}")
        col2.metric("📊 Admission Avg",
                    f"{adm:.1f}%" if pd.notna(adm) else "N/A",
                    f"{fit_icon} {fit_label}",
                    delta_color="off")
        col3.metric("💼 Employment (6m)",
                    f"{emp:.1f}%" if pd.notna(emp) else "N/A")
        col4.metric("🌍 QS Rank",
                    str(qs) if pd.notna(qs) else "Unranked")

        st.caption(f"✨ **Why this matched you:** {why}")

        # Priority score bars — only show dimensions rated ≥ 3
        PRIORITY_ROWS = [
            ("imp_jobs", "employment_score",          "💼 Employment"),
            ("imp_cost", "combined_cost",              "💰 Cost"),
            ("imp_rank", "ranking_score",              "🏅 Ranking"),
            ("imp_sust", "sustainability_score_norm",  "🌱 Sustainability"),
            ("imp_div",  "diversity_score_norm",       "🤝 Diversity"),
            ("imp_coop", "coop_rate_score",            "🔧 Co-op"),
        ]
        active = [(ik, sc, lbl) for ik, sc, lbl in PRIORITY_ROWS
                  if student.get(ik, 0) >= 3]

        if active:
            st.markdown("**Your priority scores for this school:**")
            bar_cols = st.columns(len(active))
            for i, (_, score_col, label) in enumerate(active):
                val = float(row.get(score_col, 0.5))
                bar_cols[i].metric(label, f"{val*100:.0f}%")
                bar_cols[i].progress(val)


def render_comparison(results, student):
    """Renders the side-by-side comparison table."""
    st.markdown("---")
    st.markdown("### 📊 Side-by-Side Comparison")

    status   = student["student_status"]
    cost_col = "domestic_total_cost" if status == "Domestic" else "international_total_cost"

    rows = {
        "School":           [r["school_name"]                                         for _, r in results.iterrows()],
        "Program":          [r["program_name"]                                         for _, r in results.iterrows()],
        "Annual Cost":      [f"${r[cost_col]:,.0f}" if pd.notna(r.get(cost_col)) else "N/A"
                             for _, r in results.iterrows()],
        "Admission Avg":    [f"{r['admission_overall_average']:.1f}%"
                             if pd.notna(r.get('admission_overall_average')) else "N/A"
                             for _, r in results.iterrows()],
        "Employment (6m)":  [f"{r['employment_6m']:.1f}%"
                             if pd.notna(r.get('employment_6m')) else "N/A"
                             for _, r in results.iterrows()],
        "QS Rank":          [str(r.get("qs_wur_2026_rank","Unranked"))                for _, r in results.iterrows()],
        "Sustainability":   [f"{int(r['sustainability_score'])}/5"
                             if pd.notna(r.get('sustainability_score')) else "N/A"
                             for _, r in results.iterrows()],
        "Co-op Programs":   [str(int(r['programs_coop_count']))
                             if pd.notna(r.get('programs_coop_count')) else "N/A"
                             for _, r in results.iterrows()],
    }
    st.dataframe(pd.DataFrame(rows).set_index("School"), use_container_width=True)


# ============================================================
# MAIN APP LAYOUT
# Streamlit reruns this entire file top-to-bottom on every
# widget interaction. st.session_state persists values
# across reruns so the results aren't lost when a slider moves.
# ============================================================

def main():

    # ── Load data once ───────────────────────────────────────
    df     = load_data()
    latest = build_latest(df)

    CATEGORIES = [
        "No preference",
        "Engineering & Technology",
        "Health & Medical",
        "Business",
        "Science & Environment",
        "Arts, Humanities & Social Sciences",
        "Law", "Education", "Architecture", "Food Science",
    ]
    CAMPUS_FEEL_OPTIONS = {
        "Large urban research university (like U of T or York)":   1,
        "Large but campus-focused (like Waterloo or Western)":     3,
        "Medium-sized with strong community feel (like Queen's)":  3,
        "Smaller and more intimate (like Guelph or Ontario Tech)": 4,
        "No preference":                                           None,
    }

    # ── Header ───────────────────────────────────────────────
    st.title("🎓 Ontario University Recommender")
    st.markdown(
        "Answer a few questions and we'll match you with the best-fit "
        "universities from Ontario's top 10 schools — based on what *you* value most."
    )
    st.markdown("---")

    # ── Sidebar — Quiz ───────────────────────────────────────
    # All quiz inputs live in the sidebar so the main panel
    # stays clean for results. Streamlit automatically reruns
    # and re-scores whenever any widget changes.
    with st.sidebar:
        st.header("📋 Your Profile")

        st.subheader("About You")
        status = st.selectbox("Student type", ["Domestic", "International"])
        category = st.selectbox("Area of interest", CATEGORIES)
        avg = st.slider("Your academic average (%)", 60, 100, 80)

        feel_label = st.selectbox(
            "Preferred campus environment",
            list(CAMPUS_FEEL_OPTIONS.keys())
        )
        feel_pref = CAMPUS_FEEL_OPTIONS[feel_label]

        st.subheader("Your Priorities")
        st.caption("Rate each from 1 (not important) to 5 (essential)")

        imp_jobs = st.slider("💼 Employment after graduation", 1, 5, 3)
        imp_cost = st.slider("💰 Low cost (tuition + living)",  1, 5, 3)
        imp_rank = st.slider("🏅 School ranking / prestige",    1, 5, 3)
        imp_sust = st.slider("🌱 Sustainability",               1, 5, 2)
        imp_div  = st.slider("🤝 Diversity & equity",           1, 5, 3)
        imp_coop = st.slider("🔧 Co-op availability",           1, 5, 3)

        st.markdown("---")
        run = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)

    # ── Build student profile ─────────────────────────────────
    student = {
        "student_status": status,
        "category":       category,
        "avg":            float(avg),
        "feel_pref":      feel_pref,
        "imp_jobs":       imp_jobs,
        "imp_cost":       imp_cost,
        "imp_rank":       imp_rank,
        "imp_sust":       imp_sust,
        "imp_div":        imp_div,
        "imp_coop":       imp_coop,
    }

    # Store results in session state so they persist across
    # widget interactions without re-scoring unnecessarily
    if run:
        results = score_candidates(latest, student, top_n=3)
        st.session_state["results"] = results
        st.session_state["student"] = student

    # ── Main panel — Results ─────────────────────────────────
    if "results" in st.session_state and not st.session_state["results"].empty:
        results = st.session_state["results"]
        s       = st.session_state["student"]

        tab_recs, tab_whatif, tab_about = st.tabs([
            "🏆 Recommendations", "🔧 What-If Tool", "ℹ️ About"
        ])

        # ── Tab 1: Recommendations ───────────────────────────
        with tab_recs:
            st.markdown(f"**Showing top {len(results)} matches for a "
                        f"{s['student_status'].lower()} student "
                        f"interested in {s['category']} "
                        f"with a {s['avg']:.0f}% average.**")
            st.markdown("")

            for rank, (_, row) in enumerate(results.iterrows()):
                render_card(rank, row, s)

            render_comparison(results, s)

            st.markdown("---")
            st.info(
                "**Next steps:** Visit each university's official website to explore "
                "specific program requirements. Apply through [OUAC](https://ouac.on.ca)."
            )

        # ── Tab 2: What-If Tool ──────────────────────────────
        # In Streamlit, the what-if tool is just the sidebar
        # sliders reused — changing any slider and clicking
        # 'Get Recommendations' re-scores instantly.
        # This tab explains that and shows the weight breakdown.
        with tab_whatif:
            st.markdown("### 🔧 How Your Priorities Affect Results")
            st.markdown(
                "Adjust the sliders in the sidebar and click "
                "**Get Recommendations** to see how the rankings change. "
                "The table below shows how your current ratings translate "
                "into scoring weights."
            )

            weights = build_weights(s)
            weight_data = {
                "Dimension": [
                    "💼 Employment", "💰 Cost", "🏅 Ranking",
                    "🌱 Sustainability", "🤝 Diversity", "🔧 Co-op",
                    "📊 Admission fit (fixed)", "🔒 Data quality (fixed)"
                ],
                "Your Rating": [
                    f"{s['imp_jobs']}/5", f"{s['imp_cost']}/5",
                    f"{s['imp_rank']}/5", f"{s['imp_sust']}/5",
                    f"{s['imp_div']}/5",  f"{s['imp_coop']}/5",
                    "Fixed", "Fixed"
                ],
                "Weight in Scoring": [
                    f"{weights['employment']*100:.1f}%",
                    f"{weights['cost']*100:.1f}%",
                    f"{weights['ranking']*100:.1f}%",
                    f"{weights['sustainability']*100:.1f}%",
                    f"{weights['diversity']*100:.1f}%",
                    f"{weights['coop']*100:.1f}%",
                    f"{weights['admission']*100:.1f}%",
                    f"{weights['data_quality']*100:.1f}%",
                ],
            }
            st.dataframe(
                pd.DataFrame(weight_data).set_index("Dimension"),
                use_container_width=True
            )

            st.markdown("---")
            st.markdown("### 📊 Score Breakdown for Current Results")
            st.caption("How each school scored on each dimension (0–100%)")

            breakdown_rows = []
            for _, row in results.iterrows():
                breakdown_rows.append({
                    "School":        row["school_name"],
                    "Program":       row["program_name"],
                    "💼 Employment": f"{row.get('employment_score',0)*100:.0f}%",
                    "💰 Cost":       f"{row.get('combined_cost',0)*100:.0f}%",
                    "🏅 Ranking":    f"{row.get('ranking_score',0)*100:.0f}%",
                    "🌱 Sustain.":   f"{row.get('sustainability_score_norm',0)*100:.0f}%",
                    "🤝 Diversity":  f"{row.get('diversity_score_norm',0)*100:.0f}%",
                    "🔧 Co-op":      f"{row.get('coop_rate_score',0)*100:.0f}%",
                    "Final Score":   f"{row.get('final_score',0)*100:.1f}%",
                })
            st.dataframe(
                pd.DataFrame(breakdown_rows).set_index("School"),
                use_container_width=True
            )

        # ── Tab 3: About ─────────────────────────────────────
        with tab_about:
            st.markdown("### About This Tool")
            st.markdown("""
This recommender was built as part of a feasibility study for **GNG5301 — 
Professional Skills and Responsibilities** at the University of Ottawa.

**How it works:**
- Student preferences are collected through the sidebar quiz
- Each university-program pair is scored on 8 dimensions using a weighted scoring model
- Weights are dynamically calculated from the student's importance ratings
- A confidence multiplier adjusts scores based on data quality

**Data sources:**
- CUDO (Common University Data Ontario) — program-level employment, tuition, and admission data
- GNG5301 Data Audit Spreadsheet — school-level cost of living, sustainability, campus feel, and equity scores

**Universities included:**
University of Ottawa · University of Waterloo · University of Toronto · 
McMaster University · Queen's University · Western University · 
York University · Carleton University · University of Guelph · Ontario Tech University

**Team:** GNG5301 Group 5 — Maria Arias Rivera, Ethan Fan, Nada Abdelwahab  
**Course:** GNG5301 Professional Skills and Responsibilities, University of Ottawa  
            """)

    elif "results" in st.session_state and st.session_state["results"].empty:
        st.warning(
            "No programs found matching your profile. "
            "Try lowering your academic average or selecting 'No preference' for area of interest."
        )
    else:
        # First load — show instructions
        st.markdown("### 👈 Fill in your profile in the sidebar to get started")
        col1, col2, col3 = st.columns(3)
        col1.info("**Step 1**\n\nTell us about yourself — student type, area of interest, and average")
        col2.info("**Step 2**\n\nRate what matters to you — employment, cost, ranking, sustainability and more")
        col3.info("**Step 3**\n\nClick **Get Recommendations** and explore your top 3 matches")


if __name__ == "__main__":
    main()
