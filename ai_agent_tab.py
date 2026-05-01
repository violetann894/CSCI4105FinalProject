import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# ── Configuration ─────────────────────────────────────────────────────────────

# Best practice: Use Streamlit secrets or environment variables
API_KEY = st.secrets.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
GEMINI_MODEL = "gemini-2.5-flash"

# Configure the SDK
genai.configure(api_key=API_KEY)


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    file_path = "COVID-19_Outcomes_by_Vaccination_Status_-_Historical_20260312.csv"
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    numeric_cols = [
        "Unvaccinated Rate", "Vaccinated Rate", "Boosted Rate",
        "Crude Vaccinated Ratio", "Crude Boosted Ratio",
        "Age-Adjusted Unvaccinated Rate", "Age-Adjusted Vaccinated Rate",
        "Age-Adjusted Boosted Rate", "Age-Adjusted Vaccinated Ratio",
        "Age-Adjusted Boosted Ratio",
        "Population Unvaccinated", "Population Vaccinated", "Population Boosted",
        "Outcome Unvaccinated", "Outcome Vaccinated", "Outcome Boosted",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Week End"] = pd.to_datetime(df["Week End"], errors="coerce")
    return df


# ── Dataset Summary ───────────────────────────────────────────────────────────

def build_dataset_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data available."

    outcomes = df["Outcome"].dropna().unique().tolist()
    age_groups = sorted(df["Age Group"].dropna().unique().tolist())
    date_min = df["Week End"].min().strftime("%Y-%m-%d") if not df["Week End"].isna().all() else "N/A"
    date_max = df["Week End"].max().strftime("%Y-%m-%d") if not df["Week End"].isna().all() else "N/A"

    # ── Per age group breakdown by outcome ────────────────────────────────────
    age_order = ["0-4", "5-11", "12-17", "18-29", "30-49", "50-64", "65-79", "80+"]
    breakdown_lines = []
    for outcome in outcomes:
        sub = df[(df["Outcome"] == outcome) & (df["Age Group"] != "All")]
        grp = sub.groupby("Age Group")[["Unvaccinated Rate", "Vaccinated Rate", "Boosted Rate"]].mean().round(2)
        grp = grp.reindex([g for g in age_order if g in grp.index])
        breakdown_lines.append(f"\n{outcome} — average rates per 100,000 people by age group:")
        for age, row in grp.iterrows():
            uv = f"{row['Unvaccinated Rate']:.2f}" if pd.notna(row["Unvaccinated Rate"]) else "N/A"
            v = f"{row['Vaccinated Rate']:.2f}" if pd.notna(row["Vaccinated Rate"]) else "N/A"
            b = f"{row['Boosted Rate']:.2f}" if pd.notna(row["Boosted Rate"]) else "N/A"
            breakdown_lines.append(
                f"  {age:>6}: unvaccinated={uv}, vaccinated={v}, boosted={b}"
            )

    # ── Overall All-ages averages ─────────────────────────────────────────────
    agg_rows = df[df["Age Group"] == "All"]
    overall_lines = []
    for outcome in outcomes:
        sub = agg_rows[agg_rows["Outcome"] == outcome]
        if not sub.empty:
            uv = sub["Unvaccinated Rate"].mean()
            v = sub["Vaccinated Rate"].mean()
            b = sub["Boosted Rate"].mean()
            overall_lines.append(
                f"  {outcome}: unvaccinated={uv:.2f}, vaccinated={v:.2f}, boosted={b:.2f}"
            )

    # ── Population size context ───────────────────────────────────────────────
    latest = df[df["Week End"] == df["Week End"].max()]
    pop_all = latest[latest["Age Group"] == "All"]
    pop_lines = []
    if not pop_all.empty:
        row = pop_all.iloc[0]
        for label, col in [
            ("Unvaccinated", "Population Unvaccinated"),
            ("Vaccinated", "Population Vaccinated"),
            ("Boosted", "Population Boosted"),
        ]:
            if pd.notna(row.get(col)):
                pop_lines.append(f"  {label}: {row[col]:,.0f}")

    return f"""
DATASET: COVID-19 Outcomes by Vaccination Status (Historical)
Date range: {date_min} to {date_max}
Outcomes tracked: {', '.join(outcomes)}
Age groups available: {', '.join(age_groups)}
All rates are per 100,000 people within each vaccination status group.
Vaccination groups: Unvaccinated, Vaccinated (primary series only), Boosted.

OVERALL AVERAGES (All ages combined):
{chr(10).join(overall_lines)}

POPULATION SIZE (most recent week, All ages):
{chr(10).join(pop_lines)}

PER AGE GROUP BREAKDOWN (averages across all weeks):
{chr(10).join(breakdown_lines)}

KEY INTERPRETATION NOTES:
- A higher unvaccinated rate vs vaccinated/boosted rate means vaccination reduced that outcome.
- The 80+ group has the highest absolute rates for deaths and hospitalizations.
- For Cases, rates are closer between groups because cases spread regardless of vaccination status.
- Boosted individuals consistently had the lowest rates for deaths and hospitalizations.
- The 50-64 and 65-79 groups show the clearest benefit of boosting for severe outcomes.
""".strip()


# ── Gemini Chat ───────────────────────────────────────────────────────────────

def get_chat_response(user_input: str, df: pd.DataFrame) -> str:
    summary = build_dataset_summary(df)

    system_instruction = f"""You are a knowledgeable and precise data analyst for a COVID-19 vaccination outcomes dashboard.

You have been given detailed statistics from the dataset below. Use these numbers directly in your answers.

{summary}

ANSWER GUIDELINES:
- Always reference specific numbers from the data above when answering. Do not speak in vague generalities.
- When asked which group is highest or lowest, look at the per-age-group breakdown and state the actual value.
- Compare vaccination groups (unvaccinated vs vaccinated vs boosted) when relevant.
- Keep answers concise but complete — 2 to 4 short paragraphs is ideal.
- If a question is outside the scope of this dataset, say so clearly rather than guessing.
- Give direct, confident answers backed by the numbers rather than excessive hedging.
- Use plain language suitable for a general audience.
- Format responses in Markdown for readability.
"""

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_instruction,
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            max_output_tokens=1000,
        )
    )

    # Convert session history to Gemini format (exclude the latest user message)
    history = []
    for m in st.session_state.agent_messages[:-1]:
        role = "model" if m["role"] == "assistant" else "user"
        history.append({"role": role, "parts": [m["content"]]})

    chat = model.start_chat(history=history)
    response = chat.send_message(user_input)
    return response.text


# ── UI Layout ─────────────────────────────────────────────────────────────────

def render_ai_agent_tab():
    st.header("🤖 COVID-19 Data Assistant")
    st.write(
        "Ask any question about the COVID-19 Outcomes by Vaccination Status dataset. "
        "The assistant can answer questions about trends, age group comparisons, "
        "and vaccination effectiveness."
    )

    df = load_data()
    if df.empty:
        st.warning("Please ensure the CSV file is in the project directory.")
        return

    with st.expander("💡 Example questions you can ask"):
        st.markdown("""
- Which age group had the highest unvaccinated death rate?
- Which age group had the highest vaccination rate?
- How did boosted rates compare to vaccinated rates for hospitalizations?
- Which outcome showed the biggest difference between unvaccinated and boosted groups?
- What does the data show about the 80+ age group compared to younger groups?
- How much did boosting reduce hospitalization rates compared to just being vaccinated?
        """)

    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []

    # Display chat history
    for msg in st.session_state.agent_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Ask about vaccination trends..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.agent_messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                try:
                    reply = get_chat_response(user_input, df)
                    st.markdown(reply)
                    st.session_state.agent_messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"API Error: {e}")

    if st.session_state.agent_messages:
        st.sidebar.button(
            "Clear History",
            on_click=lambda: st.session_state.update(agent_messages=[])
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    st.set_page_config(page_title="COVID Data AI Agent", layout="wide")
    render_ai_agent_tab()
