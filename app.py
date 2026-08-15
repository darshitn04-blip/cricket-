from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px

from analytics import demo_players, score_players, save_to_sqlite, team_strategy, validate

st.set_page_config(page_title="CricoNomics | IPL Auction Value", page_icon="🏏", layout="wide")
st.markdown("""<style>
  .stApp {background: radial-gradient(circle at 10% 0%, #162238 0%, #0d1117 38%, #0d1117 100%); color:#f6f7fb;}
  [data-testid='stMetric'] {background:rgba(255,255,255,.055); border:1px solid rgba(255,255,255,.12); padding:16px; border-radius:14px;}
  h1, h2, h3 {color:#f8bd3e !important;} .stTabs [data-baseweb='tab'] {font-weight:700;}
</style>""", unsafe_allow_html=True)

@st.cache_data
def base_data(): return demo_players()

st.title("CricoNomics  🏏")
st.caption("IPL Auction Value Analyzer · Which players are priced above or below their on-field contribution?")

with st.sidebar:
    st.header("Data controls")
    upload = st.file_uploader("Upload player-season CSV", type="csv", help="See the Data & methodology tab for required columns.")
    st.caption("Default mode uses clearly labelled illustrative portfolio data. Upload audited ball-by-ball and auction data for a real evaluation.")

if upload:
    try:
        raw = validate(pd.read_csv(upload)); data_label = "Uploaded dataset"
    except ValueError as error:
        st.error(str(error)); st.stop()
else:
    raw = base_data(); data_label = "Illustrative portfolio dataset"

scored = score_players(raw)
teams = team_strategy(scored)
save_to_sqlite(scored, teams)

top = scored.iloc[0]
value = scored.loc[scored.verdict.eq("Value pick")].sort_values("price_gap_cr").head(1).iloc[0]
over = scored.loc[scored.verdict.eq("Overpriced")].sort_values("price_gap_cr", ascending=False).head(1).iloc[0]

c1,c2,c3,c4 = st.columns(4)
c1.metric("Top performance index", top.player, f"{top.performance_index:.0f}/100")
c2.metric("Best value pick", value.player, f"₹{abs(value.price_gap_cr):.1f} Cr below model")
c3.metric("Largest premium", over.player, f"₹{over.price_gap_cr:.1f} Cr above model")
c4.metric("Dataset mode", data_label, f"{len(scored)} players")

value_tab, strategy_tab, price_tab, method_tab = st.tabs(["Value board", "Auction strategy", "Price lab", "Data & methodology"])

with value_tab:
    left,right = st.columns((1.55,1))
    left.plotly_chart(px.scatter(scored, x="fair_price_cr", y="auction_price_cr", size="performance_index", color="verdict", hover_name="player", hover_data=["role","team","value_per_crore"], color_discrete_map={"Value pick":"#23c483","Fair value":"#f8bd3e","Overpriced":"#fa5b5b"}, title="Auction price vs modelled fair price", labels={"fair_price_cr":"Modelled fair price (₹ Cr)","auction_price_cr":"Auction price (₹ Cr)"}), use_container_width=True)
    right.subheader("Recruiter-ready insight")
    st.write("The diagonal comparison is the story: above-model players demand a premium; below-model players are efficiency opportunities. The result is role-normalised, so batters and bowlers are not judged by the same raw statistic.")
    st.dataframe(scored[["player","team","role","performance_index","auction_price_cr","fair_price_cr","price_gap_cr","verdict"]].round(2).sort_values("price_gap_cr"), hide_index=True, use_container_width=True)
    selected = st.selectbox("Open a player scouting card", scored.player.tolist())
    player = scored[scored.player.eq(selected)].iloc[0]
    st.info(f"**{player.player}** · {player.role} · {player.team}  |  Performance index: **{player.performance_index:.0f}/100** · Auction: **₹{player.auction_price_cr:.1f} Cr** · Modelled fair price: **₹{player.fair_price_cr:.1f} Cr** · Verdict: **{player.verdict}**")

with strategy_tab:
    a,b = st.columns(2)
    a.plotly_chart(px.bar(teams, x="team", y="total_spend_cr", color="strategy", text_auto=".1f", title="Team spend and auction archetype", labels={"total_spend_cr":"Total sampled spend (₹ Cr)"}), use_container_width=True)
    b.plotly_chart(px.scatter(teams, x="star_spend_share", y="value_per_crore", size="total_spend_cr", color="strategy", hover_name="team", title="Star concentration vs value efficiency", labels={"star_spend_share":"Top-player spend share","value_per_crore":"Average index per ₹ Cr"}), use_container_width=True)
    st.dataframe(teams[["team","strategy","total_spend_cr","star_spend_share","value_per_crore","avg_performance"]].round(2), hide_index=True, use_container_width=True)

with price_tab:
    st.subheader("Next-auction price simulator")
    candidate = st.selectbox("Choose a candidate", scored.player.tolist(), key="candidate")
    row = scored[scored.player.eq(candidate)].iloc[0]
    recent_form = st.slider("Recent-form adjustment", -20, 20, 0, help="Use this to discuss scenarios, not as a production forecast.")
    predicted = max(.2, row.fair_price_cr * (1 + recent_form/100))
    x,y = st.columns(2)
    x.metric("Base model price", f"₹{row.fair_price_cr:.1f} Cr")
    y.metric("Scenario price", f"₹{predicted:.1f} Cr", f"{recent_form:+d}% form adjustment")
    st.caption("The base prediction uses a Random Forest trained on the uploaded/sample player features. A production version would train on historical auction seasons with time-based validation.")

with method_tab:
    st.subheader("What the system does")
    st.markdown("""**Performance Value Index** combines batting impact (runs + strike rate), bowling impact (wickets + economy), and availability. The mix changes by role, then scores are normalised within role.\n\n**Price model** estimates a player-specific fair price from performance features. Price gap = auction price - fair price.\n\n**Team clusters** use spend, star concentration, value efficiency and all-rounder mix to identify star-led, balanced, and value-led auction strategies.""")
    st.subheader("Required CSV fields")
    st.code("player,team,role,auction_price_cr,runs,wickets,strike_rate,economy,matches", language="text")
    st.subheader("Evidence and limitations")
    st.markdown("""The included sample is illustrative for an immediately runnable portfolio demo, not a claim about current player valuations. For real analysis, aggregate ball-by-ball data from [Cricsheet](https://cricsheet.org/downloads/) and combine it with an audited auction-price table. Add season, age, availability, venue split, injury history and player type before using the output for real auction decisions.""")
    st.download_button("Download valuation results", scored.to_csv(index=False).encode(), "criconomics_player_valuations.csv", "text/csv")
    st.code("SELECT player, role, performance_index, auction_price_cr, fair_price_cr, price_gap_cr, verdict\nFROM player_valuations\nORDER BY price_gap_cr;", language="sql")
