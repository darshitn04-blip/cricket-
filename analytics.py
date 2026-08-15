"""Explainable valuation analytics for IPL auction decision support."""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

REQUIRED_COLUMNS = {"player", "team", "role", "auction_price_cr", "runs", "wickets", "strike_rate", "economy", "matches"}


def demo_players() -> pd.DataFrame:
    """Illustrative portfolio data. Upload audited data before real decisions."""
    rows = [
        ("Virat Kohli","RCB","Batter",21.0,741,0,154,9.2,15),("Ruturaj Gaikwad","CSK","Batter",18.0,583,0,141,8.9,14),("Shubman Gill","GT","Batter",16.5,426,0,147,9.1,12),
        ("Yashasvi Jaiswal","RR","Batter",18.0,435,0,156,9.0,14),("Suryakumar Yadav","MI","Batter",16.4,345,0,168,9.4,11),("Rishabh Pant","DC","Wicketkeeper",27.0,446,0,155,9.5,13),
        ("Sanju Samson","RR","Wicketkeeper",14.0,531,0,154,8.8,15),("KL Rahul","LSG","Wicketkeeper",14.0,520,0,136,9.2,14),("Ishan Kishan","SRH","Wicketkeeper",11.3,320,0,164,9.0,14),
        ("Jasprit Bumrah","MI","Bowler",18.0,0,20,110,6.5,13),("Mohammed Shami","GT","Bowler",12.0,0,18,95,7.3,12),("Arshdeep Singh","PBKS","Bowler",18.0,0,19,102,8.1,14),
        ("Yuzvendra Chahal","RR","Bowler",6.5,0,18,90,8.2,14),("Kuldeep Yadav","DC","Bowler",10.0,0,16,88,7.4,13),("T Natarajan","SRH","Bowler",10.8,0,19,100,8.5,14),
        ("Rashid Khan","GT","All-rounder",18.0,102,11,145,7.6,12),("Ravindra Jadeja","CSK","All-rounder",18.0,267,8,142,7.8,14),("Hardik Pandya","MI","All-rounder",16.4,216,11,143,9.1,14),
        ("Andre Russell","KKR","All-rounder",12.0,222,19,185,10.1,14),("Sunil Narine","KKR","All-rounder",6.0,488,17,181,6.7,15),("Axar Patel","DC","All-rounder",16.5,235,11,132,7.9,14),
        ("Shashank Singh","PBKS","Batter",0.2,354,0,165,9.3,14),("Abhishek Sharma","SRH","All-rounder",14.0,484,2,204,10.4,16),("Tristan Stubbs","DC","Batter",10.0,378,0,190,9.1,14),
        ("Riyan Parag","RR","All-rounder",14.0,573,3,149,9.7,14),("Harshal Patel","PBKS","Bowler",11.8,0,24,88,9.8,14),("Varun Chakravarthy","KKR","Bowler",8.0,0,21,92,8.0,15),
        ("Pat Cummins","SRH","All-rounder",20.5,136,18,143,9.4,16),("Mitchell Starc","KKR","Bowler",24.8,0,17,94,10.6,14),("Heinrich Klaasen","SRH","Wicketkeeper",23.0,479,0,171,9.3,15),
        ("Nicholas Pooran","LSG","Wicketkeeper",21.0,499,0,178,9.4,14),("Avesh Khan","RR","Bowler",10.0,0,19,96,9.0,16),("Mayank Yadav","LSG","Bowler",11.0,0,7,78,8.1,4),
    ]
    return pd.DataFrame(rows, columns=["player","team","role","auction_price_cr","runs","wickets","strike_rate","economy","matches"])


def validate(data: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    clean = data.copy()
    for col in REQUIRED_COLUMNS - {"player", "team", "role"}:
        clean[col] = pd.to_numeric(clean[col], errors="coerce").fillna(0)
    return clean


def score_players(data: pd.DataFrame) -> pd.DataFrame:
    df = validate(data)
    scaler = MinMaxScaler()
    df["batting_impact"] = .55 * scaler.fit_transform(df[["runs"]]).ravel() + .45 * scaler.fit_transform(df[["strike_rate"]]).ravel()
    wicket_score = scaler.fit_transform(df[["wickets"]]).ravel()
    economy_score = 1 - scaler.fit_transform(df[["economy"]]).ravel()
    df["bowling_impact"] = .70 * wicket_score + .30 * economy_score
    df["availability"] = scaler.fit_transform(df[["matches"]]).ravel()
    df["raw_value"] = np.select(
        [df.role.isin(["Batter", "Wicketkeeper"]), df.role.eq("Bowler")],
        [.70*df.batting_impact + .30*df.availability, .70*df.bowling_impact + .30*df.availability],
        default=.42*df.batting_impact + .42*df.bowling_impact + .16*df.availability,
    )
    # Normalise within role: a player's value is compared to the job they actually perform.
    df["performance_index"] = df.groupby("role")["raw_value"].transform(lambda x: 100 * (x - x.min()) / (x.max() - x.min() + 1e-9))
    features = df[["performance_index", "runs", "wickets", "strike_rate", "economy", "matches"]]
    if len(df) >= 12:
        model = RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42)
        model.fit(features, df.auction_price_cr)
        df["fair_price_cr"] = model.predict(features)
    else:
        df["fair_price_cr"] = df.groupby("role").auction_price_cr.transform("mean")
    df["price_gap_cr"] = df.auction_price_cr - df.fair_price_cr
    df["value_per_crore"] = df.performance_index / df.auction_price_cr.clip(lower=.2)
    df["verdict"] = np.where(df.price_gap_cr > 2.5, "Overpriced", np.where(df.price_gap_cr < -2.5, "Value pick", "Fair value"))
    return df.sort_values("performance_index", ascending=False)


def team_strategy(scored: pd.DataFrame) -> pd.DataFrame:
    team = scored.groupby("team", as_index=False).agg(
        total_spend_cr=("auction_price_cr", "sum"), squad_size=("player", "count"),
        top_player_spend_cr=("auction_price_cr", "max"), avg_performance=("performance_index", "mean"),
        value_per_crore=("value_per_crore", "mean"), all_rounder_share=("role", lambda x: (x == "All-rounder").mean()),
    )
    team["star_spend_share"] = team.top_player_spend_cr / team.total_spend_cr
    features = MinMaxScaler().fit_transform(team[["total_spend_cr", "star_spend_share", "value_per_crore", "all_rounder_share"]])
    k = min(3, len(team))
    team["cluster"] = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(features)
    labels = team.groupby("cluster").star_spend_share.transform(lambda x: np.where(x >= x.median(), "Star-led", "Balanced"))
    team["strategy"] = np.where(team.star_spend_share > .29, "Star-led squad", np.where(team.value_per_crore > team.value_per_crore.median(), "Value-led squad", "Balanced squad"))
    return team.sort_values("total_spend_cr", ascending=False)


def save_to_sqlite(scored: pd.DataFrame, teams: pd.DataFrame, path="criconomics.db"):
    with sqlite3.connect(path) as con:
        scored.to_sql("player_valuations", con, if_exists="replace", index=False)
        teams.to_sql("team_auction_strategy", con, if_exists="replace", index=False)
