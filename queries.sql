CREATE VIEW auction_value_board AS
SELECT player, team, role, performance_index, auction_price_cr, fair_price_cr,
       price_gap_cr, value_per_crore, verdict
FROM player_valuations
ORDER BY price_gap_cr;

CREATE VIEW team_strategy_board AS
SELECT team, strategy, total_spend_cr, star_spend_share, value_per_crore, avg_performance
FROM team_auction_strategy
ORDER BY total_spend_cr DESC;
