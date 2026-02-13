"""Debug script to inspect LLM reliability in daily_ledger.db."""
import sqlite3
from engine.ledger import DB_PATH

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print("--- Agent Action Distribution ---")
rows = cursor.execute("""
    SELECT agent, action, COUNT(*) as count 
    FROM decisions 
    GROUP BY agent, action
""").fetchall()
for r in rows:
    print(dict(r))

print("\n--- Sample LLM Reasons (First 5 'HOLD's) ---")
rows = cursor.execute("""
    SELECT date, window, action, reasoning 
    FROM decisions 
    WHERE agent='llm' 
    ORDER BY id DESC
    LIMIT 5
""").fetchall()
for r in rows:
    print(f"Action: {r['action']} | Reasoning: {r['reasoning']}")

conn.close()
