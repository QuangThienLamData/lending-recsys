import os
import random
import datetime
import pandas as pd
import hashlib

KNOWN_NAMES = {
    "100001137": "Vo Quang Thien",
}
_FIRST = [
    "Nguyen Van", "Tran Thi", "Le Minh", "Pham Duc", "Hoang Anh",
    "Do Thi", "Bui Van", "Ngo Thi", "Vo Minh", "Dang Quoc",
    "John", "Jane", "Michael", "Sarah", "David",
    "Emma", "James", "Olivia", "Robert", "Linda",
]
_LAST = [
    "An", "Binh", "Cuong", "Dung", "Hai", "Hoa", "Hung", "Lan",
    "Long", "Mai", "Nam", "Phuong", "Quang", "Son", "Thanh", "Thu",
    "Smith", "Johnson", "Williams", "Brown", "Jones",
]

def _get_name(user_id: str) -> str:
    if user_id in KNOWN_NAMES:
        return KNOWN_NAMES[user_id]
    h = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return f"{_FIRST[h % len(_FIRST)]} {_LAST[(h >> 8) % len(_LAST)]}"

PURPOSES = [
    "debt_consolidation", "credit_card", "home_improvement", "other",
    "major_purchase", "small_business", "car", "medical",
    "moving", "vacation", "house", "wedding",
    "renewable_energy", "educational",
]

TERMS = ["36 months", "60 months"]

def generate_data(n=200):
    records = []
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(days=30)
    
    for _ in range(n):
        uid = str(random.randint(100000000, 100002000))
        name = _get_name(uid)
        amount = float(random.randint(5, 40) * 1000)
        int_rate = round(random.uniform(5.0, 25.0), 1)
        purpose = random.choice(PURPOSES)
        term = random.choice(TERMS)
        score = random.uniform(0.3, 0.95)
        
        # random timestamp within the last 30 days
        random_seconds = random.randint(0, int((end_time - start_time).total_seconds()))
        ts = start_time + datetime.timedelta(seconds=random_seconds)
        
        records.append({
            "ID": uid,
            "Full Name": name,
            "Loan Amount": amount,
            "Expected Interest Rate": int_rate,
            "Purpose": purpose,
            "Term": term,
            "Estimated Repay Score": score,
            "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S")
        })
        
    df = pd.DataFrame(records)
    # Sort by timestamp descending
    df = df.sort_values(by="Timestamp", ascending=False)
    
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "simulation_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {n} simulation records at {out_path}")

if __name__ == "__main__":
    generate_data()
