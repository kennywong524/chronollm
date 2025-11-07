"""
build_ground_truth.py
---------------------
Build ground truth for the innovation/events anachronism detection task.

Ground Truth Policy:
- For each event/activity, 'first_possible_year' is the year of the first recorded use by a sitting U.S. president, NOT the invention's public release date.
- For some events, a 'last_possible_year' is also specified, after which the event is no longer possible or relevant.
- Grey zones are eliminated: for certain events, presidents whose terms overlap ambiguous periods are excluded for that event.

Usage:
    python build_ground_truth.py [--output <output_path>]
"""

import argparse
import pandas as pd
from pathlib import Path

# List of events/activities and the year they first became possible
EVENTS = [
    {"event": "Rode in an automobile while president", "first_possible_year": 1901},
    {"event": "Flew in an airplane while president", "first_possible_year": 1943},
    {"event": "Travelled by railroad while president", "first_possible_year": 1833},
    {"event": "Used the White House telephone", "first_possible_year": 1877},
    {"event": "Gave a televised address as president", "first_possible_year": 1947},
    {"event": "Sent an e-mail while president", "first_possible_year": 1993},
    {"event": "Used generative AI while president", "first_possible_year": 2022},
    {"event": "Appeared in a photograph while president", "first_possible_year": 1845},
    {"event": "Listened to the radio while president", "first_possible_year": 1933},
    {"event": "Used a light bulb while president", "first_possible_year": 1891},
]

# Grey zone definitions: {event: (start_year, end_year)}
GREY_ZONES = {
    "Flew in an airplane while president": (1903, 1943),
    "Travelled by railroad while president": (1815, 1833),
    "Used the White House telephone": None,  # special handling
    "Gave a televised address as president": (1925, 1947),
    "Sent an e-mail while president": (1970, 1993),
    "Appeared in a photograph while president": (1836, 1845),
    "Listened to the radio while president": (1896, 1933),
    "Used a light bulb while president": (1879, 1891),
    "Rode in an automobile while president": (1885, 1901),
}

def get_president_before_and_after(df, year):
    before = df[df["end_year"] < year].sort_values("end_year").tail(1)
    after = df[df["start_year"] > year].sort_values("start_year").head(1)
    return set(before["name"]).union(set(after["name"]))

def main():
    parser = argparse.ArgumentParser(description="Build ground truth for innovation/events anachronism detection")
    parser.add_argument("--input", type=str, default="data/us_presidents_with_states.csv", help="Path to presidents CSV")
    parser.add_argument("--output", type=str, default="data/president_event_ground_truth.csv", help="Output CSV path")
    
    args = parser.parse_args()
    
    # Load president data
    df = pd.read_csv(args.input)
    if not all(col in df.columns for col in ["name", "start_year", "end_year"]):
        raise ValueError("CSV must contain columns: name, start_year, end_year")
    
    rows = []
    for _, pres in df.iterrows():
        for event in EVENTS:
            first_year = event["first_possible_year"]
            last_year = event.get("last_possible_year", float("inf"))
            event_name = event["event"]
            
            # Grey zone elimination
            if event_name in GREY_ZONES and GREY_ZONES[event_name]:
                grey_start, grey_end = GREY_ZONES[event_name]
                # If president's term overlaps grey zone, skip for this event
                if pres["end_year"] >= grey_start and pres["start_year"] <= grey_end:
                    continue
                    
            # Special handling for White House telephone
            if event_name == "Used the White House telephone":
                to_remove = get_president_before_and_after(df, 1877)
                if pres["name"] in to_remove:
                    continue
                    
            # Possible if president's term overlaps with [first_year, last_year]
            possible = (pres["end_year"] >= first_year) and (pres["start_year"] <= last_year)
            rows.append({
                "president": pres["name"],
                "start_year": pres["start_year"],
                "end_year": pres["end_year"],
                "event": event_name,
                "first_possible_year": first_year,
                "last_possible_year": last_year if "last_possible_year" in event else "",
                "possible": possible
            })
    
    ground_truth_df = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_df.to_csv(output_path, index=False)
    print(f"Ground truth table saved to {output_path}")
    print(f"Total pairs: {len(ground_truth_df)}")
    print(f"Possible pairs: {ground_truth_df['possible'].sum()}")
    print(f"Impossible pairs: {(~ground_truth_df['possible']).sum()}")

if __name__ == "__main__":
    main()

