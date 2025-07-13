import pandas as pd
from datetime import datetime, timedelta

CSV_FILE = "sheet.csv"

# Sample time slots for a day
def generate_time_slots():
    return [datetime.strptime(f"{h}:00", "%H:%M").time() for h in [10, 11, 12, 14, 15, 16]]

# Check slot availability
def is_slot_available(date, time, judge_id, duration, df):
    proposed_start = datetime.combine(date, time)
    proposed_end = proposed_start + timedelta(minutes=duration)

    for _, row in df.iterrows():
        if row['judge_id'] == judge_id and pd.notnull(row['scheduled_datetime']):
            existing_start = pd.to_datetime(row['scheduled_datetime'], dayfirst=True)
            existing_end = existing_start + timedelta(minutes=row['estimated_duration'])
            if not (proposed_end <= existing_start or proposed_start >= existing_end):
                return False
    return True

# Reschedule request - mark as pending
def reschedule_case(case_no, direction, df):
    if case_no not in df['case_no'].values:
        print(f"Case {case_no} not found.")
        return df

    row = df[df['case_no'] == case_no].iloc[0]

    if pd.notnull(row.get('reschedule_request')):
        print(f"Case {case_no} already has a pending reschedule request.")
        return df

    current_dt = pd.to_datetime(row['scheduled_datetime'])
    judge_id = row['judge_id']
    duration = int(row['estimated_duration'])

    days_range = range(1, 15) if direction == "postpone" else range(1, 15)[::-1]

    for delta in days_range:
        new_date = current_dt.date() + timedelta(days=delta if direction == "postpone" else -delta)
        if new_date < datetime.today().date():
            continue

        for time in generate_time_slots():
            if is_slot_available(new_date, time, judge_id, duration, df):
                new_dt = datetime.combine(new_date, time)
                df.loc[df['case_no'] == case_no, 'reschedule_request'] = new_dt
                df.loc[df['case_no'] == case_no, 'status'] = 'Pending'
                df.loc[df['case_no'] == case_no, 'request_date'] = datetime.now()
                df.loc[df['case_no'] == case_no, 'admin_comment'] = ""
                print(f"Requested reschedule for case {case_no} to {new_dt} [Pending Judge Approval]")
                return df

    print("No suitable slot found.")
    return df

# Admin approval function
def admin_review(case_no, decision, df):
    if decision.lower() == 'accept':
        new_dt = df.loc[df['case_no'] == case_no, 'reschedule_request']
        df.loc[df['case_no'] == case_no, 'scheduled_datetime'] = new_dt
        df.loc[df['case_no'] == case_no, 'status'] = 'Scheduled'
        df.loc[df['case_no'] == case_no, 'admin_comment'] = 'Approved by judge'
        print(f"Reschedule for case {case_no} approved.")
    else:
        df.loc[df['case_no'] == case_no, 'reschedule_request'] = pd.NaT
        df.loc[df['case_no'] == case_no, 'status'] = 'Rejected'
        df.loc[df['case_no'] == case_no, 'admin_comment'] = 'Rejected by judge'
        print(f"Reschedule for case {case_no} rejected.")

    df.to_csv(CSV_FILE, index=False)
    return df

# Display and process pending requests
def process_pending_requests(df):
    pending_cases = df[df['status'] == 'Pending']

    if pending_cases.empty:
        print("No reschedule requests pending.")
        return df

    pending_cases = pending_cases.sort_values(by='filing_date')

    for _, row in pending_cases.iterrows():
        case_no = row['case_no']
        print(f"\nCase #{case_no} - Requested new datetime: {row['reschedule_request']}")
        decision = input("Judge decision (accept/reject): ").strip().lower()
        while decision not in ['accept', 'reject']:
            decision = input("Invalid input. Please enter 'accept' or 'reject': ").strip().lower()
        df = admin_review(case_no, decision, df)

    return df

# Main function
def main():
    df = pd.read_csv(CSV_FILE, parse_dates=['filing_date', 'scheduled_datetime'], dayfirst=True)

    df = reschedule_case(case_no=2, direction="postpone", df=df)
    df = reschedule_case(case_no=3, direction="postpone", df=df)

    df = process_pending_requests(df)

if __name__ == "__main__":
    main()
