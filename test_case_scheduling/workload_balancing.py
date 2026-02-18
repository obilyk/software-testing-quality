import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pulp import LpBinary, LpMinimize, LpProblem, LpVariable, lpSum


def solve_scheduling_problem(
    testers,
    fixed_meetings_raw,
    test_cases,
    compatibility,
    block_size=15,
    day_start=10 * 60,
    day_end=19 * 60,
):
    # Setup
    num_blocks = (day_end - day_start) // block_size
    time_blocks = list(range(num_blocks))

    # Convert durations to blocks
    durations = {name: math.ceil(duration / block_size) for name, duration in test_cases}
    test_names = [name for name, _ in test_cases]

    # Convert meeting times to block indices
    fixed_meetings = {
        k: [(m_start // block_size, math.ceil(m_end / block_size)) for (m_start, m_end) in meetings]
        for k, meetings in fixed_meetings_raw.items()
    }

    # Define LP model
    model = LpProblem("Test_Scheduling_with_Meetings", LpMinimize)

    # Decision variables
    x = {
        (i, k, t): LpVariable(f"x_{i}_{k}_{t}", cat=LpBinary)
        for i in test_names
        for k in testers
        if k in compatibility[i]
        for t in time_blocks
        if t + durations[i] <= num_blocks
        and all(
            not (t + d in range(m_start, m_end))
            for d in range(durations[i])
            for (m_start, m_end) in fixed_meetings.get(k, [])
        )
    }

    # Makespan variable (in minutes)
    makespan = LpVariable("makespan", lowBound=0)

    # Objective
    model += makespan

    # Constraint: assign each test once
    for i in test_names:
        model += lpSum(x[i, k, t] for k in testers if k in compatibility[i] for t in time_blocks if (i, k, t) in x) == 1

    # Constraint: non-overlapping per tester
    for k in testers:
        for τ in time_blocks:
            model += lpSum(
                x[i, k, t]
                for i in test_names
                for t in time_blocks
                if t <= τ < t + durations[i] and (i, k, t) in x
            ) <= 1

    # Constraint: define makespan
    for (i, k, t), var in x.items():
        end_time = (t + durations[i]) * block_size
        model += makespan >= end_time * var

    epsilon = 1e-4  # tiny penalty to bias earlier slots
    model += makespan + epsilon * lpSum(t * x[i, k, t] for (i, k, t) in x)

    # Solve
    model.solve()

    # Extract schedule
    schedule = []
    for (i, k, t), var in x.items():
        if var.value() == 1:
            start_min = t * block_size
            end_min = (t + durations[i]) * block_size
            schedule.append(
                {
                    "Test Case": i,
                    "Tester": k,
                    "Start Time": start_min,
                    "End Time": end_min,
                    "Duration": end_min - start_min,
                }
            )

    schedule_df = pd.DataFrame(schedule).sort_values(by=["Tester", "Start Time"])

    return schedule_df


def plot_test_schedule_grouped(
    schedule_df, fixed_meetings_raw, day_start_min=10 * 60, day_end_min=19 * 60, time_bin=15
):
    """
    Plot a clean Gantt chart grouped by tester with 15-minute resolution and test name labels.

    Parameters:
    - schedule_df: DataFrame with 'Test Case', 'Tester', 'Start Time', 'End Time'
    - day_start_min: Start of workday in minutes (default = 600 = 10:00)
    - day_end_min: End of workday in minutes (default = 1080 = 18:00)
    - time_bin: Time bin size in minutes (default = 15)
    """

    plt.figure(figsize=(12, 6))

    # X-axis: 15-minute bins
    x_ticks = list(range(day_start_min, day_end_min + 1, time_bin))
    x_labels = [f"{t // 60:02d}:{t % 60:02d}" for t in x_ticks]

    # Colors
    base_color = sns.color_palette("Blues_d", len(schedule_df))

    # Plot fixed meetings first (gray bars)
    for tester, meetings in fixed_meetings_raw.items():
        # y = y_pos.get(tester)
        # if y is not None:
        for m_start, m_end in meetings:
            bar_left = (m_start + day_start_min) / 60
            bar_width = (m_end - m_start) / 60
            plt.barh(
                f"Tester {tester}",
                width=bar_width,
                left=bar_left,
                color="gray",
                alpha=0.5,
                edgecolor="white",
            )
            # Add label
            plt.text(
                bar_left + bar_width / 2,
                f"Tester {tester}",
                "Fixed Meeting" if bar_left != 14.0 else "Lunch",
                ha="center",
                va="center",
                fontsize=10,
                # fontweight="bold",
                rotation=90,
                rotation_mode="anchor",
            )

    # Plot tasks
    for idx, row in schedule_df.iterrows():
        tester = row["Tester"]
        # y = y_pos[tester]
        start_abs = row["Start Time"] + day_start_min
        end_abs = row["End Time"] + day_start_min

        plt.barh(
            f"Tester {tester}",
            width=(end_abs - start_abs) / 60,
            left=start_abs / 60,
            color=base_color[idx % len(base_color)],
            edgecolor="white",
        )

        # Add label inside bar
        plt.text(
            (start_abs + end_abs) / 120,
            f"Tester {tester}",
            row["Test Case"],
            ha="center",
            va="center",
            fontsize=10,
            # fontweight="bold",
            rotation=90,
            rotation_mode="anchor",
        )

    # Axes & labels
    plt.xticks([t / 60 for t in x_ticks], x_labels, rotation=45)
    # plt.xticks(x_ticks, x_labels, rotation=45)
    plt.xlabel("Time (24-hour format)")
    plt.title("Test Schedule per Tester (10:00–19:00)", fontsize=13, weight="bold")
    plt.xlim(day_start_min / 60, day_end_min / 60)
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    # plt.tight_layout()
    # plt.savefig("test_schedule_ex_1.png", dpi=300, bbox_inches="tight")
    plt.show()


def show_test_cases_table(test_cases):
    test_table = pd.DataFrame(
        [
            {
                "Test Case": name,
                "Duration (min)": duration,
                "Compatible Testers": ", ".join(f"Tester {t}" for t in compatibility[name]),
            }
            for name, duration in test_cases
        ]
    )
    print("\n# Test Cases Table #")
    print(test_table.to_markdown(index=False))
    print()


def show_fixed_meetings_table(fixed_meetings_raw):
    def minutes_to_time_str(m):
        h = 10 + m // 60
        min = m % 60
        return f"{h:02d}:{min:02d}"

    # Create table of fixed meetings
    meeting_table = pd.DataFrame(
        [
            {
                "Tester": f"Tester {tester}",
                "Event Start": minutes_to_time_str(start),
                "Event End": minutes_to_time_str(end),
                "Duration (min)": end - start,
            }
            for tester, meetings in fixed_meetings_raw.items()
            for start, end in meetings
        ]
    )
    print("\n# Fixed Meetings Table #")
    print(meeting_table.to_markdown(index=False))
    print()


if __name__ == "__main__":
    # exp 1
    # Test cases (Name, Duration in minutes)
    test_cases = [
        ("Test A", 135),
        ("Test B", 60),
        ("Test C", 90),
        ("Test D", 120),
        ("Test E", 90),
        ("Test F", 75),
        ("Test G", 75),
        ("Test H", 75),
        ("Test I", 60),
        ("Test J", 45)
    ]
    # Testers
    testers = [1, 2]
    # Compatibility matrix (mostly shared, a few restricted)
    compatibility = {
        "Test A": [1, 2],
        "Test B": [1, 2],
        "Test C": [1],         # Tester 1 only
        "Test D": [1, 2],
        "Test E": [2],         # Tester 2 only
        "Test F": [1, 2],
        "Test G": [1, 2],
        "Test H": [2],         # Tester 2 only
        "Test I": [1, 2],
        "Test J": [1, 2]
    }
    fixed_meetings_raw = {
        1: [(240, 300)],                  # Tester 1: Lunch (14:00–15:00)
        2: [(90, 120), (240, 300), (300, 330)]  # Tester 2: 11:30–12:00, Lunch, 15:00–15:30
    }

    # exp 2
    # # Existing test cases
    # test_cases = [
    #     ("Test A", 60),
    #     ("Test B", 45),
    #     ("Test C", 90),
    #     ("Test D", 60),
    #     ("Test E", 75),
    #     ("Test F", 30),
    #     ("Test G", 60),
    #     ("Test H", 45),
    #     ("Test I", 90),
    #     ("Test J", 60),
    #     ("Test K", 45),
    #     ("Test L", 30),
    #     ("Test M", 60),
    #     ("Test N", 45),
    #     ("Test O", 90),
    #     ("Test P", 75),
    #     ("Test Q", 30),
    #     ("Test R", 60),
    #     # Additional cases
    #     ("Test S", 45),
    #     ("Test T", 30),
    #     ("Test U", 60),
    #     ("Test V", 45),
    # ]
    # # Compatibility update with some variety
    # compatibility = {
    #     "Test A": [1, 2],
    #     "Test B": [2, 3],
    #     "Test C": [1],
    #     "Test D": [1, 3],
    #     "Test E": [2],
    #     "Test F": [1, 2, 3],
    #     "Test G": [3],
    #     "Test H": [1, 2],
    #     "Test I": [2, 3],
    #     "Test J": [1, 3],
    #     "Test K": [1],
    #     "Test L": [2, 3],
    #     "Test M": [1, 2, 3],
    #     "Test N": [1],
    #     "Test O": [2],
    #     "Test P": [3],
    #     "Test Q": [1, 2],
    #     "Test R": [2, 3],
    #     "Test S": [2, 3],
    #     "Test T": [1],
    #     "Test U": [1, 3],
    #     "Test V": [1, 2],
    # }
    # # Testers: use IDs 1, 2, 3
    # testers = [1, 2, 3]
    # # Fixed meetings (in minutes from 10:00)
    # # Format: tester_id -> [(start_min, end_min), ...]
    # fixed_meetings_raw = {
    #     1: [(30, 60), (240, 300), (300, 330), (360, 375)],  # 10:30–11:00, 14:00–15:00, 15:00–15:30, 16:00–16:15
    #     2: [(90, 120), (240, 300), (360, 390), (405, 420)],  # 11:30–12:00, 14:00–15:00, 16:00–16:30, 16:45–17:00
    #     3: [(180, 195), (240, 300), (330, 345)],  # 13:00–13:15, 14:00–15:00, 15:30–15:45
    # }

    schedule_df = solve_scheduling_problem(testers, fixed_meetings_raw, test_cases, compatibility)
    show_test_cases_table(test_cases)
    show_fixed_meetings_table(fixed_meetings_raw)
    plot_test_schedule_grouped(schedule_df, fixed_meetings_raw)
