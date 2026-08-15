"""
Small eval dataset for the reasoner: multi-step word problems with exact answers.

All answers are exact numbers (int or decimal) so the tier-1 verifier can
check them. Mix of 2-step and multi-step problems, plus a couple of classic
"gotcha" traps that trip naive models (ordering, units, remainders).

Format: (problem, ground_truth_answer)
"""

PROBLEMS = [
    # --- 2-step ---
    ("A farmer has 3 hens. Each hen lays 4 eggs per day. "
     "How many eggs does he collect in a week?", "84"),

    ("A train travels 60 km in 2 hours, then 90 km in 3 hours. "
     "What is its average speed in km/h over the whole trip?", "30"),

    ("A shop sells apples at 3 for $2. "
     "How much do 15 apples cost in dollars?", "10"),

    # --- multi-step with ordering traps ---
    ("Alice is twice as old as Bob. In 5 years, their combined age "
     "will be 55. How old is Bob now?", "15"),

    ("A bat and a ball cost $1.10 together. The bat costs $1.00 more "
     "than the ball. How much does the ball cost in cents?", "5"),

    ("A snail climbs 3 meters up a wall during the day and slides "
     "down 2 meters at night. The wall is 10 meters high. "
     "On which day does it first reach the top?", "8"),

    ("If 5 machines take 5 minutes to make 5 widgets, how many minutes "
     "would it take 100 machines to make 100 widgets?", "5"),

    # --- remainders / units ---
    ("A baker packs cookies into boxes of 12. She has 250 cookies. "
     "How many cookies are left over after packing as many full boxes "
     "as possible?", "10"),

    ("A car uses 8 liters of fuel per 100 km. Fuel costs $1.5 per liter. "
     "How much does fuel cost for a 350 km trip in dollars?", "42"),

    ("A rectangular garden is 12 m long and 8 m wide. A path 1 m wide "
     "surrounds it. What is the area of the path in square meters?", "44"),

    # --- decimals ---
    ("A restaurant bill is $80. A 15% tip is added, then the total is "
     "split equally among 4 people. How much does each person pay in "
     "dollars?", "23"),

    ("A water tank fills at 5 liters per minute and drains at 2 liters "
     "per minute. It starts empty. How many minutes to fill a 90-liter "
     "tank?", "30"),

    # --- combinatorics-ish (multi-step) ---
    ("A school bus has 40 seats. On the first stop, 12 students board. "
     "On the second stop, 15 board and 4 get off. On the third stop, 9 "
     "board and 6 get off. How many seats are empty at the end?", "14"),

    ("A store has a sale: buy 2 shirts, get the third at half price. "
     "Each shirt costs $20. What is the total for 3 shirts in dollars?", "50"),
]

if __name__ == "__main__":
    print(f"{len(PROBLEMS)} problems loaded")
