import argparse
import random
import json
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Generate piecewise linear functions and sample points, then store the data in a JSON file."
)
parser.add_argument("--num_functions", type=int, default=10,
                    help="Number of functions to generate (default: 10)")
parser.add_argument("--num_points", type=int, default=100,
                    help="Number of x points to sample per function (default: 100)")
parser.add_argument("--seed", type=int, default=282,
                    help="Random seed for reproducibility (default: 282)")
parser.add_argument("--lower_bound", type=float, default=-10.0,
                    help="Lower bound for sampling x (default: -10.0)")
parser.add_argument("--upper_bound", type=float, default=10.0,
                    help="Upper bound for sampling x (default: 10.0)")
parser.add_argument("--output_file", type=str, default="data.json",
                    help="Output JSON file name (default: data.json)")
args = parser.parse_args()
random.seed(args.seed)
if args.lower_bound >= args.upper_bound:
    raise ValueError("Error: lower_bound must be less than upper_bound.")
with open(args.output_file, "w") as f:
    f.write("[\n")
    for i in tqdm(range(args.num_functions), desc="Generating functions"):
        a = random.gauss(0, 1)
        b = random.gauss(0, 1)
        c = random.gauss(0, 1)
        d = random.gauss(0, 1)
        e = random.gauss(0, 1)
        x_values = [random.uniform(args.lower_bound, args.upper_bound) for _ in range(args.num_points)]
        fx_values = [a * x + b if x < c else d * x + e for x in x_values]
        func_dict = {
            "parameters": {"a": a, "b": b, "c": c, "d": d, "e": e},
            "x": x_values,
            "f_x": fx_values
        }
        json_str = json.dumps(func_dict, indent=2)
        if i < args.num_functions - 1:
            f.write(json_str + ",\n")
        else:
            f.write(json_str + "\n")
    f.write("]\n")
print(f"Data written to {args.output_file}")
