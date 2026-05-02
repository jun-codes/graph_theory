import argparse
import os
import pickle

from cp_io import write_cp_collection, write_cp_file


BASE = r"C:\Users\Arjun\Desktop\code\Graph_Theory_Project"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export generated crease-pattern pickles to editable .cp files."
    )
    parser.add_argument(
        "--input",
        default=os.path.join(BASE, "best_generated.pkl"),
        help="Path to a pickled NetworkX graph or a list of graphs.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .cp path for a single graph, or output folder for a list.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="If the pickle contains a list, export just this 0-based index.",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=None,
        help="Optionally clip coordinates to [-clip, clip] before writing.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=6,
        help="Number of decimals to keep in the .cp file.",
    )
    return parser.parse_args()


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def default_single_output(input_path, suffix=""):
    stem, _ = os.path.splitext(input_path)
    return f"{stem}{suffix}.cp"


def default_multi_output(input_path):
    stem, _ = os.path.splitext(input_path)
    return f"{stem}_cp"


def main():
    args = parse_args()
    obj = load_pickle(args.input)

    if isinstance(obj, list):
        if args.index is not None:
            if args.index < 0 or args.index >= len(obj):
                raise IndexError(f"Index {args.index} is out of range for {len(obj)} graphs.")
            output_path = args.output or default_single_output(args.input, f"_{args.index + 1}")
            write_cp_file(
                obj[args.index],
                output_path,
                decimals=args.decimals,
                clip_box=args.clip,
            )
            print(f"Saved {output_path}")
            return

        output_dir = args.output or default_multi_output(args.input)
        paths = write_cp_collection(
            obj,
            output_dir,
            prefix="rank",
            decimals=args.decimals,
            clip_box=args.clip,
        )
        print(f"Saved {len(paths)} files to {output_dir}")
        return

    output_path = args.output or default_single_output(args.input)
    write_cp_file(obj, output_path, decimals=args.decimals, clip_box=args.clip)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
