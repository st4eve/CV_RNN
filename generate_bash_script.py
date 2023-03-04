"""CV QLSTM 2023"""
import argparse
import json
import sys

from sklearn.model_selection import ParameterGrid


def generate_bash_script(num_parallel_jobs, json_config_path, script_name):
    """Generates bash script to run jobs in parallel

    Args:
        num_parallel_jobs (int): _description_
        json_config_path (dict): Dictionary with parameters and target experiment file

    """
    with open(json_config_path, "r") as file:
        config = json.load(file)

    full_script_name = f"{script_name}.sh"

    with open(full_script_name, "w", encoding="utf-8") as write_file:
        write_file.write("#!/bin/sh\n")
        for val, parameter_combo in enumerate(list(ParameterGrid(config["parameters"]))):
            param_str = " ".join([f"{key}={val}" for key, val in parameter_combo.items()])
            write_str = f"python {config['experiment_file']} with {param_str} &\n"
            write_file.write(write_str)
            if (val + 1) % num_parallel_jobs == 0:
                write_file.write("wait\n")


def main():
    """Parse args for generating bash script

    Args:
        sys_argv (list): List of command line args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_parallel", default=8, type=int, required=False, help="Number of parallel jobs")
    parser.add_argument("--json_path", type=str, required=True, help="Path to json file with configuration")
    parser.add_argument("--script_name", type=str, default="run_experiment", help="Name of bash script")
    args = parser.parse_args()
    generate_bash_script(args.num_parallel, args.json_path, args.script_name)


if __name__ == "__main__":
    sys.exit(main())
