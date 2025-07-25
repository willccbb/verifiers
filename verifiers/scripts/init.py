import argparse
from pathlib import Path

import verifiers as vf

README_TEMPLATE = """\
# {env_id}
"""

PYPROJECT_TEMPLATE = f"""\
[project]
name = "{{env_id}}"
version = "0.1.0"
dependencies = [
    "verifiers>={vf.__version__}",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
include = ["{{env_file}}.py"]
"""

ENVIRONMENT_TEMPLATE = """\
import verifiers as vf


def load_environment(**kwargs) -> vf.Environment:
    \"\"\"
    Loads a custom environment.
    \"\"\"
    raise NotImplementedError("Implement your custom environment here.")
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, help="The environment id to init")
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default="./environments",
        help="Path to environments directory (default: ./environments)",
    )
    args = parser.parse_args()

    # make environment parent directory if it doesn't exist
    local_dir = Path(args.path) / args.env.replace("-", "_")
    local_dir.mkdir(parents=True, exist_ok=True)

    # create README.md if it doesn't exist
    readme_file = local_dir / "README.md"
    if not readme_file.exists():
        readme_file.write_text(
            README_TEMPLATE.format(env_id=args.env.replace("_", "-"))
        )
    else:
        print(f"README.md already exists at {readme_file}, skipping...")

    # create pyproject.toml if it doesn't exist
    pyproject_file = local_dir / "pyproject.toml"
    if not pyproject_file.exists():
        pyproject_file.write_text(
            PYPROJECT_TEMPLATE.format(
                env_id=args.env.replace("_", "-"), env_file=args.env.replace("-", "_")
            )
        )
    else:
        print(f"pyproject.toml already exists at {pyproject_file}, skipping...")

    # create environment file if it doesn't exist
    environment_file = local_dir / f"{args.env.replace('-', '_')}.py"
    if not environment_file.exists():
        environment_file.write_text(
            ENVIRONMENT_TEMPLATE.format(env_id=args.env.replace("_", "-"))
        )
    else:
        print(
            f"{args.env.replace('-', '_')}.py already exists at {environment_file}, skipping..."
        )


if __name__ == "__main__":
    main()
