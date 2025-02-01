from pathlib import Path

import mkdocs_gen_files


def format_link(file: Path) -> str:
    """
    Formats the markdown link for a file.

    Args:
        file (Path): The file for which to format the markdown link.

    Returns:
        The formatted markdown link as a string.
    """
    link = f"{file.stem}/index.md" if file.is_dir() else f"{file.stem}.md"
    return f"- [{file.stem}]({link})\n"


def add_submodules_as_list(parent_folder: Path) -> str:
    """
    Generates a markdown list of submodules with docstrings.

    Args:
        parent_folder (Path): The parent directory to search for submodules.

    Returns:
        A markdown formatted string listing all submodules.
    """
    output = []
    for file in parent_folder.iterdir():
        # Directories containing __init__.py are considered submodules
        if file.is_dir() and (file / "__init__.py").exists():
            output.append(format_link(file))
        # Python files apart from __init__.py are considered submodules
        elif file.suffix == ".py" and file.name != "__init__.py":
            output.append(format_link(file))
    return "".join(output)


def generate_api_reference(src: Path, exclude: list) -> None:
    """
    Generates the API reference documentation for a given package.

    This function traverses through the source directory, identifies Python modules, and generates markdown
    documentation files for each module. It also creates a navigation structure for these documents.

    Args:
        src (Path): The source directory of the package for which to generate API documentation.
        exclude (list): A list of module to exclude from the API documentation. Use Python module notation (e.g. `lighter.utils.logging`).
    """
    package = src.name
    # Initialize navigation for mkdocs
    nav = mkdocs_gen_files.Nav()

    # Iterate through all Python files in the source directory
    for path in sorted(src.rglob("*.py")):
        module_path = path.relative_to(src)
        # Convert the module path to Python module notation
        module_py_notation = f"{package}.{'.'.join(module_path.with_suffix('').parts)}"
        if module_py_notation in exclude:
            print(f"Excluding '{module_py_notation}' from the API reference.")
            continue
        # Define the path for the documentation file
        doc_path = module_path.with_suffix(".md")
        # Split the module notation into parts for navigation
        parts = [package] + module_py_notation.split(".")[1:]
        # Initialize an empty string for submodules list
        submodules_list = ""
        # Check if the current module is an __init__ module
        if parts[-1] == "__init__":
            # Remove the last part as it's an __init__ module
            parts.pop()
            # Change the documentation path to index.md for __init__ modules
            doc_path = doc_path.with_name("index.md")
            # Generate a list of submodules for __init__ modules
            submodules_list = add_submodules_as_list(path.parent)
        # Skip __main__ modules
        elif parts[-1] == "__main__":
            continue
        # Add the module to the navigation structure
        nav[tuple(parts)] = doc_path.as_posix()
        # Open the documentation file for writing
        with mkdocs_gen_files.open("reference" / doc_path, "w") as fd:
            # Write the module documentation and submodules list to the file
            fd.write(f"::: {'.'.join(parts)}\n\n{submodules_list}")
        # Set the edit path for the documentation file
        mkdocs_gen_files.set_edit_path("reference" / doc_path, path.relative_to(src.parent))

    # Open the SUMMARY.md file for writing the navigation structure
    with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
        # Write the navigation structure to the SUMMARY.md file
        nav_file.writelines(nav.build_literate_nav())


# Generate the API reference.
# Control whether to include private members under `mkdocstrings.handlers.python.options.filters` in `mkdocs.yml`.
# To exclude specific modules, add them to the `exclude_modules` list.
src_path = Path(__file__).parent.parent / "lighter"
exclude_modules = []
generate_api_reference(src_path, exclude_modules)
