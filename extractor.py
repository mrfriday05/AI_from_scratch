import os
import argparse
import fnmatch

def parse_rules(file_path):
    """
    Parses the rules file (e.g., folders.txt).
    Lines starting with '+' are additions, '-' are subtractions.
    Wildcards like '*' and '?' are supported.
    Returns a list of (type, pattern) tuples, preserving order.
    """
    rules = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if not stripped_line or stripped_line.startswith('#'):
                    continue

                rule_type = stripped_line[0]
                pattern = stripped_line[1:].strip()
                
                # Normalize path separators for cross-platform consistency in patterns
                pattern = os.path.normpath(pattern)

                if rule_type in ['+', '-'] and pattern:
                    rules.append((rule_type, pattern))
                else:
                    print(f"[Warning] Skipping invalid line in '{file_path}': {stripped_line}")
    except FileNotFoundError:
        print(f"[Error] The specified rule file was not found: '{file_path}'")
        return None
        
    return rules

def should_process_path(path, rules, is_dir=False):
    """
    Determines if a given path should be processed based on the ordered rules.
    The last rule in the list that matches the path determines the outcome.
    
    Args:
        path (str): The file or directory path to check.
        rules (list): A list of (type, pattern) tuples.
        is_dir (bool): Flag indicating if the path is a directory. 
                       Directories are traversed by default unless explicitly excluded.
                       Files are not included by default unless explicitly included.
    """
    path = os.path.normpath(path)
    decision = None  # None: no matching rule yet, True: include, False: exclude.

    # Collect all parts of the path to check against patterns.
    # e.g., for "src/app/main.py", we check "main.py", "app", and "src".
    # This allows a rule like "-__pycache__" to match a directory anywhere in the path.
    parts_to_check = [os.path.basename(path)]
    # Deconstruct the path for parent directory checks
    head = path
    while True:
        head, tail = os.path.split(head)
        if tail:
            parts_to_check.append(tail)
        else:
            if head: # Handles root path component on Unix-like systems
                parts_to_check.append(head)
            break
    parts_to_check.append(path) # Also check against the full path

    for rule_type, pattern in rules:
        # Check if the pattern matches any part of the path
        if any(fnmatch.fnmatch(part, pattern) for part in parts_to_check):
            decision = (rule_type == '+')
    
    # If no rule matched, apply default behavior
    if decision is None:
        return is_dir
    
    return decision

def walk_and_collect_structure(rules):
    """
    Walks the file system from the current directory based on rules and 
    collects the structure and file contents.
    """
    structure = []
    file_contents = []
    start_dir = '.'  # Start from the current directory

    for root, dirs, files in os.walk(start_dir, topdown=True):
        # Prune directories: A directory is entered unless a rule excludes it.
        dirs[:] = [d for d in sorted(dirs) if should_process_path(os.path.join(root, d), rules, is_dir=True)]
        
        # Skip the root '.' directory from the visual output
        if root == '.':
            # Process files in the root directory
            for file in sorted(files):
                filepath = os.path.join(root, file)
                if should_process_path(filepath, rules, is_dir=False):
                    structure.append(file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except Exception as e:
                        content = f"[Error reading file: {e}]"
                    file_contents.append((filepath, content))
            continue # Move to the next iteration

        # Add directory structure
        level = root.count(os.sep)
        indent = '  ' * (level -1)
        structure.append(f"{indent}{os.path.basename(root)}/")
        
        sub_indent = '  ' * level
        # Process files: A file is included only if a rule includes it.
        for file in sorted(files):
            filepath = os.path.join(root, file)
            if should_process_path(filepath, rules, is_dir=False):
                structure.append(f"{sub_indent}{file}")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    content = f"[Error reading file: {e}]"
                file_contents.append((filepath, content))

    return structure, file_contents

def write_output(output_path, structure, file_contents):
    """Writes the collected structure and contents to the output file."""
    with open(output_path, 'w', encoding='utf-8') as out:
        out.write("--- File Structure ---\n")
        for line in structure:
            out.write(line + '\n')

        out.write("\n--- File Contents ---\n")
        for path, content in file_contents:
            out.write(f"\n## {path} ##\n")
            out.write(content + '\n')

if __name__ == "__main__":
    # --- Set up command-line argument parsing ---
    parser = argparse.ArgumentParser(
        description="Extracts a project's structure and file contents into a single text file based on wildcard rules."
    )
    parser.add_argument(
        '-i', '--input',
        default='folders.txt',
        help='The input file with inclusion/exclusion rules. Default: folders.txt'
    )
    parser.add_argument(
        '-o', '--output',
        default='output.txt',
        help='The path for the generated output file. Default: output.txt'
    )
    args = parser.parse_args()
    
    # Use the filenames from the parsed arguments
    folders_txt = args.input
    output_txt = args.output
    # --- End of argument parsing ---

    print(f"Reading rules from '{folders_txt}'...")
    rules = parse_rules(folders_txt)
    
    # Proceed only if the rules file was successfully read
    if rules is not None:
        print(f"Found {len(rules)} rules.")
        print("Processing...")
        
        structure, contents = walk_and_collect_structure(rules)
        write_output(output_txt, structure, contents)
        
        print(f"\nSuccess! Project structure and contents written to '{output_txt}'.")
    
    input("Press Enter to exit...") # Optional: uncomment to pause before exit