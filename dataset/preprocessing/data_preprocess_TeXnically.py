import re

# === Stage 1: Fix spaces around \left and \right ===
def fix_left_right_spaces(line):
    # Replace \left followed by spaces and \| with \left\|
    line = re.sub(r'\\left\s+\\\|\s*', r'\\left\| ', line)
    # Replace \right followed by spaces and \| with \right\|
    line = re.sub(r'\\right\s+\\\|\s*', r'\\right\| ', line)

    # Define a regex pattern to match \left or \right followed by specific symbols
    pattern = re.compile(
        r'''
        \\ (left|right)       # Match \left or \right
        \s+                   # Followed by one or more spaces
        (
            \\ (?:            
                langle|rangle|lfloor|rfloor|lceil|rceil|  # Match specific LaTeX symbols
                ulcorner|urcorner|llcorner|lrcorner|
                uparrow|Uparrow|downarrow|Downarrow|
                vert
            )
            | \$${}|()\[$$<>]  # Match special characters like $, {}, (), [], <, >
            | [][|()<>.,]      # Match other symbols like brackets, pipes, etc.
            | \.               # Match a dot
        )
        ''',
        flags=re.VERBOSE
    )
    # Remove unnecessary spaces between \left/\right and the following symbol
    line = pattern.sub(r'\\\1\2', line)

    # Ensure proper spacing after \left\| and \right\|
    line = re.sub(r'(\\left\\\|)\s+', r'\1 ', line)
    line = re.sub(r'(\\right\\\|)\s+', r'\1 ', line)
    return line


# === Stage 2: Simplify \operatorname{...} ===
# List of common mathematical operators
operators = ['arccos', 'arcsin', 'arctan', 'arg', 'cos', 'cosh', 'cot', 'coth', 'csc', 'deg',
             'det', 'dim', 'exp', 'gcd', 'hom', 'inf', 'injlim', 'ker', 'lg', 'lim', 'liminf',
             'limsup', 'ln', 'log', 'max', 'min', 'Pr', 'projlim', 'sec', 'sin', 'sinh',
             'sup', 'tan', 'tanh']
operator_set = set(operators)

# Define a regex pattern to match \operatorname{...}
operator_pattern = re.compile(
    r'''
    \\operatorname\s*\*?\s*  # Match \operatorname or \operatorname*
    {\s*                     # Opening brace
    (
        (?:
            [a-zA-Z]          # Match letters
            \s*               # Allow spaces between letters
        )+
    )
    \s*}                     # Closing brace
    ''',
    flags=re.VERBOSE
)

# Replace \operatorname{...} with the simplified form if valid
def replace_operator(match):
    letters = re.sub(r'\s+', '', match.group(1))  # Remove spaces within the operator name
    return f'\\{letters}' if letters in operator_set else match.group(0)


# === Stage 3: Merge consecutive spacing tokens ===
def replace_spacing_tokens(tokens):
    new_tokens = []
    i = 0
    while i < len(tokens):
        # Check for spacing tokens like \,, \:, or \;
        if tokens[i] in ["\\,", "\\:", "\\;"]:
            sym = tokens[i]
            count = 1
            j = i + 1
            # Count consecutive occurrences of the same spacing token
            while j < len(tokens) and tokens[j] == sym:
                count += 1
                j += 1

            # Replace based on the number of consecutive tokens
            if sym == "\\,":
                if 1 <= count <= 2:
                    new_tokens.extend([sym] * count)
                elif 3 <= count <= 4:
                    new_tokens.append("\\:")
                elif 5 <= count <= 8:
                    new_tokens.append("\\quad")
                else:
                    new_tokens.append("\\qquad")
            elif sym == "\\:":
                if count == 1:
                    new_tokens.append("\\:")
                elif 4 <= count <= 6:
                    new_tokens.append("\\quad")
                elif count >= 7:
                    new_tokens.append("\\qquad")
                else:
                    new_tokens.extend([sym] * count)
            elif sym == "\\;":
                if 1 <= count <= 2:
                    new_tokens.extend([sym] * count)
                elif 3 <= count <= 6:
                    new_tokens.append("\\quad")
                else:
                    new_tokens.append("\\qquad")

            i = j
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens


# === Stage 4: Remove redundant nested braces ===
def protect_braces(formula):
    # Protect braces by replacing them with placeholders
    pattern = re.compile(r'(\\(?!begin|end)[a-zA-Z]+)\s*\{([^{}]+?)\}')
    prev_formula = None
    while prev_formula != formula:
        prev_formula = formula
        formula = pattern.sub(lambda m: m.group(1) + ' <LB>' + m.group(2) + '<RB>', formula)

    # Handle cases like ^{...} or _{...}
    formula = re.sub(r'([\^_])\s*\{([^{}]+?)\}', lambda m: m.group(1) + ' <LB>' + m.group(2) + '<RB>', formula)
    formula = re.sub(r'\\left\s*\{([^{}]+?)\}', r'\\left<LB>\1<RB>', formula)
    formula = re.sub(r'\\right\s*\{([^{}]+?)\}', r'\\right<LB>\1<RB>', formula)
    return formula

def restore_braces(formula):
    # Restore braces from placeholders
    return formula.replace('<LB>', '{').replace('<RB>', '}')

def remove_redundant_nested_braces(formula):
    chars = list(formula)
    stack = []
    remove_indices = set()
    for i, ch in enumerate(chars):
        if ch == '{':
            stack.append(i)
        elif ch == '}':
            if not stack:
                continue
            start = stack.pop()
            inner = ''.join(chars[start + 1:i]).strip()
            # Remove redundant braces if the inner content is already enclosed
            if inner.startswith('{') and inner.endswith('}'):
                remove_indices.add(start)
                remove_indices.add(i)
    return ''.join([ch for idx, ch in enumerate(chars) if idx not in remove_indices])


# === Main cleaning function ===
def clean_line(line):
    # 1. Fix spaces around \left and \right
    line = fix_left_right_spaces(line.strip())

    # 2. Replace \operatorname{...} with simplified forms
    line = operator_pattern.sub(replace_operator, line)

    # 3. Merge consecutive spacing tokens
    tokens = line.split()
    tokens = replace_spacing_tokens(tokens)
    line = ' '.join(tokens)

    # 4. Remove redundant nested braces
    line = protect_braces(line)
    line = remove_redundant_nested_braces(line)
    line = restore_braces(line)

    # Remove extra spaces
    line = re.sub(r' {2,}', ' ', line)
    return line.strip()


# === Process files in bulk ===
def clean_latex_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            cleaned = clean_line(line)
            f_out.write(cleaned + '\n')
    print(f"✅ Processing complete. Output saved to: {output_file}")


# === Example entry point ===
if __name__ == '__main__':
    input_path = '../data/txt/celery.txt'
    output_path = '../data/txt/celery_nor.txt'
    clean_latex_dataset(input_path, output_path)