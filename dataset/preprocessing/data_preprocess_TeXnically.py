import re

# ========== Step 1: left right 左右间隔 ==========
def fix_left_right_spaces(line):
    line = re.sub(r'\\left\s+\\\|\s*', r'\\left\| ', line)
    line = re.sub(r'\\right\s+\\\|\s*', r'\\right\| ', line)

    pattern = re.compile(
        r'''
        \\(left|right)\s+(
            \\(?:langle|rangle|lfloor|rfloor|lceil|rceil|
                ulcorner|urcorner|llcorner|lrcorner|
                uparrow|Uparrow|downarrow|Downarrow|vert)
            | \$${}|()\[$$<>]
            | [][|()<>.,]
            | \.
        )
        ''',
        flags=re.VERBOSE
    )
    line = pattern.sub(r'\\\1\2', line)
    line = re.sub(r'(\\left\\\|)\s+', r'\1 ', line)
    line = re.sub(r'(\\right\\\|)\s+', r'\1 ', line)
    return line


# ========== Step 2: 简化操作符 ==========
operators = {
    'arccos', 'arcsin', 'arctan', 'arg', 'cos', 'cosh', 'cot', 'coth', 'csc',
    'deg', 'det', 'dim', 'exp', 'gcd', 'hom', 'inf', 'injlim', 'ker', 'lg',
    'lim', 'liminf', 'limsup', 'ln', 'log', 'max', 'min', 'Pr', 'projlim',
    'sec', 'sin', 'sinh', 'sup', 'tan', 'tanh'
}
operator_pattern = re.compile(
    r'\\operatorname\s*\*?\s*{\s*((?:[a-zA-Z]\s*)+)\s*}',
    flags=re.VERBOSE
)

def replace_operator(match):
    op = re.sub(r'\s+', '', match.group(1))
    return f'\\{op}' if op in operators else match.group(0)


# ========== Step 3: 移除和替换间距命令 ==========
def replace_spacing_tokens(tokens):
    result, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (
            (tokens[i], tokens[i + 1]) in [("\\,", "\\!"), ("\\!", "\\,")]):
            i += 2
            continue
        if tokens[i] == "\\!":
            i += 1
            continue
        if tokens[i] in ["\\,", "\\:", "\\;", "\\"]:
            sym = tokens[i]
            count, j = 1, i + 1
            while j < len(tokens) and tokens[j] == sym:
                count += 1
                j += 1
            if sym == "\\,":
                if count <= 2: result.extend([sym] * count)
                elif count <= 4: result.append("\\:")
                elif count <= 8: result.append("\\quad")
                else: result.append("\\qquad")
            elif sym == "\\:":
                if count == 1: result.append("\\:")
                elif count <= 3: result.extend([sym] * count)
                elif count <= 6: result.append("\\quad")
                else: result.append("\\qquad")
            elif sym == "\\;":
                if count <= 2: result.extend([sym] * count)
                elif count <= 6: result.append("\\quad")
                else: result.append("\\qquad")
            elif sym == "\\":
                if count <= 2: result.append("\\")
                elif count <= 6: result.append("\\quad")
                else: result.append("\\qquad")
            i = j
        else:
            result.append(tokens[i])
            i += 1

    # 取消末尾间距命令
    # spacing_cmds = {"\\,", "\\:", "\\;", "\\quad", "\\qquad", "\\", "~"}
    # for start_idx in [len(result)-1, len(result)-2]:
    #     if start_idx >= 0 and result[start_idx] in spacing_cmds:
    #         end_idx = start_idx
    #         while end_idx >= 0 and result[end_idx] in spacing_cmds:
    #             end_idx -= 1
    #         result = result[:end_idx + 1] + result[start_idx + 1:]
    #         break
    return result


# ========== Step 4: 移除无法渲染的命令 \ref{}, \label{}, etc ==========
def remove_ref_and_label(tokens):
    s = " ".join(tokens)
    s = re.sub(r'\\hline\s*', '', s)
    s = re.sub(r'\\left\(\s*\\ref\s*\{[^}]*\}\s*\\right\)', '', s)
    s = re.sub(r'(?:^|[^\\])\(?\s*\\ref\s*\{[^}]*\}\s*\)?', '', s)
    s = re.sub(r'(?:\{\s*)?\\label\s+[^\s}]+(?:\s*\})?', '', s)
    s = re.sub(r'\\slash\s*', '', s)
    return s.split()

# ========== Step 5: 移除花括号 ==========
def protect_braces(text):
    pattern = re.compile(r'(\\(?!begin|end)[a-zA-Z]+)\s*\{([^{}]+?)\}')
    prev = None
    while prev != text:
        prev = text
        text = pattern.sub(lambda m: m.group(1) + ' <LB>' + m.group(2) + '<RB>', text)
    text = re.sub(r'([\^_])\s*\{([^{}]+?)\}', lambda m: m.group(1) + ' <LB>' + m.group(2) + '<RB>', text)
    text = re.sub(r'\\left\s*\{([^{}]+?)\}', r'\\left<LB>\1<RB>', text)
    text = re.sub(r'\\right\s*\{([^{}]+?)\}', r'\\right<LB>\1<RB>', text)
    return text

def restore_braces(text):
    return text.replace('<LB>', '{').replace('<RB>', '}')

def remove_redundant_nested_braces(text):
    chars, stack, remove = list(text), [], set()
    for i, ch in enumerate(chars):
        if ch == '{': stack.append(i)
        elif ch == '}' and stack:
            start = stack.pop()
            inner = ''.join(chars[start + 1:i]).strip()
            if inner.startswith('{') and inner.endswith('}'):
                remove.update({start, i})
    return ''.join(ch for i, ch in enumerate(chars) if i not in remove)

# ========== Step 6: 处理 \mathrm { e x p } 变为 \exp ==========
def replace_exp(line):
    line = re.sub(r'\\mathrm\s*\{\s*e\s*x\s*p\s*\}', r'\\exp', line)

    # 处理 "e x p"、"\ e x p"（未被 \mathrm 包裹）
    line = re.sub(r'\\?e\s*x\s*p', r'\\exp', line)  # 匹配空格形式的 "e x p"
    line = re.sub(r'\\s*\\?e\s*x\s*p', r'\\exp', line)  # 匹配被空格断开的形式

    return line

# ========== Step 7: 处理  \mathrm { ~ \boldmath ~ ... } ==========
def replace_boldmath(line):
    # 匹配 \mathrm { ~ \boldmath ~ ... } 的结构
    pattern_1 = r"\\mathrm\s*\{\s*\\boldmath\s*(.*?)\s*\}"

    # 替换函数
    def replace_match_1(match):
        content = match.group(1).strip()  # 提取 \boldmath 后的内容

        # 如果内容中含有 \scriptstyle 或 \footnotesize，保留它们并转换为 \mathbf
        if '\\scriptstyle' in content or '\\footnotesize' in content:
            content = content.replace('\\boldmath', '')  # 移除 \boldmath
            return f"\\mathbf{{ {content} }}"  # 替换为 \mathbf

        # 否则直接转换为 \mathbf
        return f"\\mathbf{{ {content} }}"

    # 先替换带 \boldmath 的内容
    processed_line = re.sub(pattern_1, replace_match_1, line)

    # 处理 \boldmath 和 \scriptstyle 的组合结构
    # 这里我们仅仅处理 \boldmath 后的部分，保持结构不变
    pattern_2 = r"\\boldmath\s*([^}]+)"

    def replace_match_2(match):
        content = match.group(1).strip()  # 提取 \boldmath 后的内容
        # 仅替换为 \mathbf，而不影响其他的格式
        return f"\\mathbf{{ {content} }}"

    # 使用第二个正则表达式处理剩余内容
    processed_line = re.sub(pattern_2, replace_match_2, processed_line)

    return processed_line

# ========== Step 8: 处理单个字母命令 ==========
def replace_single_letter_commands(line):
    """
    将 \ 后紧跟单个字母的命令替换为对应的字母，但保留合法的多字母命令（如 \bigl, \bigr）。
    """
    # 定义正则表达式，匹配 \ 后紧跟单个字母的情况
    pattern = r"""
        \\([A-Za-z])  # 匹配 \ 后紧跟一个大小写字母
        (?![A-Za-z])  # 确保后面没有其他字母（即不是多字母命令）
    """

    # 替换逻辑
    def replace_match(match):
        letter = match.group(1)  # 提取匹配到的字母
        return letter  # 返回对应的字母

    # 使用 re.sub 应用替换逻辑
    line = re.sub(pattern, replace_match, line, flags=re.VERBOSE)
    return line

# ========== Step 9: 处理特殊符号 ==========
def replace_special_symbols(line):
    # 1. 将 ~ | ~ 替换为 \mid
    line = re.sub(r'~\s*\|\s*~', r'\\mid', line)

    # 2. 将 \begin {array} 替换为 \begin{array}
    line = re.sub(r'\\begin\s*\{\s*array\s*\}', r'\\begin{array}', line)

    # 3. 将 \end {array} 替换为 \end{array}
    line = re.sub(r'\\end\s*\{\s*array\s*\}', r'\\end{array}', line)

    # 4. 将 \footnotesize 替换为 \scriptstyle
    line = re.sub(r'\\footnotesize', r'\\scriptstyle', line)

    return line

# ========== Step 10: 处理 \mathbb { ... } ==========
def replace_bf_with_mathbf(line):
    # 匹配 { \bf ... } 的结构，去掉外部的花括号，并替换 \bf 为 \mathbf
    line = re.sub(r'\{\s*\\bf\s*(.*?)\s*\}', r'\\mathbf { \1 }', line)
    # 匹配 \bf 后面没有花括号的情况
    line = re.sub(r'\\bf\s*([a-zA-Z0-9]+)', r'\\mathbf { \1 }', line)
    line = re.sub(r'\\mathbf\{', r'\\mathbf {', line)
    return line

# ========= Step 11: 处理 ~ 符号 ==========
import re

def replace_tildes(line):
    # 首先处理行尾的单个 ~，保持不变
    line = re.sub(r'([^~])~\s*$', r'\1~', line)

    # 将每一组连续的 ~ 变成一个整体，然后根据数量处理
    def process_tilde_group(match):
        tilde_count = len(match.group(0).replace(' ', ''))  # 去掉空格后计算 ~ 的数量
        result = []

        # 每 7 个 ~ 替换为 \qquad
        while tilde_count >= 7:
            result.append('\\qquad ')
            tilde_count -= 7

        # 每 6 个 ~ 替换为 \qquad
        if tilde_count >= 6:
            result.append('\\qquad ')
            tilde_count -= 6

        # 每 2 到 5 个 ~ 替换为 \quad
        if 2 <= tilde_count <= 5:
            result.append('\\quad ')
            tilde_count -= tilde_count

        # 剩余的单个 ~ 保持不变
        if tilde_count == 1:
            result.append('~')

        return ''.join(result)

    # 使用正则表达式匹配连续的 ~，并调用 process_tilde_group 处理
    line = re.sub(r'~(\s*~)*', process_tilde_group, line)

    return line


# ========= Step 12: 处理希腊字母 ==========
def replace_greek_letters(line):
    # 常见希腊字母列表
    greek_letters = [
        'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
        'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi', 'rho',
        'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega',
        'varepsilon', 'vartheta', 'varpi', 'varrho', 'varsigma', 'varphi',
    ]

    # 构造正则表达式，匹配被空格断开的 \ a l p h a 形式
    for name in greek_letters:
        spaced = r'\\\s*' + r'\s*'.join(name)  # 构造形如 \ a l p h a 的正则
        pattern = re.compile(spaced)
        line = pattern.sub(r'\\' + name, line)

    return line

# ========= Step 12: 处理{ \cal X }命令 ==========
def remove_cal_commands(line):
    # 匹配 { \cal X } 的形式，并提取字母部分
    line = re.sub(r'\{\s*\\cal\s+([A-Za-z])\s*\}', r'\1', line)
    return line


# ========== Main cleaning function ==========
def clean_line(line, steps=None):
    if steps is None:
        steps = {
            "fix_left_right": False,
            "replace_sp": False,
            "replace_greek": False,
            "replace_exp": False,
            "replace_boldmath": False,
            "simplify_operatorname": False,
            "replace_single_letter_cmd": False,
            "spacing_tokens": True,
            "remove_refs_labels": False,
            "redundant_braces": False, # 移除花括号
            "replace_special": False,
            "replace_bf": False, # 处理 \mathbb { ... }
            "replace_tildes": False,
            "remove_cal": False  # 新增步骤
        }

    line = line.strip()
    if steps["fix_left_right"]:
        line = fix_left_right_spaces(line)
    if steps["replace_sp"]:
        line = line.replace(r'\sp', '^')
    if steps["replace_greek"]:
        line = replace_greek_letters(line)
    if steps["replace_exp"]:
        line = replace_exp(line)
    if steps["replace_boldmath"]:
        line = replace_boldmath(line)
    if steps["simplify_operatorname"]:
        line = operator_pattern.sub(replace_operator, line)
    if steps["replace_single_letter_cmd"]:
        line = replace_single_letter_commands(line)


    tokens = line.split()
    if steps["spacing_tokens"]:
        tokens = replace_spacing_tokens(tokens)
    if steps["remove_refs_labels"]:
        tokens = remove_ref_and_label(tokens)
    line = ' '.join(tokens)

    if steps["redundant_braces"]:
        line = protect_braces(line)
        line = remove_redundant_nested_braces(line)
        line = restore_braces(line)

    if steps["replace_special"]:
        line = replace_special_symbols(line)
    if steps["remove_cal"]:
        line = remove_cal_commands(line)
    if steps["replace_bf"]:
        line = replace_bf_with_mathbf(line)
    if steps["replace_tildes"]:
        line = replace_tildes(line)

    line = re.sub(r' {2,}', ' ', line)
    return line.strip()

# ========== Bulk file processing ==========
def clean_latex_dataset(input_file, output_file, steps=None):
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            cleaned = clean_line(line, steps)
            fout.write(cleaned + '\n')
    print(f"✅ Done. Output saved to {output_file}")



# === Example entry point ===
if __name__ == '__main__':
    # 输入文件路径和输出文件路径
    input_path = r'D:\Desktop\lab\code\TeXnical\TeXnically\dataset\data\txt\full_math_0419.txt'  # 输入文件路径
    output_path = r'D:\Desktop\lab\code\TeXnical\TeXnically\dataset\data\txt\full_math_end.txt'  # 输出文件路径

    # 可以选择性地设置步骤字典
    steps = {
        "fix_left_right": True,
        "replace_sp": True,
        "replace_greek": True,
        "replace_exp": True,
        "replace_boldmath": True,
        "simplify_operatorname": True,
        "replace_single_letter_cmd": True,
        "spacing_tokens": False,
        "remove_refs_labels": True,
        "redundant_braces": True,  # 移除花括号
        "replace_special": True,
        "replace_bf": True,  # 处理 \mathbb { ... }
        "replace_tildes": True,
        "remove_cal": True  # 新增步骤
    }

    # 调用批量处理函数
    clean_latex_dataset(input_path, output_path, steps)
