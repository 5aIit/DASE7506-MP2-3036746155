"""Features for the frozen selector."""

import ast

from collections import Counter

from fractions import Fraction

import math

import re

import numpy as np

from common import parse_answer

_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")

_WORD = re.compile(r"[a-z]+")

_STOP = set("a an the is are was were be been being to of in on for with at by and or as if then than that this these those it its he she his her they their them we you i how what when which who many much does do did will would can could should have has had get gets find calculate determine total answer step let now so from each all more".split())

_LATEX_FRAC = re.compile(r"\\(?:dfrac|tfrac|frac)\{([^{}]*)\}\{([^{}]*)\}")

_LATEX_WRAPPER = re.compile(r"\\(?:boxed|mathbf|mathrm|textbf)\{([^{}]*)\}")

_LATEX_TEXT = re.compile(r"\\(?:text|mbox)\{[^{}]*\}")

_LITERAL = r"[\d\s.()+*/%\-]+"

_LEFT_EXPR = re.compile(r"(" + _LITERAL + r")$")

_RIGHT_EXPR = re.compile(r"^(" + _LITERAL + r")")

ANSWER_NAMES = [
    "valid", "vote_count", "distinct_count", "largest_vote_count", "vote_margin",
    "plurality_tie", "log_magnitude", "signed_log", "integer", "negative", "zero",
    "log_denominator", "equals_question_number", "relative_to_question_scale",
    "numeric_rank", "distance_group_median", "body_mentions", "final_number_match",
    "largest_relative_disagreement", "others_body_mentions",
]

STYLE_NAMES = [
    "log_chars", "log_words", "relative_chars", "chars_rank", "log_lines",
    "log_paragraphs", "log_numbers", "equation_count", "boxed_count",
    "final_marker_count", "hash_marker", "bullet_count", "step_count",
    "questionmark_count", "fraction_count", "percent_count", "currency_count",
    "round_count", "approximation_count", "caution_count", "assumption_count",
    "revision_count", "alternative_count", "certainty_count", "unable_count",
    "negation_count", "word_repetition", "final_marker_position", "variable_count",
]

ARITHMETIC_NAMES = [
    "check_count", "correct_count", "error_count", "error_fraction",
    "max_relative_error", "last_check_correct", "answer_result_count",
    "answer_result_fraction", "plus_count", "minus_count", "multiply_count",
    "divide_count", "relative_equation_count", "check_coverage",
    "distinct_result_count", "last_result_answer", "relative_correct_count",
    "relative_error_count",
]

SEMANTIC_NAMES = [
    "question_word_recall", "target_word_recall", "question_number_recall",
    "number_novel_fraction", "number_question_fraction", "target_tail_recall",
    "question_log_chars", "embedding_mean", "embedding_max", "embedding_min",
    "embedding_std", "same_answer_embedding_mean", "different_answer_embedding_mean",
    "embedding_agreement_gap", "embedding_centroid_cosine", "answer_peer_jaccard",
    "all_peer_jaccard",
]

FEATURE_NAMES = (["answer_" + n for n in ANSWER_NAMES]
                 + ["style_" + n for n in STYLE_NAMES]
                 + ["arithmetic_" + n for n in ARITHMETIC_NAMES]
                 + ["semantic_" + n for n in SEMANTIC_NAMES])

def answer_groups(bank):
    """Return canonical rational-answer strings (or None), in bank order.

    Canonicalization delegates to the unchanged official parser: equivalent
    fractions and decimals vote together; invalid answers never receive votes.
    """
    return [[parse_answer(c["text"]) for c in row["candidates"]] for row in bank]

def _numbers(text):
    values = []
    for match in _NUMBER.finditer(text):
        try:
            values.append(str(Fraction(match.group().replace(",", ""))))
        except (ValueError, ZeroDivisionError):
            pass
    return values

def _float(value):
    if value is None:
        return 0.0
    try:
        number = float(Fraction(value))
        return min(1e15, max(-1e15, number))
    except (ValueError, OverflowError, ZeroDivisionError):
        return 0.0

def _words(text):
    # A small, fixed plural normalization; not a learned language model.
    return [w[:-1] if len(w) > 4 and w.endswith("s") else w
            for w in _WORD.findall(text.lower()) if w not in _STOP]

def _recall(reference, observed):
    return len(reference & observed) / max(1, len(reference))

def _jaccard(first, second):
    return len(first & second) / max(1, len(first | second))

def _count(pattern, text):
    return len(re.findall(pattern, text))

def _normalize_math(text):
    """Normalize a restricted set of ordinary arithmetic/LaTeX spellings."""
    # A Markdown bullet is formatting, not a unary minus on the first quantity.
    text = re.sub(r"(?m)^\s*(?:[-*•]\s+|\d+\.\s+)", "", text)
    text = text.replace(",", "")
    # Bounded passes support nested fractions while keeping preprocessing bounded.
    for _ in range(4):
        newer = _LATEX_FRAC.sub(r"((\1)/(\2))", text)
        newer = _LATEX_WRAPPER.sub(r"(\1)", newer)
        if newer == text:
            break
        text = newer
    # Preserve an alphabetic barrier: removing units/variable descriptions can
    # turn '6 hours = 360 minutes' into a spurious numeric inequality, or erase
    # a named variable from an expression. Such equalities are left unchecked.
    text = _LATEX_TEXT.sub(" TEXT ", text)
    for before, after in (("\\times", "*"), ("\\cdot", "*"), ("×", "*"),
                          ("\\div", "/"), ("÷", "/"), ("−", "-"),
                          ("\\left", ""), ("\\right", ""), ("\\%", "%")):
        text = text.replace(before, after)
    text = text.replace("**", "").replace("$", "").replace("\\,", " ")
    return text

def _literal_value(expression):
    """Evaluate a tiny AST arithmetic grammar; never call eval/exec.

    Only finite numeric constants, parentheses, unary signs and + - * / are
    accepted. Percent suffixes are interpreted as division by 100. Variables,
    functions, attributes, exponentiation and all other Python nodes are rejected.
    """
    expression = expression.strip().rstrip(".").strip()
    if not expression or len(expression) > 200 or not re.search(r"\d", expression):
        return None
    expression = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expression)
    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 80:
            return None

        def visit(node):
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                result = float(node.value)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                left, right = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Add):
                    result = left + right
                elif isinstance(node.op, ast.Sub):
                    result = left - right
                elif isinstance(node.op, ast.Mult):
                    result = left * right
                else:
                    result = left / right
            else:
                raise ValueError("Not literal arithmetic")
            if not math.isfinite(result) or abs(result) > 1e15:
                raise ValueError("Outside bounded arithmetic range")
            return result

        return visit(tree)
    except (ArithmeticError, ValueError, SyntaxError, RecursionError):
        return None

def _arithmetic(text):
    normalized = _normalize_math(text)
    checked = []
    for line in normalized.splitlines():
        sections = line.split("=")
        for left, right in zip(sections, sections[1:]):
            lm, rm = _LEFT_EXPR.search(left), _RIGHT_EXPR.match(right)
            if lm is None or rm is None:
                continue
            lexpr, rexpr = lm.group(1).strip(), rm.group(1).strip()
            # School solutions mix fractions and percentage-point notation,
            # e.g. 8/25*100 = 32%. Checking that convention requires semantics.
            if "%" in lexpr or "%" in rexpr:
                continue
            # Do not treat the numeric fragment of a variable term (x2, 2x)
            # or LaTeX argument as a standalone arithmetic expression.
            if lm.start() and left[lm.start()-1].isalnum():
                continue
            if rm.end() < len(right) and right[rm.end()].isalnum() and not right[rm.end()-1].isspace():
                continue
            if re.match(r"[xyz]\b", right[rm.end():]):
                continue
            lv, rv = _literal_value(lexpr), _literal_value(rexpr)
            if lv is not None and rv is not None:
                error = abs(lv - rv) / max(1.0, abs(lv), abs(rv))
                checked.append((rv, error <= 1e-6, error))
    return normalized, checked

def extract_features(bank, embeddings):
    """Return ``(float32[N,4,84], names)`` using only prediction inputs.

    The schema is fixed and also works on a new/audit bank or an empty batch.
    ``embeddings`` must be the supplied array with shape [N,4,D]. Similarity
    statistics use cosine normalization and need no fitted transformation.
    """
    n = len(bank)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 3 or embeddings.shape[:2] != (n, 4):
        raise ValueError("Expected embeddings shaped [N, 4, D]")
    if not np.isfinite(embeddings).all():
        raise ValueError("Nonfinite embeddings")
    x = np.empty((n, 4, len(FEATURE_NAMES)), dtype=np.float32)
    norm = np.maximum(np.linalg.norm(embeddings, axis=-1, keepdims=True), 1e-8)
    unit = embeddings / norm
    similarities = np.einsum("nkd,nld->nkl", unit, unit)
    centroid = unit.mean(axis=1)
    centroid /= np.maximum(np.linalg.norm(centroid, axis=-1, keepdims=True), 1e-8)
    cent_sim = np.einsum("nkd,nd->nk", unit, centroid)
    all_answers = answer_groups(bank)

    for row_idx, (row, answers) in enumerate(zip(bank, all_answers)):
        if len(row["candidates"]) != 4:
            raise ValueError("Expected exactly four candidates per question")
        question = row["question"]
        qwords, qnums = set(_words(question)), set(_numbers(question))
        target = re.split(r"(?<=[?.!])\s+", question.strip())[-1]
        target_words = set(_words(target))
        question_scale = max([abs(_float(v)) for v in qnums] + [1.0])
        votes = Counter(a for a in answers if a is not None)
        most_votes = max(votes.values(), default=0)
        tied = sum(v == most_votes for v in votes.values()) > 1
        values = [_float(a) for a in answers]
        valid_values = [v for v, a in zip(values, answers) if a is not None]
        median = float(np.median(valid_values)) if valid_values else 0.0
        texts = [str(c["text"]) for c in row["candidates"]]
        words = [_words(t) for t in texts]
        word_sets = [set(w) for w in words]
        numbers = [_numbers(t) for t in texts]
        # Exclude the final-answer declaration from body occurrence counts.
        bodies = [re.split(r"####|\\boxed\{", t)[0] for t in texts]
        body_numbers = [Counter(_numbers(t)) for t in bodies]
        lengths = np.asarray([len(t) for t in texts], dtype=float)
        arith = [_arithmetic(t) for t in texts]
        eq_counts = np.asarray([t.count("=") for t in texts], dtype=float)
        correct_counts = [sum(v[1] for v in checks) for _, checks in arith]
        error_counts = [len(checks) - count for (_, checks), count in zip(arith, correct_counts)]

        for j, (text, answer, value) in enumerate(zip(texts, answers, values)):
            lower = text.lower()
            valid = answer is not None
            fraction = Fraction(answer) if valid else Fraction(0)
            magnitude = math.log1p(abs(value))
            vote = votes.get(answer, 0)
            ans_features = [
                valid, vote, len(votes), most_votes, vote-most_votes, tied,
                magnitude, math.copysign(magnitude, value),
                valid and fraction.denominator == 1, valid and value < 0, valid and value == 0,
                math.log1p(min(fraction.denominator, 10**15)), answer in qnums if valid else 0,
                math.log1p(abs(value)/question_scale),
                sum(v < value for v in valid_values)/max(1, len(valid_values)),
                abs(value-median)/max(1.0, abs(value), abs(median)),
                body_numbers[j].get(answer, 0), bool(numbers[j]) and numbers[j][-1] == answer,
                max([abs(value-v)/max(1.0, abs(value), abs(v)) for v in valid_values] + [0]),
                sum(body_numbers[k].get(answer, 0) for k in range(4) if k != j),
            ]
            final_positions = [m.start() for m in re.finditer(r"final\s+answer|####|\\boxed", lower)]
            style_features = [
                math.log1p(len(text)), math.log1p(len(words[j])),
                len(text)/max(1.0, lengths.mean()), sum(lengths < len(text))/4,
                math.log1p(text.count("\n")+1), math.log1p(len(re.split(r"\n\s*\n", text))),
                math.log1p(len(numbers[j])), text.count("="), text.count("\\boxed"),
                _count(r"final\s+answer", lower), "####" in text,
                _count(r"(?m)^\s*(?:[-*]|\d+\.)\s", text), _count(r"\bstep\s*\d*", lower),
                text.count("?"), _count(r"\\(?:dfrac|tfrac|frac)\{", text), text.count("%"),
                _count(r"\$|\bdollars?\b|\bcents?\b", lower),
                _count(r"\bround(?:ing|ed)?\b|\bnearest\b", lower),
                _count(r"\bapprox(?:imately)?\b|\babout\b|≈|\\approx", lower),
                _count(r"\b(?:however|but|unless|ambiguous|unclear|might|maybe|perhaps)\b", lower),
                _count(r"\bassum(?:e|ed|ing|ption)\b", lower),
                _count(r"\b(?:wait|mistake|incorrect|reconsider|correction|instead)\b", lower),
                _count(r"\b(?:alternatively|otherwise|either)\b", lower),
                _count(r"\b(?:clearly|obviously|certainly|definitely)\b|✅", lower),
                _count(r"\b(?:impossible|insufficient|undefined|cannot)\b|not enough", lower),
                _count(r"\b(?:not|no|never)\b", lower),
                1-len(word_sets[j])/max(1, len(words[j])),
                max(final_positions, default=0)/max(1, len(text)),
                _count(r"\b[xyz]\b", lower),
            ]
            normalized, checks = arith[j]
            correct = correct_counts[j]
            errors = error_counts[j]
            result_matches = [valid and abs(result-value) <= 1e-6*max(1.0, abs(result), abs(value))
                              for result, _, _ in checks]
            arithmetic_features = [
                len(checks), correct, errors, errors/max(1, len(checks)),
                max([e for _, _, e in checks] + [0]), checks[-1][1] if checks else 0,
                sum(result_matches), sum(result_matches)/max(1, len(checks)),
                normalized.count("+"), normalized.count("-"), normalized.count("*"), normalized.count("/"),
                eq_counts[j]/max(1.0, eq_counts.mean()), len(checks)/max(1.0, eq_counts[j]),
                len(set(round(result, 6) for result, _, _ in checks)),
                result_matches[-1] if result_matches else 0,
                correct-float(np.mean(correct_counts)), errors-float(np.mean(error_counts)),
            ]
            peers = [k for k in range(4) if k != j]
            same = [k for k in peers if valid and answers[k] == answer]
            different = [k for k in peers if answers[k] != answer]
            sim = similarities[row_idx, j]
            same_mean = float(np.mean(sim[same])) if same else 0.0
            different_mean = float(np.mean(sim[different])) if different else 0.0
            nums = set(numbers[j])
            semantic_features = [
                _recall(qwords, word_sets[j]), _recall(target_words, word_sets[j]), _recall(qnums, nums),
                len(nums-qnums)/max(1, len(nums)), len(nums & qnums)/max(1, len(nums)),
                _recall(target_words, set(_words(text[-max(120, len(text)//4):]))),
                math.log1p(len(question)), float(np.mean(sim[peers])), float(np.max(sim[peers])),
                float(np.min(sim[peers])), float(np.std(sim[peers])), same_mean, different_mean,
                same_mean-different_mean, cent_sim[row_idx, j],
                float(np.mean([_jaccard(word_sets[j], word_sets[k]) for k in same])) if same else 0,
                float(np.mean([_jaccard(word_sets[j], word_sets[k]) for k in peers])),
            ]
            x[row_idx, j] = ans_features + style_features + arithmetic_features + semantic_features

    np.nan_to_num(x, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    return x, list(FEATURE_NAMES)
