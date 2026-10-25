"""Semantic features for the frozen selector."""

from collections import Counter

from fractions import Fraction

import ast

import math

import re

import numpy as np

from common import parse_answer

from selector.features import _literal_value, _normalize_math, _words

from selector.reasoning_features import WORD_NUMBERS, literal_checks

_NUMBER = re.compile(r'(?<![\w.])[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:/\d+)?')

_TOKEN = re.compile(r"[A-Za-z]+|(?<![\w.])[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:/\d+)?")

_LEFT = re.compile(r'([\d\s.()+*/\-]+)$')

_RIGHT = re.compile(r'^([\d\s.()+*/\-]+)')

_FORMATTING = re.compile(r'\\(?:text|mathrm|mbox|textbf)\{([^{}]*)\}')

_CONTEXT_STOP = set('number amount quantity value final result numeric solution problem given need needed want wants asked find calculate determine total step using use used equals equal answer times plus minus divided multiplied much many long does will would can should has have had gets get got now then therefore hence since thus let'.split())

_UNIT_PATTERN = {
    'time': r'\b(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?|secs?|mins?|hrs?)\b',
    'length': r'\b(?:meters?|metres?|centimeters?|centimetres?|kilometers?|kilometres?|miles?|feet|foot|inches?|yards?|cm|km)\b',
    'mass': r'\b(?:ounces?|pounds?|grams?|kilograms?|lbs?|kg|tons?)\b',
    'volume': r'\b(?:liters?|litres?|milliliters?|gallons?|quarts?|pints?|cups?)\b',
    'money': r'\b(?:dollars?|cents?|euros?|pesetas?|cost|price|money|profit|paid|pay|earn|spend)\b|(?<!\$)\$(?!\$)(?=\d)|[€£]',
    'area': r'\b(?:area|square\s+(?:feet|foot|inches?|meters?|metres?|cm))\b|\^\s*\{?2\}?',
    'cubic': r'\b(?:volume|cubic)\b|\^\s*\{?3\}?',
    'percent': r'%|\b(?:percent|percentage)\b',
}

_RELATIONS = {
    'total': r'\b(?:total|altogether|combined|together|everything|in all)\b',
    'per_item': r'\b(?:each|every|per|apiece)\b',
    'remainder': r'\b(?:left|remaining|remainder|rest|still|uncovered|absent)\b',
    'difference': r'\b(?:difference|more than|less than|fewer than|compared|behind|tie|catch up)\b',
    'increase': r'\b(?:increase\w*|more|gain\w*|add\w*|receive\w*|earn\w*|raise)\b',
    'decrease': r'\b(?:decrease\w*|less|fewer|lose|lost|spend\w*|spent|give|gave|remove\w*|subtract\w*)\b',
    'initial': r'\b(?:initial\w*|original\w*|already|start\w*|first|before)\b',
    'additional': r'\b(?:another|additional|more|extra|next|second|after)\b',
    'profit': r'\b(?:profit|gain|net|savings?|saved|change)\b',
    'ratio': r'\b(?:twice|double|triple|times|half|quarter|ratio|fraction)\b',
}

_BASE_NAMES = [
    'q_quantity_mentions', 'q_distinct_quantities', 'q_repeated_quantity_fraction',
    'q_quantity_entity_pairs', 'relaxed_steps', 'strict_steps_fraction',
    'q_operand_value_recall', 'q_operand_mention_recall', 'q_operand_entity_recall',
    'q_final_value_recall', 'q_final_mention_recall', 'q_final_entity_recall',
    'q_restatement_only_fraction', 'calculation_context_recall',
    'final_chain_step_count', 'final_chain_step_fraction',
    'unused_result_count', 'unused_result_fraction', 'unused_result_q_context_recall',
    'answer_has_computational_support', 'answer_is_last_result',
    'answer_is_nonfinal_intermediate', 'final_direct_operand_count',
    'final_add_count', 'final_subtract_count', 'final_multiply_count', 'final_divide_count',
    'q_repeat_minus_operand_recall', 'target_answer_context_recall',
    'target_final_calculation_recall', 'target_other_result_recall',
    'target_answer_vs_other_result', 'target_tail_recall',
    'target_named_entity_recall', 'q_named_entity_tail_recall',
    'answer_context_unit_recall', 'tail_unit_recall', 'final_calculation_unit_recall',
    'answer_context_unrequested_dimension_count', 'q_unit_dimension_count',
    'area_target_cubic_conclusion', 'cubic_target_area_conclusion',
    'unit_conversion_count', 'unit_conversion_errors', 'unit_conversion_error_fraction',
    'last_boxed_exists', 'last_boxed_matches_official', 'last_boxed_conflicts_official',
    'final_span_matches_official', 'final_span_conflicts_official',
    'last_result_sign_conflict', 'hash_marker_before_last_boxed',
]

_REL_NAMES = [f'{name}_{suffix}' for name in _RELATIONS
              for suffix in ('target_present', 'answer_context', 'final_operation_compatibility')]

_PEER_BASE = ('q_final_value_recall', 'q_final_mention_recall', 'q_final_entity_recall',
              'unused_result_fraction', 'target_answer_context_recall',
              'last_boxed_conflicts_official', 'unit_conversion_errors')

NAMES = ['sem4_' + name for name in _BASE_NAMES + _REL_NAMES]

NAMES += ['sem4_' + name + '_vs_peers' for name in _PEER_BASE]

def _key(value):
    return round(float(value), 7)

def _number(value):
    try:
        out = float(Fraction(value.replace(',', '')))
        return _key(out) if math.isfinite(out) and abs(out) < 1e12 else None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None

def _context_words(text):
    return set(_words(text)) - _CONTEXT_STOP - set(WORD_NUMBERS)

def _mentions(text, with_words=True):
    """Ordered numerical mentions with local lexical context; no deduplication."""
    result = []
    for match in _TOKEN.finditer(text):
        token = match.group()
        value = WORD_NUMBERS.get(token.lower()) if with_words and token.isalpha() else _number(token)
        if value is None:
            continue
        # Ordinal labels usually identify an item, rather than its magnitude.
        if not token.isalpha() and re.match(r'(?:st|nd|rd|th)\b', text[match.end():]):
            continue
        context = _context_words(text[max(0, match.start()-32):match.end()+48])
        result.append((_key(value), context, match.start(), match.end()))
    return result

def _target(question):
    starts = list(re.finditer(r'\b(?:how|what|calculate|determine)\b', question, flags=re.I))
    if starts:
        return question[starts[-1].start():]
    return re.split(r'(?<=[.!?])\s+', question.strip())[-1]

def _recall(required, observed):
    return len(required & observed) / max(1, len(required))

def _units(text):
    return {name for name, pattern in _UNIT_PATTERN.items() if re.search(pattern, text, flags=re.I)}

def _names(text):
    excluded = {'A', 'An', 'The', 'If', 'How', 'What', 'When', 'After', 'Before', 'Each', 'Every', 'They', 'He', 'She', 'His', 'Her', 'It', 'In', 'On', 'At', 'For', 'To', 'This', 'That', 'So', 'Then', 'With', 'There', 'During', 'Since'}
    return {m.group().lower() for m in re.finditer(r'\b[A-Z][a-z]+\b', text) if m.group() not in excluded}

def _relaxed_steps(text):
    """Numeric calculation graph; removing unit annotations is soft evidence.

    Retain algebraic words and variables as barriers. Remove only LaTeX text
    wrappers, parenthetical word annotations, and a fixed unit vocabulary.
    The resulting equalities are NOT used to certify arithmetic correctness.
    """
    normalized = _normalize_math(text).replace('TEXT', ' ')
    normalized = normalized.replace('\\ ', ' ').replace('\\;', ' ')
    normalized = re.sub(r'\([A-Za-z][A-Za-z\s/-]*\)', ' ', normalized)
    normalized = re.sub(r'(?<=\d)\s+(?:hours?|minutes?|seconds?|days?|weeks?|months?|years?|dollars?|cents?|euros?|pesetas?|pounds?|ounces?|grams?|kilograms?|feet|inches?|miles?|meters?|liters?|litres?|people|students?|rows?|items?|pieces?)(?:\s*/\s*[A-Za-z]+)?\b', ' ', normalized, flags=re.I)
    # Normalization preserves line breaks except LaTeX wrappers containing them;
    # contextual labels use a rolling original-text window rather than alignment.
    steps = []
    latest = {}
    cursor = 0
    for line in normalized.splitlines():
        # Nearby prose often supplies the entity of an otherwise bare equation.
        context = normalized[max(0, cursor-140):cursor+len(line)+80]
        cursor += len(line)+1
        pieces = line.split('=')
        for left, right in zip(pieces, pieces[1:]):
            lm, rm = _LEFT.search(left), _RIGHT.match(right)
            if not lm or not rm:
                continue
            if lm.start() and (left[lm.start()-1].isalnum() or left[lm.start()-1] in '_{}^'):
                continue
            tail = right[rm.end():]
            if tail.startswith(('_', '^', '%')) or (tail and tail[0].isalnum() and not right[rm.end()-1].isspace()):
                continue
            lexpr, rexpr = lm.group(1).strip(), rm.group(1).strip()
            lv, rv = _literal_value(lexpr), _literal_value(rexpr)
            if lv is None or rv is None:
                continue
            # Without an operator this is a restatement, not a computation.
            if not re.search(r'\d\s*[+*/-]|[)]\s*[+*/-]', lexpr):
                continue
            try:
                tree = ast.parse(lexpr.strip().rstrip('.').strip(),mode='eval')
                operands = [_key(node.value) for node in ast.walk(tree)
                            if isinstance(node,ast.Constant) and type(node.value) in (int,float)]
            except (SyntaxError,ValueError,OverflowError):
                continue
            if not operands:
                continue
            value = _key(rv)
            predecessors = [latest.get(v) for v in operands]
            operators = [lexpr.count(op) for op in '+-*/']
            steps.append({'value': value, 'operands': operands, 'pred': predecessors,
                          'fraction_values': [_number(m.group()) for m in _NUMBER.finditer(lexpr) if '/' in m.group()],
                          'words': _context_words(context), 'context': context, 'ops': operators})
            latest[value] = len(steps)-1
    return steps

def _graph_features(steps, answer, question_mentions):
    qcounts = Counter(v for v, *_ in question_mentions)
    qvalues = set(qcounts)
    operands = Counter(v for s in steps for v in s['operands'])
    operands.update(v for s in steps for v in s['fraction_values'] if v in qvalues)
    all_words = set().union(*(s['words'] for s in steps)) if steps else set()
    final_id = next((k for k in range(len(steps)-1, -1, -1) if steps[k]['value'] == answer), None)
    active, leaves = set(), Counter()
    def visit(k):
        if k in active:
            return
        active.add(k)
        for value, pred in zip(steps[k]['operands'], steps[k]['pred']):
            if pred is None:
                leaves[value] += 1
            else:
                visit(pred)
        leaves.update(v for v in steps[k]['fraction_values'] if v in qvalues)
    if final_id is not None:
        visit(final_id)
    final_words = set().union(*(steps[k]['words'] for k in active)) if active else set()
    unused = [s for k,s in enumerate(steps) if k not in active]
    unused_words = set().union(*(s['words'] for s in unused)) if unused else set()
    def mention_recall(counts):
        return sum(min(n, counts[v]) for v,n in qcounts.items())/max(1,sum(qcounts.values()))
    def entity_recall(counts, words):
        return sum(v in counts and bool(ctx & words) for v,ctx,*_ in question_mentions)/max(1,len(question_mentions))
    return qcounts, operands, leaves, active, final_id, unused_words, [
        _recall(qvalues,set(operands)), mention_recall(operands), entity_recall(operands,all_words),
        _recall(qvalues,set(leaves)), mention_recall(leaves), entity_recall(leaves,final_words)]

_CONVERSION = {'second':('time',1.),'minute':('time',60.),'hour':('time',3600.),'day':('time',86400.),
    'week':('time',604800.),'ounce':('mass',1.),'pound':('mass',16.),'lb':('mass',16.),
    'gram':('metric_mass',1.),'kilogram':('metric_mass',1000.),'kg':('metric_mass',1000.),
    'inch':('length',1.),'inches':('length',1.),'foot':('length',12.),'feet':('length',12.),
    'yard':('length',36.),'mile':('length',63360.),'centimeter':('metric_length',1.),
    'meter':('metric_length',100.),'kilometer':('metric_length',100000.),'cm':('metric_length',1.),
    'dollar':('money',100.),'cent':('money',1.),'liter':('metric_volume',1000.),'milliliter':('metric_volume',1.)}

def _conversion_stats(text):
    # Local explicit statements only, e.g. "8 ounces = 1 pound". Unknown units,
    # currencies, variable expressions and cross-system conversions are skipped.
    clean = _FORMATTING.sub(r' \1 ', text).replace('**','').replace('$','').replace('\\,',' ')
    pattern = r'([-+]?(?:\d+(?:\.\d+)?|\.\d+))\s+([a-z]+)\s*=\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+))\s+([a-z]+)\b'
    count = errors = 0
    for match in re.finditer(pattern,clean.lower()):
        a,u,b,v = match.groups()
        prefix = clean[clean.rfind('\n',0,match.start())+1:match.start()]
        # A suffix such as "7 days = 56 hours" inside "8 hours/day * 7
        # days = 56 hours" is not an asserted standalone conversion.
        if _NUMBER.search(prefix) or re.search(r'\\(?:times|cdot|frac|div)\b|[×*/]',prefix):
            continue
        u = u if u in _CONVERSION else u.rstrip('s')
        v = v if v in _CONVERSION else v.rstrip('s')
        if u not in _CONVERSION or v not in _CONVERSION or _CONVERSION[u][0] != _CONVERSION[v][0]:
            continue
        count += 1
        left,right = float(a)*_CONVERSION[u][1],float(b)*_CONVERSION[v][1]
        errors += abs(left-right)>1e-6*max(1.,abs(left),abs(right))
    return count,errors

def extract_semantic_features(bank):
    """Return float32[N,4,89] and stable names, including an empty-bank result."""
    result = np.zeros((len(bank),4,len(NAMES)), dtype=np.float32)
    for i,row in enumerate(bank):
        if len(row['candidates']) != 4:
            raise ValueError('Expected exactly four candidates per question')
        question = str(row['question']); target = _target(question)
        qmentions = _mentions(question); qcounts = Counter(v for v,*_ in qmentions)
        qwords, target_words = _context_words(question),_context_words(target)
        qnames,target_names = _names(question),_names(target)
        q_units,target_units = _units(question),_units(target)
        for j,candidate in enumerate(row['candidates']):
            text = str(candidate['text']); parsed = parse_answer(text)
            answer = _number(parsed) if parsed is not None else None
            mentions = _mentions(text,with_words=False)
            answer_mentions = [m for m in mentions if m[0] == answer]
            answer_context = ' '.join(text[max(0,start-95):end+65] for _,_,start,end in answer_mentions[-5:])
            answer_words = _context_words(answer_context)
            tail = text[max(0,len(text)-max(220,len(text)//3)):]
            steps = _relaxed_steps(text)
            _, strict = literal_checks(text)
            counts, operands, leaves, active, final_id, unused_words, graph = _graph_features(steps,answer,qmentions)
            final_step = steps[final_id] if final_id is not None else None
            final_context = final_step['context'] if final_step else ''
            final_words = final_step['words'] if final_step else set()
            ops = final_step['ops'] if final_step else [0]*4
            other_words = set().union(*(s['words'] for s in steps if s['value'] != answer)) if steps else set()
            all_calc_words = set().union(*(s['words'] for s in steps)) if steps else set()
            textual_values = {v for v,*_ in mentions}
            conversion_count,conversion_errors = _conversion_stats(text)
            boxed_matches = list(re.finditer(r'\\boxed\{([^{}]+)\}',text))
            boxed = parse_answer(boxed_matches[-1].group()) if boxed_matches else None
            markers = list(re.finditer(r'final\s+answer\s*:?',text,flags=re.I))
            span = text[markers[-1].end():] if markers else ''
            span_mentions = _mentions(span,with_words=False)
            final_span = span_mentions[0][0] if span_mentions else None
            target_recall = _recall(target_words,answer_words)
            other_recall = _recall(target_words,other_words)
            values = [len(qmentions),len(qcounts),1-len(qcounts)/max(1,len(qmentions)),
                sum(len(ctx) for _,ctx,*_ in qmentions),len(steps),len(strict)/max(1,len(steps)),
                *graph,len((set(qcounts)&textual_values)-set(operands))/max(1,len(qcounts)),
                _recall(qwords,all_calc_words),len(active),len(active)/max(1,len(steps)),
                len(steps)-len(active),(len(steps)-len(active))/max(1,len(steps)),
                _recall(qwords,unused_words),final_id is not None,
                bool(steps) and answer==steps[-1]['value'],
                any(s['value']==answer for s in steps[:-1]),
                len(final_step['operands']) if final_step else 0,*ops,
                (len(qmentions)-len(qcounts))/max(1,len(qmentions))*(1-graph[1]),
                target_recall,_recall(target_words,final_words),other_recall,
                target_recall-other_recall,_recall(target_words,_context_words(tail)),
                _recall(target_names,set(re.findall(r'[a-z]+',answer_context.lower()))),
                _recall(qnames,set(re.findall(r'[a-z]+',tail.lower()))),
                _recall(target_units,_units(answer_context)),_recall(target_units,_units(tail)),
                _recall(target_units,_units(final_context)),len(_units(answer_context)-target_units),len(q_units),
                'area' in target_units and 'cubic' in _units(tail),
                'cubic' in target_units and 'area' in _units(tail),
                conversion_count,conversion_errors,conversion_errors/max(1,conversion_count),
                boxed is not None,boxed is not None and boxed==parsed,boxed is not None and boxed!=parsed,
                final_span is not None and final_span==answer,final_span is not None and final_span!=answer,
                bool(steps) and answer is not None and steps[-1]['value']==-answer and answer!=0,
                bool(boxed_matches) and '####' in text and text.rfind('####')<boxed_matches[-1].start()]
            for name,pattern in _RELATIONS.items():
                present = bool(re.search(pattern,target,flags=re.I))
                in_answer = bool(re.search(pattern,answer_context,flags=re.I))
                operation = {'total':ops[0], 'per_item':ops[2]+ops[3], 'remainder':ops[1],
                    'difference':ops[1], 'increase':ops[0], 'decrease':ops[1],
                    'initial':ops[0]+ops[1], 'additional':ops[0], 'profit':ops[1],
                    'ratio':ops[2]+ops[3]}[name]
                values.extend([present,present*in_answer,present*math.log1p(operation)])
            result[i,j,:len(values)] = values
        for offset,name in enumerate(_PEER_BASE):
            index = NAMES.index('sem4_'+name)
            result[i,:,-len(_PEER_BASE)+offset] = result[i,:,index]-result[i,:,index].mean()
    if not np.isfinite(result).all():
        raise ValueError('Semantic feature extraction produced nonfinite values')
    return result,list(NAMES)
