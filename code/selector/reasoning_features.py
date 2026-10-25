"""Reasoning features for the frozen selector."""

import ast

import math

import re

from fractions import Fraction

import numpy as np

from common import parse_answer

from selector.features import _normalize_math, _literal_value

WORD_NUMBERS = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split(),range(21)))

WORD_NUMBERS.update({'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,'eighty':80,'ninety':90,'hundred':100,'thousand':1000,'half':.5,'quarter':.25,'dozen':12,'twice':2,'double':2,'triple':3})

NUMBER = re.compile(r'(?<![\w.])\d[\d,]*(?:\.\d+)?(?:/\d+)?')

LEFT = re.compile(r'([\d\s.()+*/\-]+)$')

RIGHT = re.compile(r'^([\d\s.()+*/\-]+)')

TRIGGERS = {
    'repeat':r'\b(?:each|every|per|times|twice|double|triple|both|coats?|layers?|trips?)\b',
    'remaining':r'\b(?:left|remain\w*|rest|after|spent|sold|lost|gave|used|less|fewer)\b',
    'comparison':r'\b(?:more|less|fewer|than|difference|greater|increas\w*|decreas\w*)\b',
    'rate':r'\b(?:per|each|every|average|rate|speed|hour|minute|second)\b',
    'group':r'\b(?:pairs?|dozens?|groups?|teams?|boxes|bags?|batches|packs?)\b',
    'percentage':r'%|\b(?:percent|percentage|discount|tax|interest|profit)\b',
    'fraction':r'\b(?:half|quarter|third|fourth|fifth|fraction)\b|\d/\d',
    'total':r'\b(?:total|altogether|combined|sum|together|in all)\b',
}

NAMES = [
    'trace_q_number_count','trace_literal_checks','trace_correct_checks','trace_error_checks',
    'trace_error_fraction','trace_question_operand_recall','trace_question_missing_fraction',
    'trace_novel_operand_fraction','trace_final_provenance_recall','trace_final_provenance_size',
    'trace_answer_is_result','trace_answer_is_operand','trace_answer_is_question_number',
    'trace_reused_intermediate_count','trace_unsupported_equalities','trace_unit_barriers',
    'trace_final_result_matches','trace_log_max_error','trace_implicit_two_used',
    'trace_implicit_twelve_used','trace_sixty_used','trace_hundred_used',
    'trace_unique_results','trace_question_numbers_only_in_prose','trace_coverage_vs_peers',
    'trace_provenance_vs_peers','trace_errors_vs_peers',
] + [f'trace_{name}_{suffix}' for name in TRIGGERS for suffix in ('question','body','coverage','operation_interaction')]

def key(value):
    return round(float(value),7)

def quantities(text):
    values=set()
    for m in NUMBER.finditer(text):
        try:
            v=float(Fraction(m.group().replace(',','')))
            if math.isfinite(v) and abs(v)<1e12: values.add(key(v))
        except (ValueError,ZeroDivisionError,OverflowError): pass
    for word in re.findall(r'[a-z]+',text.lower()):
        if word in WORD_NUMBERS: values.add(key(WORD_NUMBERS[word]))
    return values

def constants(expression):
    tree=ast.parse(expression.strip().rstrip('.').strip(),mode='eval')
    return {key(n.value) for n in ast.walk(tree) if isinstance(n,ast.Constant) and type(n.value) in (int,float)}

def literal_checks(text):
    normalized=_normalize_math(text)
    checks=[]
    for line in normalized.splitlines():
        sections=line.split('=')
        for left,right in zip(sections,sections[1:]):
            lm,rm=LEFT.search(left),RIGHT.match(right)
            if not lm or not rm: continue
            # Do not take a numeric suffix from S_1, x2, {2}, or indexed algebra.
            if lm.start() and (left[lm.start()-1].isalnum() or left[lm.start()-1] in '_{}^'):
                continue
            tail=right[rm.end():]
            if tail.startswith(('_','^','%')) or re.match(r'\s*[xyz]\b',tail):continue
            if tail and tail[0].isalnum() and not right[rm.end()-1].isspace():continue
            le,rex=lm.group(1).strip(),rm.group(1).strip()
            lv,rv=_literal_value(le),_literal_value(rex)
            if lv is None or rv is None:continue
            if re.search(r'%\s*$',left[:lm.start()]):continue
            try: operands=constants(le)
            except (ValueError,SyntaxError,OverflowError):continue
            error=abs(lv-rv)/max(1.,abs(lv),abs(rv))
            checks.append((operands,key(rv),error))
    return normalized,checks

def extract_reasoning_features(bank):
    result=np.zeros((len(bank),4,len(NAMES)),dtype=np.float32)
    for i,row in enumerate(bank):
        question=row['question'];q=quantities(question)
        qt={name:len(re.findall(pattern,question.lower())) for name,pattern in TRIGGERS.items()}
        for j,candidate in enumerate(row['candidates']):
            text=candidate['text'];normalized,checks=literal_checks(text)
            answer=parse_answer(text)
            a=key(Fraction(answer)) if answer is not None else None
            provenance={number:{number} for number in q}
            operands=set();values=[];reuse=0;errors=[]
            for used,value,error in checks:
                operands|=used;values.append(value);errors.append(error)
                sources=set()
                for number in used:
                    sources|=provenance.get(number,set())
                    reuse+=int(number in provenance and number not in q)
                if error<=1e-6:
                    provenance[value]=provenance.get(value,set())|sources
            covered=len(operands&q)/max(1,len(q))
            p=provenance.get(a,set()) if a is not None else set()
            err=sum(e>1e-6 for e in errors)
            vals=[len(q),len(checks),len(checks)-err,err,err/max(1,len(checks)),
                  covered,1-covered,len(operands-q)/max(1,len(operands)),len(p)/max(1,len(q)),len(p),
                  a in values,a in operands,a in q,reuse,max(0,text.count('=')-len(checks)),
                  normalized.count('TEXT'),bool(values) and a==values[-1],
                  math.log1p(max(errors,default=0)),2. in operands,12. in operands,60. in operands,
                  100. in operands,len(set(values)),len((q&quantities(text))-operands)/max(1,len(q)),
                  0.,0.,0.]
            # Use the latter half to reduce repeated question text dominating coverage.
            body=text[len(text)//2:].lower()
            operators={op:normalized.count(op) for op in '+-*/'}
            operation_signal={'repeat':operators['*'],'remaining':operators['-'],
                'comparison':operators['+']+operators['-'],'rate':operators['*']+operators['/'],
                'group':operators['*']+operators['/'],'percentage':operators['*']+operators['/'],
                'fraction':operators['*']+operators['/'],'total':operators['+']}
            for name,pattern in TRIGGERS.items():
                c=len(re.findall(pattern,body))
                vals.extend([qt[name],c,min(c,qt[name])/max(1,qt[name]),qt[name]*math.log1p(operation_signal[name])])
            result[i,j]=vals
        result[i,:,24]=result[i,:,5]-result[i,:,5].mean()
        result[i,:,25]=result[i,:,8]-result[i,:,8].mean()
        result[i,:,26]=result[i,:,3]-result[i,:,3].mean()
    return result,list(NAMES)
