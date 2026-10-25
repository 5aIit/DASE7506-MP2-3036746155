"""Binding features for the frozen selector."""

from fractions import Fraction

import math

import re

import numpy as np

from common import parse_answer

from selector.reasoning_features import WORD_NUMBERS

_NUM = r'(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:/\d+)?'

_FRAC = re.compile(r'\\(?:frac|dfrac|tfrac)\{('+_NUM+r')\}\{('+_NUM+r')\}')

_WRAPPER = re.compile(r'\\(?:text|mathrm|mbox|textbf|mathbf)\{([^{}]*)\}')

_UNIT_STOP = set('a an the of for every each per at in on to from by with his her their its this that these those is are was were he she it they and then now has have had need needs get gets total number amount left remaining more less as much many if since than when while because can will would shall should could'.split())

_INDEFINITE_RATE_UNITS=set('second minute hour day week month year decade season episode shift night weeknight weekend gram kilogram ounce pound meter metre kilometer kilometre mile inch foot yard liter litre gallon quart pint cup box carton bottle pack package bag batch set load dozen person student customer patient animal car truck bus train trip'.split())

_CANON = {'dollars':'dollar','cents':'cent','euros':'euro','pesetas':'peseta',
    'hours':'hour','hrs':'hour','hr':'hour','minutes':'minute','mins':'minute','min':'minute',
    'seconds':'second','secs':'second','sec':'second','feet':'foot','inches':'inch',
    'people':'person','persons':'person','children':'child','men':'man','women':'woman',
    'batches':'batch','boxes':'box','classes':'class','cookies':'cookie','bodies':'body'}

def _value(text):
    try:
        v=float(Fraction(str(text).replace(',','')))
        return round(v,8) if math.isfinite(v) and abs(v)<1e12 else None
    except (ValueError,ZeroDivisionError,OverflowError):
        return None

def _plain(text):
    text=text.lower().replace('**','').replace('\\$','$').replace('\\,',' ')
    text=_FRAC.sub(lambda m:str(float(Fraction(m[1].replace(',',''))/Fraction(m[2].replace(',',''))))
                   if _value(m[2]) not in (0,None) else m[0],text)
    text=_WRAPPER.sub(r' \1 ',text)
    # Ordinary count words only; "half" is retained to avoid rewriting the
    # relational phrase "half as many" as a standalone quantitative assertion.
    words={k:v for k,v in WORD_NUMBERS.items() if k not in ('half','quarter','twice','double','triple') and v<=20}
    text=re.sub(r'\b(?:'+ '|'.join(words)+r')\b',lambda m:str(words[m[0]]),text)
    return text

def _unit(phrase):
    # "loads of laundry" -> load; "detergent pod" -> pod.
    phrase=re.split(r'\b(?:of|for|per|and|on|in|at|as|which)\b',phrase)[0]
    words=[w for w in re.findall('[a-z]+',phrase) if w not in _UNIT_STOP]
    if not words:return None
    word=words[-1]
    if word in _CANON:return _CANON[word]
    if len(word)>3 and word.endswith('s') and not word.endswith('ss'):
        word=word[:-1]
    return word

def explicit_rates(text):
    """Extract unique (value, numerator label, denominator label) assertions.

    Amounts are NOT converted between calendar or physical units: an hourly
    wage and a daily wage need not assume 24 working hours per day.
    """
    text=_plain(text);found=[]
    def add(value,numunit,denunit,source):
        value=_value(value);numunit=_unit(numunit);denunit=_unit(denunit)
        if value is not None and numunit and denunit:
            item=(value,numunit,denunit)
            if item not in [x[:3] for x in found]:found.append((*item,source))
    # Currency is often adjacent to its value, unlike count units.
    currency=r'(?P<symbol>[$€£])\s*(?P<value>'+_NUM+r')\s*(?:for\s+)?(?:per|a|an|each|every)\s+(?P<den>[a-z]+)\b'
    for m in re.finditer(currency,text):
        add(m['value'],{'$':'dollar','€':'euro','£':'poundsterling'}[m['symbol']],m['den'],m[0])
    explicit=(r'(?<![\w.])(?P<value>'+_NUM+r')[ \t]+'
        r'(?P<num>[a-z]+(?:[ \t]+[a-z]+){0,3}?)[ \t]+'
        r'(?P<link>per|for[ \t]+(?:each|every)|each|every|a|an)[ \t]+(?P<den>[a-z]+)\b')
    for m in re.finditer(explicit,text):
        if re.search(r'\b(?:if|since|when|while|because|to|than|he|she|they)\b',m['num']):continue
        if m['link'] in ('a','an') and _unit(m['den']) not in _INDEFINITE_RATE_UNITS:continue
        add(m['value'],m['num'],m['den'],m[0])
    for m in re.finditer(r'(?<![\w.])(?P<value>'+_NUM+r')\s+(?P<num>[a-z]+)\s*/\s*(?P<den>[a-z]+)\b',text):
        add(m['value'],m['num'],m['den'],m[0])
    # An article is an explicit singular count in these tightly bounded forms.
    article=r'\b(?:a|an)\s+(?P<num>[a-z]+(?:\s+[a-z]+){0,2}?)\s+(?:for\s+)?(?:each|every|per)\s+(?P<den>[a-z]+)\b'
    for m in re.finditer(article,text):add(1,m['num'],m['den'],m[0])
    per_entity=(r'\b(?:each|every)\s+(?P<den>[a-z]+)\s+'
        r'(?:has|holds|contains|uses|needs|takes|makes|produces|costs|earns|weighs)\s+'
        r'(?P<symbol>[$€£])?\s*(?P<value>'+_NUM+r')[ \t]*(?P<num>[a-z]+(?:[ \t]+[a-z]+){0,2})?')
    for m in re.finditer(per_entity,text):
        num={'$':'dollar','€':'euro','£':'poundsterling'}.get(m['symbol'],m['num'])
        if num:add(m['value'],num,m['den'],m[0])
    return found

def last_boxed_answer(text):
    matches=list(re.finditer(r'\\boxed\{([^{}]+)\}',text))
    return parse_answer(matches[-1][0]) if matches else None

_BASE=[
    'question_rate_count','candidate_rate_count','question_rate_type_count',
    'exact_rate_matches','exact_rate_recall','same_type_compared_count',
    'same_type_value_conflicts','same_type_value_conflict_fraction',
    'same_value_denominator_conflicts','same_value_numerator_conflicts',
    'unsupported_rate_type_count','rate_value_max_log_ratio',
    'same_type_candidate_self_conflicts','matched_type_value_consistency',
    'question_singleton_rate_conflicts','question_money_rate_conflicts',
    'official_boxed_conflict','boxed_destination_exists','boxed_destination_group_size',
    'incoming_boxed_support_count','incoming_boxed_support_fraction',
    'incoming_conflicting_boxed_support_count','incoming_conflicting_boxed_source_groups',
    'outgoing_boxed_support_count','boxed_consensus_for_official_answer',
]

_PEER=('exact_rate_recall','same_type_value_conflicts','same_value_denominator_conflicts',
       'same_value_numerator_conflicts','incoming_conflicting_boxed_support_count')

NAMES=['bind7_'+n for n in _BASE]+['bind7_'+n+'_vs_peers' for n in _PEER]

def extract_binding_features(bank):
    """Return finite float32[N,4,30] and the stable, label-free schema."""
    out=np.zeros((len(bank),4,len(NAMES)),dtype=np.float32)
    for i,row in enumerate(bank):
        if len(row['candidates'])!=4:raise ValueError('Expected four candidates')
        question_rates=explicit_rates(str(row['question']))
        qr=[x[:3] for x in question_rates]
        types={(u,v) for _,u,v in qr}
        answers=[parse_answer(c['text']) for c in row['candidates']]
        boxed=[last_boxed_answer(c['text']) for c in row['candidates']]
        for j,candidate in enumerate(row['candidates']):
            cr=[x[:3] for x in explicit_rates(candidate['text'])]
            matched=sum(item in qr for item in cr)
            compared=conflicts=den_conflicts=num_conflicts=single_conflicts=money_conflicts=0
            max_ratio=0.
            for value,num,den in cr:
                qsame=[qv for qv,qu,qd in qr if (qu,qd)==(num,den)]
                if qsame:
                    compared+=1
                    if value not in qsame:
                        conflicts+=1
                        single_conflicts+=1. in qsame
                        money_conflicts+=num in ('dollar','cent','euro','peseta','poundsterling')
                        ratios=[abs(math.log((abs(value)+1e-6)/(abs(v)+1e-6))) for v in qsame]
                        max_ratio=max(max_ratio,min(ratios))
                den_conflicts+=any(qv==value and qu==num and qd!=den for qv,qu,qd in qr) and (value,num,den) not in qr
                num_conflicts+=any(qv==value and qd==den and qu!=num for qv,qu,qd in qr) and (value,num,den) not in qr
            self_conflict=sum(len({v for v,nu,de in cr if (nu,de)==t})>1 for t in {(nu,de) for _,nu,de in cr})
            a=answers[j];box=boxed[j]
            incoming=[k for k,b in enumerate(boxed) if k!=j and a is not None and b==a]
            transfer=[k for k in incoming if answers[k]!=boxed[k]]
            destination=[k for k,b in enumerate(answers) if box is not None and b==box and k!=j]
            values=[len(qr),len(cr),len(types),matched,matched/max(1,len(qr)),compared,
                conflicts,conflicts/max(1,compared),den_conflicts,num_conflicts,
                sum((u,v) not in types for _,u,v in cr),max_ratio,self_conflict,
                (compared-conflicts)/max(1,compared),single_conflicts,money_conflicts,
                box is not None and box!=a,bool(destination),len(destination),len(incoming),len(incoming)/3,
                len(transfer),len({answers[k] for k in transfer}),
                len(destination) if box is not None and box!=a else 0,
                sum(b==a for b in boxed if b is not None) if a is not None else 0]
            out[i,j,:len(_BASE)]=values
        for k,name in enumerate(_PEER):
            v=out[i,:,_BASE.index(name)];out[i,:,len(_BASE)+k]=v-v.mean()
    if not np.isfinite(out).all():raise ValueError('Nonfinite binding features')
    return out,list(NAMES)

def boxed_transfer_features(bank,scores):
    """Expose donor quality to a learned reranker; never force a replacement.

    ``scores`` is [N,4] or [N,4,M]. For multiple models, each is centered/scaled
    within its question before averaging, so no single model scale dominates.
    The six output features are independent of the number of input models.
    """
    scores=np.asarray(scores,dtype=np.float32)
    if scores.ndim==2:scores=scores[...,None]
    if scores.ndim!=3 or scores.shape[:2]!=(len(bank),4) or not np.isfinite(scores).all():
        raise ValueError('Expected finite [N,4] or [N,4,M] scores')
    normalized=(scores-scores.mean(1,keepdims=True))/np.maximum(scores.std(1,keepdims=True),1e-5)
    mean=normalized.mean(-1)
    out=np.zeros((len(bank),4,6),dtype=np.float32)
    for i,row in enumerate(bank):
        answers=[parse_answer(c['text']) for c in row['candidates']]
        boxed=[last_boxed_answer(c['text']) for c in row['candidates']]
        for j,a in enumerate(answers):
            incoming=[k for k in range(4) if k!=j and a is not None and boxed[k]==a and boxed[k]!=answers[k]]
            outgoing=a is not None and boxed[j] is not None and boxed[j]!=a and boxed[j] in answers
            if incoming:
                donor=mean[i,incoming];donormax=float(donor.max())
                out[i,j,:4]=[1.,donormax,float(donor.mean()),donormax-float(mean[i,j])]
            out[i,j,4:]=[outgoing,mean[i,j] if outgoing else 0.]
    return out,['bind7_transfer_'+x for x in ('has_donor','max_donor_score','mean_donor_score',
        'donor_score_gap','has_destination','outgoing_score')]
