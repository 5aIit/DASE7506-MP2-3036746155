"""Inventory rules for the frozen selector."""

import re

from collections import Counter

import numpy as np

from selector.binding_features import _plain, _unit

from selector.semantic_features import _target

from selector.finance_rules import NUM, value

from selector.features import answer_groups

def constraint(question):
    text=_plain(question)
    if not re.search(r'\bprofits?\b',_target(text)):return None
    if re.search(r'\b(?:tax|fee|fees|commission|discount|interest)\b|%',text):return None
    sentences=re.split(r'(?<=[.!?])\s+',text)
    purchases=[];sales=[];remainders=[]
    for sentence in sentences:
        buy=re.search(r'\b(?:buy|buys|bought|purchase|purchases|purchased)\s+('+NUM+r')\s+([a-z]+)',sentence)
        sale=re.search(r'\b(?:sell|sells|sold)\s+('+NUM+r')\b',sentence)
        prices=list(re.finditer(r'[$€£]\s*('+NUM+r')',sentence))
        if buy and len(prices)==1 and re.search(r'\beach\b',sentence):
            purchases.append((value(buy[1]),_unit(buy[2]),value(prices[0][1])))
        if sale and len(prices)==1 and re.search(r'\beach\b',sentence):
            sales.append((value(sale[1]),value(prices[0][1]),sentence))
        if re.search(r'\b(?:rest|remaining|remainder|unsold)\b',sentence) and re.search(r'\bvalue\b',sentence):
            factors=[]
            if re.search(r'\b(?:double|doubles|doubled|doubling)\b',sentence):factors.append(2)
            if re.search(r'\b(?:triple|triples|tripled|tripling)\b',sentence):factors.append(3)
            if len(factors)==1:remainders.append(factors[0])
    if len(purchases)!=1 or len(sales)!=1 or len(remainders)!=1:return None
    count,item,buy_price=purchases[0];sold,sale_price,sale_sentence=sales[0]
    if item is None or item not in {_unit(w) for w in re.findall('[a-z]+',sale_sentence)}:return None
    if not (count>0 and 0<=sold<=count and buy_price>=0 and sale_price>=0):return None
    # Any unexplained quantity/price indicates an incomplete event model.
    numbers=[value(m[0]) for m in re.finditer(r'(?<![\w.])'+NUM,text)]
    if Counter(numbers)!=Counter((count,buy_price,sold,sale_price)):return None
    if len(re.findall(r'[$€£]\s*'+NUM,text))!=2:return None
    return str(sold*sale_price+(count-sold)*remainders[0]*buy_price-count*buy_price)

def features(bank):
    out=np.zeros((len(bank),4),float)
    for i,(row,aa) in enumerate(zip(bank,answer_groups(bank))):
        try:answer=constraint(row['question'])
        except (ValueError,ZeroDivisionError,OverflowError):answer=None
        if answer is not None and answer in aa:out[i]=[v==answer for v in aa]
    return out
