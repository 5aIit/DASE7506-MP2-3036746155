"""Currency rules for the frozen selector."""

import re

import numpy as np

import selector.finance_rules as finance_rules

import selector.inventory_rules as inventory_rules

def currency_is_unambiguous(question):
    """Accept one supported currency; never invent an exchange rate."""
    markers = set(re.findall(r"[$€£]", question))
    names = {
        "$": r"\b(?:usd|dollars?)\b",
        "€": r"\b(?:eur|euros?)\b",
        "£": r"\b(?:gbp|pounds?\s+sterling)\b",
    }
    for symbol, pattern in names.items():
        if re.search(pattern, question, re.IGNORECASE):
            markers.add(symbol)
    return len(markers) == 1

def monetary_features(bank):
    allowed = np.array([currency_is_unambiguous(row['question']) for row in bank])
    production = finance_rules.features(bank)
    inventory = inventory_rules.features(bank)
    production *= allowed[:, None, None]
    inventory *= allowed[:, None]
    return production, inventory
