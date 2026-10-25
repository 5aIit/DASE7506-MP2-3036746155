"""Finite-scope rule regressions using arbitrary synthetic inputs only."""
from pathlib import Path
import sys
import unittest

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from selector import unit_rules, finance_rules, inventory_rules
from selector.arithmetic_rules import arithmetic_text, routes
import submission



class TargetUnitBehaviorTests(unittest.TestCase):
    def bank(self,question,texts):
        return [{'question':question,'candidates':[{'text':v} for v in texts]}]

    def test_area_and_volume_units(self):
        bank=self.bank('How many square inches are left?',[
            r'Volume = 12 \text{ in}^3. \boxed{12}',
            r'Area = 9 square inches. \boxed{9}',r'\boxed{4}',r'\boxed{3}'])
        x=unit_rules.features(bank)
        self.assertEqual(x[0,0,0],1);self.assertEqual(x[0,1,0],0)
        self.assertEqual(unit_rules.unit('in',2),('length',2))
        self.assertIsNone(unit_rules.unit('in'))

    def test_requested_count_and_person_synonyms(self):
        bank=self.bank('How many students can receive a book?',[
            r'15 books remain. \boxed{15}',r'3 people can be served. \boxed{3}',
            r'4 children receive books. \boxed{4}',r'\boxed{5}'])
        x=unit_rules.features(bank)
        np.testing.assert_array_equal(x[0,:,1],[1,0,0,0])

    def test_same_answer_unknown_unit_abstains(self):
        bank=self.bank('How many students can receive a book?',[
            r'15 books remain. \boxed{15}',r'\boxed{15}',r'\boxed{4}',r'\boxed{5}'])
        np.testing.assert_array_equal(unit_rules.features(bank),0)

    def test_missing_target_or_result_unit_abstains(self):
        bank=self.bank('Find the required amount.',[r'9 books. \boxed{9}',r'\boxed{3}',r'\boxed{4}',r'\boxed{5}'])
        np.testing.assert_array_equal(unit_rules.features(bank),0)


class ExplicitConstraintTests(unittest.TestCase):
    def test_percentage_of_and_off(self):
        q='A coat was bought at 30% of the original price. The original price was $900. How much was paid?'
        self.assertEqual(finance_rules.price_constraint(q),'270')
        self.assertEqual(finance_rules.price_constraint(q.replace('30% of','30% off')),'630')
        self.assertIsNone(finance_rules.price_constraint(q.replace('A coat','2 coats')))

    def test_production_profit_with_arbitrary_items(self):
        q='Each hat needs 3 rolls of cloth. Each roll of cloth costs $8. How much profit is made on all 24 hats if the merchant sells each hat for $30?'
        self.assertEqual(finance_rules.profit_constraint(q),'144')
        self.assertIsNone(finance_rules.profit_constraint(q+' Shipping costs $5.'))
        self.assertIsNone(finance_rules.profit_constraint(q.replace('profit','revenue')))
        self.assertIsNone(finance_rules.profit_constraint(q.replace('Each roll of cloth costs $8.','')))

    def test_missing_supplied_solution_abstains(self):
        q='A coat was bought at 30% of the original price. The original price was $900. How much was paid?'
        b=[{'question':q,'candidates':[{'text':'#### '+str(n)} for n in (1,2,3,4)]}]
        np.testing.assert_array_equal(finance_rules.features(b),0)

    def test_unit_annotated_fraction_repair(self):
        text=r'\frac{0.6}{9} = \frac{2}{15} \text{ hours}'+'\n'+r'\frac{2}{15} \text{ hours} * 60 \text{minutes/hour} = 8 \text{minutes}'+'\n'+r'\boxed{8}'
        b=[{'question':'Use a gap of 0.6 and relative speeds 24 and 15.',
            'candidates':[{'text':text},{'text':'#### 4'},{'text':'#### 3'},{'text':'#### 6'}]}]
        self.assertEqual(routes(b),[{'8':'4'}])

    def test_unit_conversion_and_algebra_abstain(self):
        for text in (r'1 \text{hour} = 60 \text{minutes}. \boxed{60}',
                     r'2 + 3 = 4x. \boxed{4}',r'2 + 3 = 5^2. \boxed{5}'):
            self.assertNotIn('=',arithmetic_text(text))


class InventoryConservationTests(unittest.TestCase):
    def question(self,n,b,m,s,factor='doubles'):
        return f'Nora buys {n} widgets that cost ${b} each. She sells {m} widgets for ${s} each. The remaining widgets {factor} in value. How much profit is made?'

    def test_arbitrary_quantities_and_prices(self):
        for n,b,m,s in ((18,5,4,8),(12,2,7,3),(30,7,15,6),(9,4,0,8),(8,3,8,5)):
            self.assertEqual(inventory_rules.constraint(self.question(n,b,m,s)),str(m*s+(n-m)*2*b-n*b))
            self.assertEqual(inventory_rules.constraint(self.question(n,b,m,s,'triples')),str(m*s+(n-m)*3*b-n*b))

    def test_additional_cost_and_invalid_inventory_abstain(self):
        self.assertIsNone(inventory_rules.constraint(self.question(18,5,4,8)+' A fee of $2 is charged.'))
        self.assertIsNone(inventory_rules.constraint(self.question(18,5,19,8)))
        self.assertIsNone(inventory_rules.constraint(self.question(18,5,4,8).replace('4 widgets','4 paintings')))
        self.assertIsNone(inventory_rules.constraint(self.question(18,5,4,8).replace('doubles','changes')))

    def test_missing_supplied_answer_abstains(self):
        bank=[{'question':self.question(18,5,4,8),'candidates':[{'text':'#### '+str(i)} for i in (1,2,3,4)]}]
        np.testing.assert_array_equal(inventory_rules.features(bank),0)


class CurrencyRegressionTests(unittest.TestCase):
    def bank(self, question, answer):
        return [{'question': question, 'candidates': [{'text': '#### ' + str(x)} for x in (answer, 1, 2, 3)]}]

    def test_mixed_currency_inventory_abstains(self):
        q = 'Nora buys 18 widgets that cost $5 each. She sells 4 widgets for €8 each. The remaining widgets double in value. How much profit is made?'
        _, x = submission.monetary_features(self.bank(q, 82))
        np.testing.assert_array_equal(x, 0)
        _, same = submission.monetary_features(self.bank(q.replace('€', '$'), 82))
        np.testing.assert_array_equal(same, [[1, 0, 0, 0]])

    def test_mixed_currency_production_abstains(self):
        q = 'Each hat needs 3 rolls of cloth. Each roll of cloth costs £8. How much profit is made on all 24 hats if the merchant sells each hat for €30?'
        x, _ = submission.monetary_features(self.bank(q, 144))
        np.testing.assert_array_equal(x, 0)
        same, _ = submission.monetary_features(self.bank(q.replace('€', '£'), 144))
        self.assertEqual(same[0, 0, 1], 1)

    def test_currency_names_and_no_currency(self):
        self.assertTrue(submission.currency_is_unambiguous('Cost $5 dollars.'))
        self.assertFalse(submission.currency_is_unambiguous('Cost $5; sell for 8 euros.'))
        self.assertFalse(submission.currency_is_unambiguous('Cost $5; profit in GBP.'))
        self.assertFalse(submission.currency_is_unambiguous('There are 5 items.'))

    def test_no_supported_answer_still_abstains(self):
        q = 'Each hat needs 3 rolls of cloth. Each roll of cloth costs £8. How much profit is made on all 24 hats if the merchant sells each hat for £30?'
        x, _ = submission.monetary_features(self.bank(q, 999))
        np.testing.assert_array_equal(x, 0)


if __name__ == "__main__":
    unittest.main()
