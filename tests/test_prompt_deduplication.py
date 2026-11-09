import unittest
from backend import prompt_builder

class PromptDeduplicationTests(unittest.TestCase):

    def test_distinct_weighted_groups_with_shared_prefix_remain_intact(self):
        text = '(a, b:1.2), (a, c:1.2)'
        self.assertEqual(prompt_builder.dedupe_prompt_terms(text), (text, False))

    def test_exact_duplicates_are_removed_as_whole_groups(self):
        text = '(a, b:1.2), (a, c:1.2), (a, b:1.2), portrait, portrait'
        self.assertEqual(prompt_builder.dedupe_prompt_terms(text), ('(a, b:1.2), (a, c:1.2), portrait', True))

    def test_nested_attention_and_scheduled_groups_remain_intact(self):
        text = '((a, b:1.2), [c, d]:1.1), ((a, b:1.2), [c, e]:1.1), [a, b:c, d:0.5]'
        self.assertEqual(prompt_builder.dedupe_prompt_terms(text), (text, False))

    def test_escaped_brackets_and_commas_do_not_break_group_boundaries(self):
        for text in ['(a \\(b\\), c:1.2), (a \\(b\\), d:1.2)', 'a\\, b, a\\, c']:
            with self.subTest(text=text):
                self.assertEqual(prompt_builder.dedupe_prompt_terms(text), (text, False))

    def test_unbalanced_or_mismatched_brackets_are_not_deduplicated(self):
        for text in ['a, a, (b, b', 'a, a), b, b', 'a, a, (b], b)']:
            with self.subTest(text=text):
                self.assertEqual(prompt_builder.dedupe_prompt_terms(text), (text, False))

    def test_manual_and_generated_merge_preserve_weighted_groups(self):
        generated = '(a, b:1.2), portrait'
        original = '(a, c:1.2), portrait'
        expected = '(a, b:1.2), portrait, (a, c:1.2)'
        self.assertEqual(prompt_builder.build_manual_positive(generated, original), expected)
        details = prompt_builder.build_final_positive_details(generated, original)
        self.assertEqual(details['final_positive_after_dedupe'], expected)
        self.assertTrue(details['dedupe_applied'])
if __name__ == '__main__':
    unittest.main()
