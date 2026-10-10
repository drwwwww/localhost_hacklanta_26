import unittest
from unittest import mock

from src.model.verdict import combined_risk
from src.realworld import live, reddit
from src.realworld.dataset import snippet


class QuotedMessageTest(unittest.TestCase):
    def test_extracts_curly_and_straight_quotes(self):
        body = ('I got this on LinkedIn: “Hello, we reviewed your resume and would like to offer you a remote '
                'data entry role paying $45 an hour, please text our hiring manager on Telegram today.” Is it real?')
        self.assertIn("Telegram", reddit.quoted_message(body))
        self.assertNotIn("Is it real", reddit.quoted_message(body))

    def test_extracts_italic_block(self):
        body = "They sent:\n\n*Hi there, you have been selected for a flexible task-based position. Complete simple tasks and earn daily, a small deposit unlocks withdrawals.*\n\nThoughts?"
        self.assertIn("deposit unlocks", reddit.quoted_message(body))

    def test_short_quotes_are_ignored(self):
        self.assertEqual(reddit.quoted_message('They said "you are hired" and nothing else.'), "")


class CommunityVerdictTest(unittest.TestCase):
    def verdict(self, comments):
        tree = [{"kind": "t1", "data": {"author": a, "score": s, "body": b}} for a, s, b in comments]
        with mock.patch.object(reddit, "_cached", return_value=tree):
            return reddit.community_verdict("abc")[0]

    def test_scam_consensus(self):
        self.assertEqual(self.verdict([("u1", 40, "Classic task scam, block them."), ("u2", 5, "Scam.")]), 1)

    def test_not_a_real_job_counts_as_scam(self):
        self.assertEqual(self.verdict([("u1", 12, "This is not a real job, they want your money.")]), 1)

    def test_legit_consensus(self):
        self.assertEqual(self.verdict([("u1", 20, "This is legit, that company really does hire like this.")]), 0)

    def test_automoderator_and_mixed_threads_are_dropped(self):
        self.assertIsNone(self.verdict([("AutoModerator", 99, "scam scam scam")]))
        self.assertIsNone(self.verdict([("u1", 10, "Scam."), ("u2", 10, "Looks legit to me.")]))


class LegitPostingFalsePositiveTest(unittest.TestCase):
    """Phrases from real 2023 LinkedIn postings that used to be rated high risk by the rules."""

    def test_staffing_boilerplate_is_not_upfront_payment(self):
        from src.redflags import rule_risk
        text = "Resumes submitted by agencies are unsolicited and Guidehouse will not be obligated to pay a placement fee."
        self.assertEqual(rule_risk(text), "low")

    def test_cashier_and_bank_duties_are_not_money_forwarding(self):
        from src.redflags import rule_risk
        self.assertEqual(rule_risk("Performs point-of-sale service, e.g. cash sales, credit and check sales, returns "
                                   "and exchanges. Performs the daily balance of cash drawer."), "low")
        self.assertEqual(rule_risk("Opening and closing of the branch, overdraft monitoring, wire transfers and collections."), "low")
        self.assertEqual(rule_risk("Enter payment records into the accounting system and reconcile bank accounts."), "low")

    def test_scam_phrasing_still_fires(self):
        from src.redflags import rule_risk
        self.assertEqual(rule_risk("Pay a placement fee of $150 to secure your position."), "high")
        self.assertEqual(rule_risk("Deposit the check, keep your pay and send the remaining balance by wire transfer."), "high")
        self.assertEqual(rule_risk("Send your full name, address and bank details to our hiring manager."), "high")


class VerdictAndSnippetTest(unittest.TestCase):
    def test_rules_can_raise_but_never_lower(self):
        scam = "Pay a $60 registration fee for your starter kit and send your bank details on Telegram."
        self.assertEqual(combined_risk("low", scam), "high")
        self.assertEqual(combined_risk("high", "Software engineering intern, paid, apply on our careers page."), "high")

    def test_snippet_is_deterministic_and_short(self):
        text = " ".join(f"w{i}" for i in range(200))
        self.assertEqual(snippet(text), snippet(text))
        self.assertEqual(len(snippet(text).split()), 17)


class LivePostingsTest(unittest.TestCase):
    def test_html_becomes_plain_text_with_line_breaks(self):
        markup = "&lt;p&gt;Pay: $20&amp;nbsp;an hour&lt;/p&gt;&lt;ul&gt;&lt;li&gt;Close the register&lt;/li&gt;&lt;li&gt;Stock shelves&lt;/li&gt;&lt;/ul&gt;"
        self.assertEqual(live.html_to_text(markup), "Pay: $20 an hour\nClose the register\nStock shelves")

    def test_script_content_is_dropped(self):
        self.assertEqual(live.html_to_text("<p>Apply today</p><script>track('x')</script>"), "Apply today")

    def test_lever_rows_join_description_and_lists(self):
        payload = {"fetched": "2026-10-10", "data": [{
            "id": "abc", "text": "Barista", "hostedUrl": "https://jobs.lever.co/x/abc", "categories": {"location": "Atlanta"},
            "descriptionPlain": "Make coffee.", "lists": [{"text": "Requirements", "content": "<li>Friendly</li>"}],
            "additionalPlain": "Equal opportunity employer."}]}
        row = next(live._rows("lever", "x", payload))
        self.assertEqual(row["body"], "Make coffee.\nRequirements\nFriendly\nEqual opportunity employer.")
        self.assertEqual((row["source_id"], row["title"], row["location"]), ("abc", "Barista", "Atlanta"))


if __name__ == "__main__":
    unittest.main()
