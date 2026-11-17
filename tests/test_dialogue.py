"""The example dialogues of tasks/task_families.md, played against the task logic with a fake clock.

Each user line comes with the intent the LLM is expected to give. Some lines give a wrong intent on
purpose, because a short answer ("Five.") is often misread and the rules must still place it.
"""

import unittest
from pathlib import Path

from dialogue import Dialogue
from nlu import extract, normalize

KB_PATH = Path(__file__).resolve().parent.parent / "toy_kb" / "knowledge_base.json"
MONDAY = 0


class DialogueTest(unittest.TestCase):
    def setUp(self):
        self.clock = 0.0
        self.dialogue = Dialogue(KB_PATH, now=lambda: self.clock, today=MONDAY)

    def user(self, transcript, intent, **llm_fields):
        action = self.dialogue.interpret(transcript, {"intent": intent, **llm_fields})
        return self.dialogue.handle(action)

    def test_timer_set_change_check(self):
        self.assertEqual(
            self.user("Start a 10 minute timer for the tea.", "timer_set", label="tea"),
            "Ten minutes for the tea, starting now.",
        )
        self.assertEqual(self.user("Actually make it fifteen.", "timer_change"), "Changed. Fifteen minutes for the tea.")
        self.clock = 20
        self.assertEqual(self.user("How long is left?", "timer_check"), "About fourteen minutes left on the tea timer.")

    def test_timer_without_length_asks(self):
        self.assertEqual(self.user("Set a timer for the pasta.", "timer_set", label="pasta"), "How long should the pasta timer be?")
        self.assertEqual(self.user("Twelve minutes.", "timer_change"), "Twelve minutes for the pasta, starting now.")
        self.assertEqual(
            self.user("Add 3 more minutes", "timer_change"),
            "Added three minutes. The pasta timer now has about fifteen minutes left.",
        )
        self.assertEqual(self.user("Set a timer.", "timer_set"), "How long should the timer be?")
        self.assertEqual(self.user("90 seconds", "other"), "One minute and thirty second timer, starting now.")

    def test_two_timers_list_and_cancel(self):
        self.user("Set a timer for 5 minutes for the eggs", "timer_set", label="eggs")
        self.user("Set another timer for twenty minutes for the laundry.", "timer_set", label="the laundry")
        self.clock = 30
        self.assertEqual(
            self.user("What timers do I have?", "list"),
            "Two timers: eggs with about four minutes left and laundry with about nineteen minutes left.",
        )
        self.assertEqual(self.user("Cancel a timer.", "timer_cancel"), "Which one: the eggs timer or the laundry timer?")
        self.assertEqual(self.user("The eggs one.", "other"), "Cancelled the eggs timer.")
        self.assertEqual(
            self.user("Cancel the soup timer.", "timer_cancel", label="soup"),
            "I don't have a soup timer. One timer: laundry with about nineteen minutes left.",
        )
        self.assertEqual(self.user("Cancel it.", "timer_cancel"), "Cancelled the laundry timer.")

    def test_reminders(self):
        self.assertEqual(self.user("Remind me to water the plants.", "reminder_set", label="water the plants"), "At what time?")
        self.assertEqual(
            self.user("6 in the evening", "booking"), "I will remind you to water the plants at six in the evening."
        )
        self.assertEqual(
            self.user("Also remind me to pick up the laundry at 4.30 in the afternoon", "reminder_set", label="pick up the laundry"),
            "I will remind you to pick up the laundry at four thirty in the afternoon.",
        )
        self.assertEqual(
            self.user("What reminders do I have?", "list"),
            "Two reminders: water the plants at six in the evening and pick up the laundry at four thirty in the afternoon.",
        )
        self.assertEqual(self.user("Cancel the plants reminder", "reminder_cancel"), "Cancelled the reminder to water the plants.")
        self.assertEqual(self.user("Cancel all my reminders", "reminder_cancel"), "Cancelled all reminders.")
        self.assertEqual(self.user("What do I have set?", "list"), "You have no timers or reminders.")

    def test_booking_match(self):
        self.assertEqual(self.user("I need a room on Thursday afternoon.", "booking"), "How many people?")
        self.assertEqual(self.user("Five.", "other"), "Do you need any equipment?")
        self.assertEqual(
            self.user("A projector.", "other"),
            "Cedar Room on the first floor is free on Thursday at two in the afternoon. "
            "It holds eight and has a projector. Shall I book it?",
        )
        self.assertEqual(
            self.user("Yes, under the name Sam.", "booking"),
            "Booked. Cedar Room, Thursday at two in the afternoon, for Sam.",
        )
        # The slot is now taken, so the same request gets the next free hour.
        reply = self.user("A room for 5 people with a projector on Thursday afternoon", "booking")
        self.assertIn("Cedar Room on the first floor is free on Thursday at three in the afternoon", reply)

    def test_booking_no_room_satisfies_both(self):
        self.assertEqual(
            self.user("Can I get a room for 15 people with a whiteboard?", "booking"),
            "No room holds fifteen and has a whiteboard. Maple Hall holds twenty but has no whiteboard. "
            "Cedar Room has a whiteboard but holds only eight. Would you like Maple Hall, or a smaller group?",
        )
        self.assertEqual(
            self.user("Maple Hall is fine, Wednesday at 2.", "booking"),
            "Maple Hall on the third floor is free on Wednesday at two in the afternoon. "
            "It holds twenty but has no whiteboard. Shall I book it?",
        )
        self.assertEqual(self.user("Yes please.", "other"), "Under what name should I book it?")
        self.assertEqual(self.user("Jordan.", "other"), "Booked. Maple Hall, Wednesday at two in the afternoon, for Jordan.")

    def test_booking_change_of_mind(self):
        self.assertEqual(
            self.user("A quiet room for two people on Monday at noon.", "booking"),
            "Oak Nook on the third floor is free on Monday at noon. It holds two and is quiet. Shall I book it?",
        )
        self.assertEqual(
            self.user("Make it 3 people instead", "timer_change"),
            "No quiet room for three is free on Monday at noon. Pine Room holds four and is quiet, "
            "but it is free only on Tuesday, Thursday and Friday. Would you like one of those days?",
        )
        self.assertIn(
            "Pine Room on the second floor is free on Tuesday at eleven in the morning",
            self.user("OK, Tuesday at 11 then", "timer_change"),
        )
        self.assertEqual(
            self.user("Yes, book it under Maria", "booking"),
            "Booked. Pine Room, Tuesday at eleven in the morning, for Maria.",
        )

    def test_booking_edge_cases(self):
        self.assertIn("No room holds thirty", self.user("I need a room for thirty people", "booking"))
        self.assertIn("No room has a piano", self.user("Four people, and we need a piano", "booking", features=["piano"]))
        self.assertEqual(self.user("No equipment then, on Monday", "booking", features=[]), "What time on Monday?")
        reply = self.user("Noon.", "other")
        self.assertIn("No room for four is free on Monday at noon. Birch Room is free on Monday at nine", reply)
        self.assertIn("Birch Room on the first floor is free on Monday at nine", self.user("Nine.", "other"))
        self.assertIn("Okay, I have not booked it", self.user("No, don't book it.", "booking"))
        self.assertIn("too small for four", self.user("What about Oak Nook at noon?", "booking"))

    def test_equipment_question_answered_no(self):
        self.user("I'd like to book a room for Friday morning.", "booking")
        self.assertEqual(self.user("3 people", "booking"), "Do you need any equipment?")
        self.assertIn("Birch Room on the first floor is free on Friday at ten", self.user("No thanks.", "other"))

    def test_features_add_and_drop(self):
        self.user("A room for 4 with a whiteboard on Friday at 11", "booking")
        self.assertEqual(self.dialogue.booking.proposal[0]["name"], "Pine Room")
        self.user("Forget the whiteboard, it has to be quiet.", "booking")
        self.assertEqual(self.dialogue.booking.features, ["quiet"])

    def test_a_command_is_not_taken_for_an_answer(self):
        self.user("I need a room on Thursday afternoon.", "booking")
        self.assertEqual(self.user("Set a timer for 5 minutes.", "timer_set"), "Five minute timer, starting now.")
        self.assertIsNone(self.user("Where can I print?", "lookup"))
        self.assertEqual(self.user("Six of us.", "other"), "Do you need any equipment?")

    def test_rules(self):
        self.assertEqual(normalize("Four-thirty p.m. for twenty five people"), "4 30 pm for 25 people")
        self.assertEqual(normalize("the eggs one"), "the eggs one")

        def slots(text):
            return extract(text, ["projector", "quiet", "tv screen"], ["oak", "pine"], MONDAY)

        self.assertEqual(slots("an hour and a half")["seconds"], 5400)
        self.assertEqual(slots("at half past two")["time"], "14:30")
        self.assertEqual(slots("9 a.m. tomorrow")["time"], "09:00")
        self.assertEqual(slots("9 a.m. tomorrow")["day"], "tuesday")
        self.assertEqual(slots("a TV and a projector")["features"], ["projector", "tv screen"])
        self.assertEqual(slots("there will be 6 of us")["people"], 6)
        self.assertNotIn("time", slots("for 6 people"))


if __name__ == "__main__":
    unittest.main()
