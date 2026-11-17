"""The example dialogues of tasks/task_families.md, played against the task logic with a fake clock."""

import unittest
from pathlib import Path

from dialogue import Dialogue

KB_PATH = Path(__file__).resolve().parent.parent / "toy_kb" / "knowledge_base.json"


class DialogueTest(unittest.TestCase):
    def setUp(self):
        self.clock = 0.0
        self.dialogue = Dialogue(KB_PATH, now=lambda: self.clock)

    def say(self, intent, **fields):
        return self.dialogue.handle({"intent": intent, **fields})

    def test_timer_set_change_check(self):
        self.assertEqual(self.say("timer_set", minutes=10, label="tea"), "Ten minutes for the tea, starting now.")
        self.assertEqual(self.say("timer_change", mode="set", minutes=15), "Changed. Fifteen minutes for the tea.")
        self.clock = 20
        self.assertEqual(self.say("timer_check"), "About fourteen minutes left on the tea timer.")

    def test_timer_without_length_asks(self):
        self.assertEqual(self.say("timer_set", label="pasta"), "How long should the pasta timer be?")
        self.assertEqual(self.say("timer_set", minutes=12), "Twelve minutes for the pasta, starting now.")
        self.assertIn("now has about fifteen minutes left", self.say("timer_change", mode="add", minutes=3))

    def test_two_timers_list_and_cancel(self):
        self.say("timer_set", minutes=5, label="eggs")
        self.say("timer_set", minutes=20, label="the laundry")
        self.clock = 30
        self.assertEqual(
            self.say("list", what="timers"),
            "Two timers: eggs with about four minutes left and laundry with about nineteen minutes left.",
        )
        self.assertEqual(self.say("timer_cancel"), "Which one: the eggs timer or the laundry timer?")
        self.assertEqual(self.say("timer_cancel", label="eggs"), "Cancelled the eggs timer.")
        self.assertEqual(self.say("timer_cancel", label="soup"), "I don't have a soup timer. One timer: laundry with about nineteen minutes left.")

    def test_reminder_asks_for_time(self):
        self.assertEqual(self.say("reminder_set", label="water the plants"), "At what time?")
        self.assertEqual(
            self.say("reminder_set", time="18:00"),
            "I will remind you to water the plants at six in the evening.",
        )
        self.assertEqual(
            self.say("list", what="reminders"), "One reminder: water the plants at six in the evening."
        )
        self.assertEqual(self.say("reminder_cancel", label="plants"), "Cancelled the reminder to water the plants.")
        self.assertEqual(self.say("list", what="all"), "You have no timers or reminders.")

    def test_booking_match(self):
        self.assertEqual(self.say("booking", day="thursday", period="afternoon"), "How many people?")
        self.assertEqual(self.say("booking", people=5), "Do you need any equipment?")
        self.assertEqual(
            self.say("booking", features=["projector"]),
            "Cedar Room on the first floor is free on Thursday at two in the afternoon. "
            "It holds eight and has a projector. Shall I book it?",
        )
        self.assertEqual(
            self.say("booking", confirm="yes", name="Sam"),
            "Booked. Cedar Room, Thursday at two in the afternoon, for Sam.",
        )
        # The slot is now taken, so the same request gets the next free hour.
        self.say("booking", day="thursday", period="afternoon", people=5)
        self.assertIn("Thursday at three in the afternoon", self.say("booking", features=["projector"]))

    def test_booking_no_room_satisfies_both(self):
        self.assertEqual(
            self.say("booking", people=15, features=["whiteboard"]),
            "No room holds fifteen and has a whiteboard. Maple Hall holds twenty but has no whiteboard. "
            "Cedar Room has a whiteboard but holds only eight. Would you like Maple Hall, or a smaller group?",
        )
        self.assertEqual(
            self.say("booking", room="Maple Hall", day="wednesday", time="02:00", confirm="yes"),
            "Maple Hall on the third floor is free on Wednesday at two in the afternoon. "
            "It holds twenty but has no whiteboard. Shall I book it?",
        )
        self.assertEqual(self.say("booking", confirm="yes"), "Under what name should I book it?")
        self.assertEqual(
            self.say("booking", name="Jordan"), "Booked. Maple Hall, Wednesday at two in the afternoon, for Jordan."
        )

    def test_booking_change_of_mind(self):
        self.assertEqual(
            self.say("booking", features=["quiet"], people=2, day="monday", time="12:00"),
            "Oak Nook on the third floor is free on Monday at noon. It holds two and is quiet. Shall I book it?",
        )
        self.assertEqual(
            self.say("booking", people=3),
            "No quiet room for three is free on Monday at noon. Pine Room holds four and is quiet, "
            "but it is free only on Tuesday, Thursday and Friday. Would you like one of those days?",
        )
        self.assertIn("Pine Room on the second floor is free on Tuesday at eleven", self.say("booking", day="tuesday", time="11:00"))

    def test_booking_edge_cases(self):
        self.assertIn("No room holds thirty", self.say("booking", people=30))
        self.dialogue.booking.people = None
        self.assertIn("No room has a piano", self.say("booking", people=4, features=["piano"]))
        self.assertEqual(self.say("booking", features=[], day="monday"), "What time on Monday?")
        reply = self.say("booking", time="12:00")
        self.assertIn("No room for four is free on Monday at noon. Birch Room is free on Monday at nine", reply)
        self.assertIn("Okay, I have not booked it", (self.say("booking", time="09:00"), self.say("booking", confirm="no"))[1])
        self.assertIn("too small for four", self.say("booking", room="oak nook", time="12:00"))

    def test_equipment_question_answered_no(self):
        self.say("booking", day="friday", period="morning", people=3)
        self.assertIn("Birch Room on the first floor is free on Friday at ten", self.say("booking", confirm="no"))

    def test_lookup_goes_to_the_llm(self):
        self.assertIsNone(self.say("lookup"))
        self.assertIn("Corner Cup", self.dialogue.knowledge_text())


if __name__ == "__main__":
    unittest.main()
