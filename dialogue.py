"""Conversation state and the task logic that must never be wrong: timers, reminders, room booking.

The LLM only turns the user's words into an action (ACTION_SCHEMA) and speaks the reply. Everything
that needs arithmetic or a search over the knowledge base happens here, so a small model cannot
invent a room or break a constraint.
"""

from __future__ import annotations

import datetime
import difflib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from nlu import DAYS, ONES, TENS, extract

# Hours a part of the day covers, as [first, last).
PERIODS = {"morning": (0, 12), "afternoon": (12, 17), "evening": (17, 24)}

# Rooms are free from 09:00 to 18:00, so a spoken "two" or "six" means the afternoon.
EARLIEST_ROOM_HOUR = 7

INTENTS = (
    "timer_set",
    "timer_change",
    "timer_check",
    "timer_cancel",
    "reminder_set",
    "reminder_cancel",
    "list",
    "booking",
    "booking_cancel",
    "lookup",
    "other",
)

# What the LLM is asked for. Every other value is read from the transcript by nlu.extract.
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "label": {"type": "string"},
        "name": {"type": "string"},
        "room": {"type": "string"},
        "features": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["intent"],
    "additionalProperties": False,
}

# Intents that a short answer to the agent's own question can be mistaken for.
VAGUE_INTENTS = ("other", "booking", "timer_set", "timer_change", "timer_check", "reminder_set")

BOOKING_SLOTS = ("people", "day", "time", "period", "features", "room", "confirm", "name", "number", "nothing")

ORDINALS = {1: "first", 2: "second", 3: "third"}

IGNORED_LABEL_WORDS = {"the", "a", "an", "my", "to", "for", "timer", "reminder", "one", "it"}
GENERIC_ROOM_WORDS = {"the", "room", "hall", "studio", "nook"}


# ---- Saying values aloud -------------------------------------------------------------------------


def say_number(number: int) -> str:
    if number < 20:
        return ONES[number]

    if number < 100:
        ones = number % 10
        return TENS[number // 10] + (f" {ONES[ones]}" if ones else "")

    if number < 1000:
        rest = number % 100
        return f"{ONES[number // 100]} hundred" + (f" {say_number(rest)}" if rest else "")

    return str(number)


def say_list(items: list[str], conjunction: str = "and") -> str:
    if len(items) <= 1:
        return "".join(items)

    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"


def say_duration(seconds: float) -> str:
    hours, rest = divmod(max(0, round(seconds)), 3600)
    minutes, seconds = divmod(rest, 60)
    parts = [
        f"{say_number(value)} {unit}{'' if value == 1 else 's'}"
        for value, unit in ((hours, "hour"), (minutes, "minute"), (seconds, "second"))
        if value
    ]

    return " and ".join(parts) or "zero seconds"


def say_remaining(seconds: float) -> str:
    """Round down to whole minutes, as a person reading a timer does; keep seconds when little is left."""
    if seconds >= 120:
        return "about " + say_duration(seconds // 60 * 60)

    return "about " + say_duration(max(5, round(seconds / 5) * 5))


def say_time(hhmm: str) -> str:
    hour, minute = (int(part) for part in hhmm.split(":"))

    if minute == 0 and hour in (0, 12):
        return "noon" if hour else "midnight"

    if hour < 12:
        part_of_day = "in the morning"
    elif hour < 17:
        part_of_day = "in the afternoon"
    elif hour < 21:
        part_of_day = "in the evening"
    else:
        part_of_day = "at night"

    minutes = ""

    if minute:
        minutes = f" oh {ONES[minute]}" if minute < 10 else f" {say_number(minute)}"

    return f"{say_number(hour % 12 or 12)}{minutes} {part_of_day}"


def parse_time(value: object) -> str | None:
    """Return "HH:MM" for anything that reads as a time of day, or None."""
    match = re.fullmatch(r"\s*(\d{1,2})(?::?(\d{2}))?\s*", str(value or ""))

    if not match:
        return None

    hour, minute = int(match[1]), int(match[2] or 0)

    return f"{hour:02d}:{minute:02d}" if hour < 24 and minute < 60 else None


def with_article(noun: str) -> str:
    return f"{'an' if noun[0] in 'aeiou' else 'a'} {noun}"


def capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


def whole_number(value: object) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0


def duration_seconds(action: dict) -> int:
    return whole_number(action.get("seconds"))


def label_words(label: str | None) -> set[str]:
    words = re.findall(r"[a-z0-9']+", (label or "").lower())

    return {word.rstrip("s") for word in words} - IGNORED_LABEL_WORDS


def clean_label(label: object) -> str | None:
    text = re.sub(r"^(?:me\s+)?(?:to|the|a|an|my)\s+", "", str(label or "").strip(" ."), flags=re.I)
    text = re.sub(r"\s+(?:timer|reminder)$", "", text, flags=re.I)

    return text or None


# ---- State ---------------------------------------------------------------------------------------


@dataclass
class Timer:
    label: str | None
    seconds: int
    started: float

    @property
    def name(self) -> str:
        """What the timer is called: its label, or its length ("five minute")."""
        return self.label or re.sub(r"s\b", "", say_duration(self.seconds))

    @property
    def ends(self) -> float:
        return self.started + self.seconds


@dataclass
class Reminder:
    label: str
    time: str
    day: str | None = None

    @property
    def name(self) -> str:
        return self.label

    @property
    def when(self) -> str:
        return (f"on {self.day.capitalize()} " if self.day else "") + f"at {say_time(self.time)}"


@dataclass
class Booking:
    """The room request being built up over several turns."""

    day: str | None = None
    time: str | None = None
    period: str | None = None
    people: int | None = None
    features: list[str] | None = None  # None: not asked yet; []: the user wants none
    room: dict | None = None  # a room the user chose by name
    name: str | None = None
    proposal: tuple[dict, str, str] | None = None  # room, day, time waiting for a yes
    accepted: bool = False
    question: str | None = None  # what the agent just asked: when, people, features, confirm, name


class Dialogue:
    """One conversation. `handle` applies an action and returns what to tell the user, or None when
    the LLM should answer by itself from the knowledge base (lookup and small talk)."""

    def __init__(
        self, kb_path: Path, now: Callable[[], float] = time.monotonic, today: int | None = None
    ) -> None:
        self.kb = json.loads(Path(kb_path).read_text(encoding="utf-8"))
        self.rooms = self.kb["rooms"]
        self.known_features = sorted({f for room in self.rooms for f in room["features"]})
        self.now = now
        self.today = datetime.date.today().weekday() if today is None else today
        self.timers: list[Timer] = []
        self.reminders: list[Reminder] = []
        self.booking = Booking()
        self.booked: list[tuple[dict, str, str, str]] = []  # room, day, time, name
        self.pending: dict | None = None

    def interpret(self, transcript: str, llm_action: dict) -> dict:
        """Build the action from the LLM's intent and labels plus the values the rules read."""
        slots = extract(transcript, self.known_features, [room["id"] for room in self.rooms], self.today)
        text = slots["text"]
        booking = self.booking
        intent = llm_action.get("intent") if llm_action.get("intent") in INTENTS else "other"

        def said(value: object) -> str | None:
            """Keep a free-text value from the LLM only if the user actually said it."""
            value = str(value or "").strip(" .")
            return value if value and re.sub(r"[^a-z0-9' ]", "", value.lower()) in text else None

        label = said(llm_action.get("label"))

        # A short answer takes its meaning from the question, whatever the LLM made of it.
        if "timer" in text or "remind" in text or intent not in VAGUE_INTENTS:
            pass

        elif self.pending:
            asked = self.pending["intent"]

            if asked.startswith("timer") and ("seconds" in slots or "number" in slots):
                intent = asked
            elif asked == "reminder_set" and ("time" in slots or "number" in slots or intent == "other"):
                intent = asked
            elif asked.endswith("cancel") and intent == "other":
                intent = asked

        elif booking.question == "name":
            intent = "booking"

        elif "seconds" not in slots and any(key in slots for key in BOOKING_SLOTS):
            if booking.question or (booking != Booking() and intent in ("other", "timer_change")):
                intent = "booking"

        action: dict = {"intent": intent}
        number = slots.get("number")

        if intent.startswith("timer") or intent == "reminder_cancel":
            items = self.reminders if intent == "reminder_cancel" else self.timers
            words = set(text.split())

            # "Cancel the eggs one" names a timer even when the LLM gave no label.
            if not label:
                label = next((item.name for item in items if label_words(item.name) & label_words(text)), None)

            action["label"] = label or ("all" if words & {"all", "every", "everything"} else None)
            action["seconds"] = slots.get("seconds") or (number or 0) * 60
            action["mode"] = slots["mode"]

        elif intent == "reminder_set":
            action["label"] = label
            action["time"] = slots.get("time")
            action["day"] = slots.get("day")

            if action["time"] is None and number is not None and number < 24:
                action["time"] = f"{number + 12 if 1 <= number <= 6 else number}:00"

        elif intent == "list":
            action["what"] = slots["what"]

        elif intent == "booking":
            for key in ("day", "time", "period", "people", "confirm"):
                action[key] = slots.get(key)

            if number is not None:
                if booking.question == "when" and number < 24:
                    action["time"] = f"{number}:00"
                elif action["people"] is None:
                    action["people"] = number

            current = booking.features or []
            wanted = slots.get("features") or [
                str(f) for f in llm_action.get("features") or [] if said(f) and f not in current
            ]
            dropped = slots.get("dropped_features", [])

            if wanted or dropped:
                kept = [] if slots.get("only_features") else current
                action["features"] = [f for f in kept + wanted if f not in dropped]

            elif slots.get("nothing"):
                action["features"] = []

            elif booking.question == "features" and slots.get("confirm") == "no":
                # "No" in answer to "Do you need any equipment?" declines equipment, not a booking.
                action["features"], action["confirm"] = [], None

            action["room"] = slots.get("room") or said(llm_action.get("room"))
            action["name"] = slots.get("name") or said(llm_action.get("name"))

            if booking.question == "name" and not action["name"]:
                opener = r"^(?:(?:it's|it is|my name is|the name is|name is|under|for|yes|sure|okay)[, ]+)+"
                rest = re.sub(opener, "", transcript.strip(" .!"), flags=re.I)
                action["name"] = rest if 0 < len(rest.split()) <= 3 else None

        return {key: value for key, value in action.items() if value is not None}

    def handle(self, action: dict) -> str | None:
        intent = action.get("intent")

        # A short answer to "At what time?" carries only the missing value; the rest was kept.
        if self.pending and self.pending["intent"] == intent:
            given = {key: value for key, value in action.items() if value not in (None, "")}
            action = {**self.pending, **given}

        self.pending = None
        handler = getattr(self, f"_{intent}", None) if intent in INTENTS else None

        return handler(action) if handler else None

    def summary(self) -> str:
        """The state the action model needs to read a short follow-up correctly."""
        lines = []

        if self.timers:
            lines.append("Timers: " + ", ".join(timer.name for timer in self.timers))

        if self.reminders:
            lines.append("Reminders: " + ", ".join(f"{r.label} at {r.time}" for r in self.reminders))

        booking = self.booking
        slots = {
            "day": booking.day,
            "time": booking.time or booking.period,
            "people": booking.people,
            "features": booking.features,
            "room": booking.room and booking.room["name"],
        }
        known = {key: value for key, value in slots.items() if value is not None}

        if known:
            lines.append(f"Room request so far: {json.dumps(known)}")

        if booking.proposal:
            room, day, hhmm = booking.proposal
            lines.append(f"Proposed and waiting for a yes: {room['name']}, {day} {hhmm}")

        if self.booked:
            lines.append(
                "Booked: " + "; ".join(f"{r['name']} {d} {t} for {n}" for r, d, t, n in self.booked)
            )

        return "\n".join(lines) or "Nothing is set."

    def knowledge_text(self) -> str:
        """The whole knowledge base as plain lines, for the lookup prompt."""
        kb = self.kb
        lines = [f"{kb['venue']['name']} has {kb['venue']['floors']} floors.", "Opening hours:"]
        lines += [f"- {day}: {h['open']} to {h['close']}" for day, h in kb["opening_hours"].items()]
        lines.append("Areas and services:")
        lines += [f"- {e['name']} (floor {e['floor']}): {e['text']}" for e in kb["areas"] + kb["services"]]
        lines.append("Policies:")
        lines += [f"- {policy['topic']}: {policy['text']}" for policy in kb["policies"]]
        lines.append("Meeting rooms (free one-hour slots by start time):")

        for room in self.rooms:
            free = "; ".join(f"{day} {', '.join(t)}" for day, t in room["free_hour_starts"].items())
            lines.append(
                f"- {room['name']}: floor {room['floor']}, holds {room['capacity']}, "
                f"features: {', '.join(room['features'])}. Free: {free}."
            )

        return "\n".join(lines)

    # ---- Timers ----------------------------------------------------------------------------------

    def _timer_set(self, action: dict) -> str:
        seconds = duration_seconds(action)
        label = clean_label(action.get("label"))

        if not seconds:
            self.pending = {"intent": "timer_set", "label": label}
            return f"How long should the {label + ' ' if label else ''}timer be?"

        timer = Timer(label, seconds, self.now())
        self.timers.append(timer)

        if label:
            return f"{capitalize(say_duration(seconds))} for the {label}, starting now."

        return f"{capitalize(timer.name)} timer, starting now."

    def _timer_change(self, action: dict) -> str:
        label = clean_label(action.get("label"))
        timer, _ = self._select(self.timers, label)

        if not self.timers:
            return "You have no timers running."

        if timer is None and label_words(label):
            return f"I don't have a {label} timer. {self._list_timers()}"

        # "Make it fifteen" means the timer just talked about.
        timer = timer or self.timers[-1]
        seconds = duration_seconds(action)

        if not seconds:
            self.pending = {"intent": "timer_change", "label": timer.label, "mode": action.get("mode")}
            return f"What should I change the {timer.name} timer to?"

        if action.get("mode") == "add":
            timer.seconds += seconds
            left = say_remaining(timer.ends - self.now())
            return f"Added {say_duration(seconds)}. The {timer.name} timer now has {left} left."

        timer.seconds = seconds
        length = capitalize(say_duration(seconds))

        if timer.label:
            return f"Changed. {length} for the {timer.label}."

        return f"Changed. {capitalize(timer.name)} timer."

    def _timer_check(self, action: dict) -> str:
        label = clean_label(action.get("label"))
        timer, _ = self._select(self.timers, label)

        if not self.timers:
            return "You have no timers running."

        if timer is None and label_words(label):
            return f"I don't have a {label} timer. {self._list_timers()}"

        return " ".join(self._remaining_sentence(t) for t in ([timer] if timer else list(self.timers)))

    def _timer_cancel(self, action: dict) -> str:
        label = clean_label(action.get("label"))

        if not self.timers:
            return "You have no timers running."

        if label and label.lower() in ("all", "everything"):
            self.timers.clear()
            return "Cancelled all timers."

        timer, ambiguous = self._select(self.timers, label)

        if ambiguous:
            self.pending = {"intent": "timer_cancel"}
            names = [f"the {t.name} timer" for t in self.timers]
            return f"Which one: {say_list(names, 'or')}?"

        if timer is None:
            return f"I don't have a {label} timer. {self._list_timers()}"

        self.timers.remove(timer)

        return f"Cancelled the {timer.name} timer."

    def _remaining_sentence(self, timer: Timer) -> str:
        left = timer.ends - self.now()

        if left <= 0:
            self.timers.remove(timer)
            return f"The {timer.name} timer has finished."

        return f"{capitalize(say_remaining(left))} left on the {timer.name} timer."

    def _list_timers(self) -> str:
        if not self.timers:
            return "You have no timers running."

        items = []

        for timer in list(self.timers):
            left = timer.ends - self.now()

            if left <= 0:
                self.timers.remove(timer)
                items.append(f"{timer.name} which has finished")
            else:
                items.append(f"{timer.name} with {say_remaining(left)} left")

        count = capitalize(say_number(len(items)))

        return f"{count} timer{'' if len(items) == 1 else 's'}: {say_list(items)}."

    def _select(self, items: list, label: str | None) -> tuple[object | None, bool]:
        """Return (the item the label means, whether the choice is ambiguous)."""
        wanted = label_words(label)

        if wanted:
            for item in reversed(items):
                have = label_words(item.name)

                if have and (wanted <= have or have <= wanted):
                    return item, False

            return None, False

        return (items[0], False) if len(items) == 1 else (None, len(items) > 1)

    # ---- Reminders -------------------------------------------------------------------------------

    def _reminder_set(self, action: dict) -> str:
        label = clean_label(action.get("label"))
        hhmm = parse_time(action.get("time"))
        day = action.get("day") if action.get("day") in DAYS else None
        existing, _ = self._select(self.reminders, label) if label else (None, False)

        if existing and hhmm:
            existing.time, existing.day = hhmm, day or existing.day
            return f"Changed. I will remind you to {existing.label} {existing.when}."

        if not label:
            self.pending = {"intent": "reminder_set", "time": hhmm, "day": day}
            return "What should I remind you about?"

        if not hhmm:
            self.pending = {"intent": "reminder_set", "label": label, "day": day}
            return "At what time?"

        reminder = Reminder(label, hhmm, day)
        self.reminders.append(reminder)

        return f"I will remind you to {label} {reminder.when}."

    def _reminder_cancel(self, action: dict) -> str:
        label = clean_label(action.get("label"))

        if not self.reminders:
            return "You have no reminders."

        if label and label.lower() in ("all", "everything"):
            self.reminders.clear()
            return "Cancelled all reminders."

        reminder, ambiguous = self._select(self.reminders, label)

        if ambiguous:
            self.pending = {"intent": "reminder_cancel"}
            return f"Which one: {say_list([r.label for r in self.reminders], 'or')}?"

        if reminder is None:
            return f"I don't have a reminder about {label}. {self._list_reminders()}"

        self.reminders.remove(reminder)

        return f"Cancelled the reminder to {reminder.label}."

    def _list_reminders(self) -> str:
        if not self.reminders:
            return "You have no reminders."

        items = [f"{reminder.label} {reminder.when}" for reminder in self.reminders]
        count = capitalize(say_number(len(items)))

        return f"{count} reminder{'' if len(items) == 1 else 's'}: {say_list(items)}."

    def _list(self, action: dict) -> str:
        what = action.get("what")

        if what == "timers":
            return self._list_timers()

        if what == "reminders":
            return self._list_reminders()

        if not self.timers and not self.reminders:
            return "You have no timers or reminders."

        return f"{self._list_timers()} {self._list_reminders()}"

    # ---- Room booking ----------------------------------------------------------------------------

    def _booking(self, action: dict) -> str:
        booking = self.booking
        before = (booking.day, booking.time, booking.period, booking.people, booking.features, booking.room)

        if action.get("day") in DAYS:
            booking.day = action["day"]

        hhmm = parse_time(action.get("time"))

        if hhmm:
            hour = int(hhmm[:2])
            hhmm = f"{hour + 12:02d}{hhmm[2:]}" if hour < EARLIEST_ROOM_HOUR else hhmm
            booking.time, booking.period = hhmm, None

        elif action.get("period") in PERIODS:
            first, last = PERIODS[action["period"]]

            # "Thursday afternoon" after "at three" adds nothing; keep the exact time.
            if not (booking.time and first <= int(booking.time[:2]) < last):
                booking.time, booking.period = None, action["period"]

        if whole_number(action.get("people")):
            booking.people = whole_number(action["people"])

        if isinstance(action.get("features"), list):
            booking.features = self._normalize_features(action["features"])

        booking.question = None

        if action.get("room"):
            room = self._find_room(action["room"])

            if room is None:
                names = say_list([r["name"] for r in self.rooms])
                return f"I don't have a room called {action['room']}. The rooms are {names}."

            booking.room = room

        elif (booking.people, booking.features) != (before[3], before[4]):
            # A room picked for the old constraints may not fit the new ones.
            booking.room = None

        if action.get("name"):
            booking.name = str(action["name"]).strip(" .")

        after = (booking.day, booking.time, booking.period, booking.people, booking.features, booking.room)

        if after != before:
            booking.proposal, booking.accepted = None, False

        elif booking.proposal:
            if action.get("confirm") == "no":
                booking.proposal, booking.accepted, booking.question = None, False, "when"
                return "Okay, I have not booked it. Would you like a different room, day or time?"

            if action.get("confirm") == "yes" or action.get("name"):
                booking.accepted = True

            if booking.accepted:
                if not booking.name:
                    booking.question = "name"
                    return "Under what name should I book it?"

                room, day, hhmm = booking.proposal
                self.booked.append((room, day, hhmm, booking.name))
                self.booking = Booking()

                return f"Booked. {room['name']}, {day.capitalize()} at {say_time(hhmm)}, for {booking.name}."

        return self._search()

    def _booking_cancel(self, action: dict) -> str:
        if self.booked:
            room, day, hhmm, name = self.booked.pop()
            return f"Cancelled. {room['name']}, {day.capitalize()} at {say_time(hhmm)}, for {name}."

        if self.booking != Booking():
            self.booking = Booking()
            return "Okay, I have dropped the room request."

        return "You have no bookings."

    def _search(self) -> str:
        booking = self.booking
        wanted = booking.features or []

        if booking.room:
            room = booking.room

            if booking.people and room["capacity"] < booking.people:
                booking.room = None
                too_small = (
                    f"{room['name']} holds only {say_number(room['capacity'])}, "
                    f"too small for {say_number(booking.people)}."
                )
                return f"{too_small} {self._search()}"

            candidates = [room]

        else:
            candidates = sorted(
                (room for room in self.rooms if self._fits(room, booking.people, wanted)),
                key=lambda room: room["capacity"],
            )

            if not candidates:
                booking.question = "people"
                return self._nothing_fits()

        # Unless a branch below says otherwise, the next answer is about the day or the time.
        booking.question = "when"

        if not booking.day and not (booking.time or booking.period):
            return "Which day and time would you like?"

        if not booking.day:
            return "Which day?"

        if not (booking.time or booking.period):
            return f"What time on {booking.day.capitalize()}?"

        if booking.people is None:
            booking.question = "people"
            return "How many people?"

        if booking.features is None and not booking.room:
            booking.question = "features"
            return "Do you need any equipment?"

        for room in candidates:
            slots = [hhmm for hhmm in self._free(room, booking.day) if self._time_wanted(hhmm)]

            if slots:
                booking.proposal, booking.question = (room, booking.day, slots[0]), "confirm"
                return (
                    f"{room['name']} on the {ORDINALS[room['floor']]} floor is free on "
                    f"{booking.day.capitalize()} at {say_time(slots[0])}. "
                    f"It {self._describe(room, wanted)}. Shall I book it?"
                )

        return self._nothing_free(candidates)

    def _nothing_free(self, candidates: list[dict]) -> str:
        """Every fitting room is taken at that time: offer other times that day, else other days."""
        booking = self.booking
        wanted = booking.features or []
        day = booking.day.capitalize()
        when = f"on {day} at {say_time(booking.time)}" if booking.time else f"on {day} {booking.period}"

        if booking.room:
            head = f"{booking.room['name']} is not free {when}."
        else:
            head = f"No {self._wanted_phrase()} is free {when}."

        for room in candidates:
            times = self._free(room, booking.day)

            if times:
                spoken = say_list([say_time(hhmm) for hhmm in times], "or")
                return f"{head} {room['name']} is free on {day} at {spoken}. Would you like one of those times?"

        for room in candidates:
            days = [d.capitalize() for d in DAYS if self._free(room, d)]

            if days:
                return (
                    f"{head} {room['name']} {self._describe(room, wanted)}, but it is free only on "
                    f"{say_list(days)}. Would you like one of those days?"
                )

        return f"{head} Would you like to change the day or the room?"

    def _nothing_fits(self) -> str:
        """No room satisfies the constraints at any time: say which ones clash and offer the closest."""
        booking = self.booking
        wanted = booking.features or []
        people = booking.people
        unknown = [feature for feature in wanted if feature not in self.known_features]

        if unknown:
            offered = say_list(self.known_features)
            return f"No room has {say_list([with_article(f) for f in unknown], 'or')}. The rooms offer: {offered}."

        largest = max(self.rooms, key=lambda room: room["capacity"])

        if people and people > largest["capacity"]:
            return (
                f"No room holds {say_number(people)}. The largest is {largest['name']}, which holds "
                f"{say_number(largest['capacity'])}. Would you like a smaller group?"
            )

        clauses = ([f"holds {say_number(people)}"] if people else []) + self._feature_clauses(wanted)
        big_enough = [room for room in self.rooms if self._fits(room, people, [])]
        closest = max(
            big_enough,
            key=lambda room: (sum(f in room["features"] for f in wanted), -room["capacity"]),
        )
        reply = f"No room {say_list(clauses)}. {closest['name']} {self._describe(closest, wanted)}."
        with_everything = [room for room in self.rooms if self._fits(room, None, wanted)]

        if not with_everything:
            return f"{reply} Would you like {closest['name']}?"

        best = max(with_everything, key=lambda room: room["capacity"])

        return (
            f"{reply} {best['name']} {say_list(self._feature_clauses(wanted))} but holds only "
            f"{say_number(best['capacity'])}. Would you like {closest['name']}, or a smaller group?"
        )

    def _fits(self, room: dict, people: int | None, features: list[str]) -> bool:
        return (not people or room["capacity"] >= people) and all(f in room["features"] for f in features)

    def _free(self, room: dict, day: str) -> list[str]:
        taken = {(r["id"], d, hhmm) for r, d, hhmm, _ in self.booked}

        return [t for t in room["free_hour_starts"].get(day, []) if (room["id"], day, t) not in taken]

    def _time_wanted(self, hhmm: str) -> bool:
        if self.booking.time:
            return hhmm == self.booking.time

        first, last = PERIODS[self.booking.period]

        return first <= int(hhmm[:2]) < last

    def _feature_clauses(self, features: list[str]) -> list[str]:
        """["is quiet", "has a projector and a whiteboard"] for the features given."""
        nouns = [with_article(feature) for feature in features if feature != "quiet"]

        return (["is quiet"] if "quiet" in features else []) + ([f"has {say_list(nouns)}"] if nouns else [])

    def _describe(self, room: dict, wanted: list[str]) -> str:
        """"holds eight and has a projector", plus "but has no whiteboard" for what it lacks."""
        present = [feature for feature in wanted if feature in room["features"]]
        missing = [feature for feature in wanted if feature not in room["features"]]
        text = say_list([f"holds {say_number(room['capacity'])}"] + self._feature_clauses(present))

        if missing:
            lacks = "is not quiet" if missing == ["quiet"] else f"has no {say_list(missing, 'or')}"
            text += f" but {lacks}"

        return text

    def _wanted_phrase(self) -> str:
        """"quiet room for three with a whiteboard"."""
        booking = self.booking
        wanted = booking.features or []
        nouns = [with_article(feature) for feature in wanted if feature != "quiet"]

        return (
            ("quiet " if "quiet" in wanted else "")
            + "room"
            + (f" for {say_number(booking.people)}" if booking.people else "")
            + (f" with {say_list(nouns)}" if nouns else "")
        )

    def _normalize_features(self, features: list) -> list[str]:
        normalized = []

        for feature in features:
            text = str(feature).lower().strip(" .")

            if text in ("", "no", "none", "nothing"):
                continue

            # "a projector screen" or "video kit" still names a feature the rooms have.
            for known in self.known_features:
                if text in known or known in text:
                    text = known
                    break

            if text not in normalized:
                normalized.append(text)

        return normalized

    def _find_room(self, text: str) -> dict | None:
        """Match a spoken room name by its distinctive word, forgiving a misheard letter or two."""
        words = [w for w in re.findall(r"[a-z]+", str(text).lower()) if w not in GENERIC_ROOM_WORDS]
        ids = [room["id"] for room in self.rooms]

        for word in words:
            close = difflib.get_close_matches(word, ids, n=1, cutoff=0.6)

            if close:
                return self.rooms[ids.index(close[0])]

        return None
