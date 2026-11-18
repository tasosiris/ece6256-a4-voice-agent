"""Reads the closed-vocabulary values out of a transcript with rules: numbers, lengths of time, days,
times of day, people, room features, yes and no.

A small LLM gets these wrong often enough to fail a booking, and rules cost no latency. The LLM is
left with what rules cannot do: the intent and free-text labels.
"""

from __future__ import annotations

import re

DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
TENS = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()

NUMBER_WORDS = {word: value for value, word in enumerate(ONES)}
NUMBER_WORDS.update({word: value * 10 for value, word in enumerate(TENS) if value >= 2})
# How Whisper spells an accented "five".
NUMBER_WORDS["fife"] = 5

# "one" is a number only next to these; elsewhere it is a pronoun ("the eggs one", "which one").
WORDS_BEFORE_A_COUNT = {"at", "for", "to", "it", "just", "only", "add", "another", "about"}
UNITS = "hour hours minute minutes second seconds person people pm am o'clock more".split()

FEATURE_ALIASES = {
    "tv": "tv screen",
    "television": "tv screen",
    "monitor": "tv screen",
    "white board": "whiteboard",
    "video call": "video call kit",
    "video calls": "video call kit",
    "video conference": "video call kit",
    "video conferencing": "video call kit",
    "mic": "microphone",
    "silent": "quiet",
}

# Words that carry no value in a short answer such as "Actually, make it fifteen."
FILLER = set(
    "yes yeah no please just about around maybe um uh it's its is for at make it to the then ok okay "
    "actually instead let's say change i think be should and so well hmm oh".split()
)

DURATION = re.compile(r"\b(\d+) ?(?:more |extra |additional )?(hour|hr|minute|min|second|sec)s?\b")
PEOPLE = re.compile(
    r"\b(\d+) (?:people|persons?|of us|guests|folks|attendees|students|members)\b"
    r"|\b(?:group|party|team) of (\d+)\b"
    r"|\bfor (\d+)\b(?! ?(?:hour|hr|minute|min|second|sec|am\b|pm\b|o'clock|\d|in the))"
)
YES = re.compile(
    r"^(?:yes|yeah|yep|yup|sure|ok|okay|please|correct|right|perfect|great)\b"
    r"|\b(?:book it|go ahead|sounds good|that works|is fine|that's fine|will do)\b"
)
NO = re.compile(r"^(?:no|nope|nah)\b|\b(?:never mind|don't book|do not book)\b")
NOTHING = re.compile(r"\b(?:none|nothing|no equipment|don't need|do not need)\b")
ADD = re.compile(r"\b(?:add|more|extend|extra|longer|plus|another)\b")
NAME = re.compile(
    r"(?:\bunder(?: the name(?: of)?)?|\bname is|\bname's|\bnamed)\s+"
    r"([A-Z][\w'-]*(?:\s[A-Z][\w'-]*)?)"
)


def normalize(text: str) -> str:
    """Lower-case words with numbers as digits: "Four-thirty p.m." -> "4 30 pm"."""
    text = text.lower().replace("-", " ").replace("’", "'")
    text = re.sub(r"\b([ap])\.? ?m\b\.?", r"\1m", text)
    # Whisper sometimes writes a spoken number as an ordinal: "6th in the evening".
    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", text)
    words = re.findall(r"\d+(?:[:.]\d+)?|[a-z']+", text)
    out = []
    index = 0

    while index < len(words):
        word = words[index]
        following = words[index + 1] if index + 1 < len(words) else ""
        value = NUMBER_WORDS.get(word)

        if value is None:
            out.append(word)

        elif value >= 20 and 1 <= NUMBER_WORDS.get(following, 0) <= 9:
            out.append(str(value + NUMBER_WORDS[following]))
            index += 1

        elif word == "one" and len(words) > 1 and not (
            following in UNITS or (out and out[-1] in WORDS_BEFORE_A_COUNT)
        ):
            out.append(word)

        else:
            out.append(str(value))

        index += 1

    text = " ".join(out)
    text = re.sub(r"\bhalf an? hour\b", "30 minutes", text)
    text = re.sub(r"\ban? (hour|minute|second)\b", r"1 \1", text)
    text = re.sub(r"\b(\d+) hours? and a half\b", r"\1 hours 30 minutes", text)
    text = re.sub(r"\b(\d+) and a half hours?\b", r"\1 hours 30 minutes", text)
    text = re.sub(r"\b(\d+) and a half minutes?\b", r"\1 minutes 30 seconds", text)
    text = re.sub(r"\ba couple(?: of)?\b", "2", text)

    return text


def find_time(text: str) -> str | None:
    """Return the time of day in normalized text as "HH:MM", or None."""
    if re.search(r"\b(?:noon|midday)\b", text):
        return "12:00"

    if re.search(r"\bmidnight\b", text):
        return "00:00"

    spoken = re.search(r"\b(half|quarter) (past|to) (\d{1,2})\b", text)
    clock = (
        re.search(r"\b(\d{1,2})[:.](\d{2})\b", text)
        or re.search(r"\b(\d{1,2}) (\d{2}) ?(?:am\b|pm\b|in the|at night)", text)
        or re.search(r"\bat (\d{1,2}) (\d{2})\b", text)
        or re.search(r"\b(\d{1,2})() ?(?:am|pm|o'clock|oclock)\b", text)
        or re.search(r"\b(\d{1,2})() in the (?:morning|afternoon|evening)", text)
        or re.search(r"\bat (\d{1,2})()\b(?! ?(?:hour|minute|second|people|person))", text)
    )

    if spoken:
        hour = int(spoken[3])
        minute = 30 if spoken[1] == "half" else 15

        if spoken[2] == "to":
            hour, minute = hour - 1, 60 - minute

    elif clock:
        hour, minute = int(clock[1]), int(clock[2] or 0)

    else:
        return None

    if re.search(r"\b(?:pm|afternoon|evening|tonight|night)\b", text):
        hour += 12 if hour < 12 else 0

    elif re.search(r"\d ?am\b|\bmorning\b", text):
        hour = 0 if hour == 12 else hour

    elif 1 <= hour <= 6:
        # Nobody books a room or sets a reminder for two in the morning without saying so.
        hour += 12

    return f"{hour:02d}:{minute:02d}" if hour < 24 and minute < 60 else None


def extract(transcript: str, features: list[str], room_ids: list[str], today: int) -> dict:
    """Return the values found in a transcript. `today` is the weekday index, Monday being 0.

    Keys, each present only when found: seconds, time, period, day, people, number (a lone number
    whose meaning depends on the question it answers), features, dropped_features, only_features,
    room, confirm, nothing, mode, what, name, text (the normalized transcript).
    """
    text = normalize(transcript)
    slots: dict = {"text": text}

    lengths = DURATION.findall(text)

    if lengths:
        per_unit = {"h": 3600, "m": 60, "s": 1}
        slots["seconds"] = sum(int(count) * per_unit[unit[0]] for count, unit in lengths)

    time_of_day = None if lengths else find_time(text)

    if time_of_day:
        slots["time"] = time_of_day

    else:
        period = re.search(r"\b(morning|afternoon|evening|tonight)\b", text)

        if period:
            slots["period"] = "evening" if period[1] == "tonight" else period[1]

    day = re.search(rf"\b({'|'.join(DAYS)})s?\b", text)

    if day:
        slots["day"] = day[1]

    elif re.search(r"\b(?:today|tonight)\b", text):
        slots["day"] = DAYS[today]

    elif re.search(r"\btomorrow\b", text):
        slots["day"] = DAYS[(today + 1) % 7]

    people = PEOPLE.search(text)

    if people:
        slots["people"] = int(next(group for group in people.groups() if group))

    rest = [word for word in text.split() if word not in FILLER]

    if len(rest) == 1 and rest[0].isdigit():
        slots["number"] = int(rest[0])

    names = {**{feature: feature for feature in features}, **FEATURE_ALIASES}
    found, dropped = [], []

    for spoken, feature in names.items():
        if feature not in features or not re.search(rf"\b{spoken}s?\b", text):
            continue

        refused = re.search(rf"\b(?:no|without|forget|drop|skip|not)(?: the| an?| any)? {spoken}", text)
        target = dropped if refused else found

        if feature not in target:
            target.append(feature)

    if found:
        slots["features"] = found
        slots["only_features"] = bool(re.search(r"\b(?:instead|only|just)\b", text))

    if dropped:
        slots["dropped_features"] = dropped

    room = re.search(rf"\b({'|'.join(room_ids)})\b", text)

    if room:
        slots["room"] = room[1]

    # "No, don't book it" also contains a yes cue, so a refusal is looked for first.
    if NO.search(text):
        slots["confirm"] = "no"

    elif YES.search(text):
        slots["confirm"] = "yes"

    if NOTHING.search(text):
        slots["nothing"] = True

    slots["mode"] = "add" if ADD.search(text) else "set"

    mentions = [kind for kind in ("timers", "reminders") if kind[:-1] in text]
    slots["what"] = mentions[0] if len(mentions) == 1 else "all"

    name = NAME.search(transcript)

    if name:
        slots["name"] = name[1].strip(" .")

    return slots
