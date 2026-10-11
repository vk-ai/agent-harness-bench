"""The task set: 4 categories, 32 tasks. Expected answers are checked by substring."""
from __future__ import annotations

import random

# --- tool use: parallel (independent calls) and sequential (dependent calls)
PARALLEL = [
    ("What's the weather in Pune and in Leeds right now?", ["31", "12"]),
    ("Compare the temperature in Austin and Osaka.", ["29", "22"]),
    ("Give me the weather for Lisbon, Nairobi and Denver.", ["24", "25", "8"]),
    ("What are the statuses of orders A-4417 and B-9032?", ["shipped", "processing"]),
    ("Convert 100 USD to EUR and also 100 USD to GBP.", ["92", "79"]),
    ("Weather in Leeds, and the status of order C-1185?", ["12", "delivered"]),
    ("Use the calculator for 17*23 and for 1234-567.", ["391", "667"]),
    ("Is it warmer in Pune or Denver? Check both.", ["31", "8"]),
]
SEQUENTIAL = [
    ("What is the total of order B-9032 in USD?", ["82.07"]),
    ("How much is order C-1185 worth in EUR?", ["37.69"]),
    ("Look up order A-4417 and convert its total to INR.", ["9960"]),
    ("What is order A-4417's total times 3? Use the calculator.", ["360"]),
    ("Convert 50 GBP to JPY.", ["9493.67"]),
    ("Calculate (12.5+7.5)*4 then convert that many USD to EUR.", ["73.6"]),
    ("What's the status of order B-9032?", ["processing"]),
    ("What is 2 to the power of 10? Use the calculator.", ["1024"]),
]

# --- structured output (with repair-retry)
INVOICE = {"type": "object", "additionalProperties": False,
           "required": ["vendor", "total", "currency", "date", "paid"],
           "properties": {"vendor": {"type": "string"}, "total": {"type": "number"},
                          "currency": {"type": "string", "enum": ["USD", "EUR", "GBP", "INR"]},
                          "date": {"type": "string", "pattern": r"\d{4}-\d{2}-\d{2}"},
                          "paid": {"type": "boolean"}}}
STRUCTURED = [
    ("Invoice from Acme Tools dated 3 March 2026, total 1,250.00 euros, already paid.",
     {"vendor": "Acme Tools", "total": 1250.0, "currency": "EUR", "date": "2026-03-03", "paid": True}),
    ("Bill: Northwind Traders. Amount due $89.99. Issued 2026-07-15. Status: unpaid.",
     {"vendor": "Northwind Traders", "total": 89.99, "currency": "USD", "date": "2026-07-15", "paid": False}),
    ("Received invoice #22 from Globex on the 1st of January 2026 for £430; settled by card.",
     {"vendor": "Globex", "total": 430.0, "currency": "GBP", "date": "2026-01-01", "paid": True}),
    ("Rs. 15,000 invoice, Initech Pvt Ltd, 28/02/2026, payment pending.",
     {"vendor": "Initech Pvt Ltd", "total": 15000.0, "currency": "INR", "date": "2026-02-28", "paid": False}),
    ("Umbrella Corp charged us USD 2,000 on Dec 5 2025. We paid it the same day.",
     {"vendor": "Umbrella Corp", "total": 2000.0, "currency": "USD", "date": "2025-12-05", "paid": True}),
    ("Stark Industries invoice: EUR 310.5, date 2026-09-30, not yet paid.",
     {"vendor": "Stark Industries", "total": 310.5, "currency": "EUR", "date": "2026-09-30", "paid": False}),
    ("Hooli sent a GBP 75 invoice on 2026-04-10 which is paid.",
     {"vendor": "Hooli", "total": 75.0, "currency": "GBP", "date": "2026-04-10", "paid": True}),
    ("Wayne Enterprises — INR 999 — 2026-06-06 — outstanding.",
     {"vendor": "Wayne Enterprises", "total": 999.0, "currency": "INR", "date": "2026-06-06", "paid": False}),
]

# --- long context with a needle; asked with and without compression
_CHAT = ["Thanks!", "Can you check the delivery window?", "The app was slow today.", "OK.",
         "Do you have gift cards?", "My friend says hi.", "Is support open on weekends?", "Great, noted.",
         "I may order again next week.", "What's your returns policy?"]
_FACTS = [("My account number is QX-5521.", "What is my account number?", "QX-5521"),
          ("Please ship everything to my office in Leeds.", "Which city should my order ship to?", "Leeds"),
          ("My budget is 450 dollars, no more.", "What budget did I give you?", "450"),
          ("I'm allergic to peanuts.", "What allergy did I mention?", "peanut"),
          ("My daughter's name is Mira.", "What's my daughter's name?", "Mira"),
          ("I want the blue model, not the red one.", "Which colour did I choose?", "blue"),
          ("My delivery code is 7781.", "What's my delivery code?", "7781"),
          ("I'm on the Premium plan.", "Which plan am I on?", "Premium")]


def long_history(i: int, turns: int = 30):
    rng = random.Random(i)
    fact, q, ans = _FACTS[i]
    h = []
    pos = rng.randrange(1, 5)
    for t in range(turns):
        u = fact if t == pos else rng.choice(_CHAT) + " " + " ".join(rng.choice(_CHAT) for _ in range(3))
        h += [{"role": "user", "content": u}, {"role": "assistant", "content": "Sure, happy to help with that."}]
    return h, q, ans


COMPRESSION = list(range(len(_FACTS)))


def check(answer: str, expected: list[str]) -> bool:
    a = (answer or "").lower().replace(",", "")
    return all(e.lower().replace(",", "") in a for e in expected)
