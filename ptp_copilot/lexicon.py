"""Code-mixed (Hinglish / Kanglish / English) lexicons for in-call signal detection.

Patterns are written against romanised ASR output, are case-insensitive and
tolerant to the common ASR artefacts seen in the corpus (dropped words,
"[inaudible]" tokens, filler prefixes such as "umm", "matlab", "andre").
Every pattern family is a *signal*, never a verdict: the model combines them
with account and acoustic evidence.
"""
import re

FLAGS = re.IGNORECASE


def _rx(*parts):
    return re.compile(r"(?:" + "|".join(parts) + r")", FLAGS)


# ------------------------------------------------------------------ customer signals
HEDGE = _rx(
    r"\bdekhta\b", r"dekh(?:te|ke) (?:hain|batata)", r"\btry kar(?:unga|ungi|enge)", r"\bkoshish",
    r"\bshayad\b", r"\bmaybe\b", r"let me see", r"\bi will try\b", r"try my best",
    r"\bnodtini\b", r"\bnodona\b", r"try maadtini", r"prayatna", r"kuch karta hoon", r"\bsee what i can\b",
    r"\bprobably\b", r"\bdekhenge\b", r"\benaadru maadtini\b",
)
EXIT = _rx(
    r"baad (?:mein|me) (?:call|baat)", r"abhi busy", r"meeting (?:mein|nalli|me)", r"\bi am in a meeting",
    r"rakhta hoon", r"have to go", r"call (?:me )?later", r"\bamele (?:call|maataadi)", r"phone idtini",
    r"\bdriving\b", r"gaadi chala", r"bahar hoon", r"shaam ko call", r"kal call karna", r"naale call",
    r"horage idini", r"busy idini", r"i am busy", r"talk later", r"\bbye\b.*\bbye\b",
)
CONDITIONAL = _rx(
    r"salary aayi toh", r"paise aaye toh", r"\bagar\b.*\btoh\b", r"if (?:the )?money comes", r"if my salary",
    r"\bif (?:i|he|she) get", r"party (?:payment )?kottre", r"party ne payment de diya", r"salary bandre",
    r"duddu bandre", r"\bonce (?:my )?salary", r"\bif the client",
)
VAGUE_DATE = _rx(
    r"next week", r"agle hafte", r"month end", r"mahine ke (?:end|aakhir)", r"mundina vaara", r"kuch din",
    r"\bsoon\b", r"\bjaldi\b", r"few days", r"\bthis month\b", r"is mahine (?:mein|me) (?:kabhi|kar)",
)
FIRM = _rx(
    r"\bpakka\b", r"for sure", r"\bdefinitely\b", r"khud kar dunga", r"do it myself", r"confirm (?:maadtini|ho jayega)",
    r"naane maadtini", r"clear (?:kar dunga|maadtini)", r"\bpoora\b.*(?:dunga|kodtini|clear)", r"pay the full",
    r"\bguarantee", r"i will definitely",
)
PARTIAL_OFFER = _rx(
    r"kar sakta hoon,? baaki", r"de dunga,? poora abhi nahi", r"can give .* not the full", r"can pay .* now.* rest later",
    r"kodbahudu,? baaki", r"kodtini,? poora eega aagalla", r"\bpart payment\b",
)
HARDSHIP = _rx(
    r"salary (?:abhi tak )?(?:aayi nahi|late|nahi aayi|hold|innu bandilla|has not come)", r"salary is late",
    r"salary late aagide", r"company ne (?:salary )?hold", r"job (?:chali gayi|lost|hogide)", r"lost my job",
    r"company band", r"kaam nahi hai", r"medical emergency", r"\bhospital\b", r"\badmit", r"business (?:bahut )?down",
    r"business (?:is )?(?:very )?slow", r"business tumba", r"sale (?:hi )?nahi ho rahi", r"no sales", r"vyapara illa",
    r"client has not paid", r"party ke paas", r"party inda payment bandilla", r"payment phasa", r"paise nahi hain",
    r"duddu illa", r"don'?t have money", r"no money", r"balance nahi tha", r"no balance", r"balance irlilla",
    r"duddu ella alli", r"all the money went", r"\bkharcha\b", r"kharchu",
)
SEVERE_HARDSHIP = _rx(
    r"medical emergency", r"\bhospital\b", r"\badmit", r"job (?:chali gayi|lost|hogide)", r"lost my job",
    r"company band", r"kaam nahi hai", r"\bdeath\b", r"guzar ga", r"\baccident\b", r"all the money went",
    r"saare paise wahan", r"duddu ella alli",
)
THIRD_PARTY = _rx(
    r"main unk[ia] (?:wife|bhai|behen|mummy|mummi|maa|beta|beti|papa|pati|colleague|dost)",
    r"unki (?:wife|mummy) (?:bol rahi|hoon)", r"main beta bol", r"main (?:unka )?bhai bol",
    r"this is (?:his|her) (?:wife|husband|mother|mothe|father|brother|sister|son|daughter|colleague)",
    r"i am (?:his|her) (?:brother|brothe|sister|wife|husband|mother|father|son|colleague)",
    r"naanu avra (?:amma|thamma|hendathi|appa|akka|anna|maga|magalu|ganda)", r"avra (?:amma|thamma|hendathi)",
    r"papa abhi", r"ghar pe nahi hain", r"not at home", r"maneli illa", r"kaam pe gaya", r"she is at work",
    r"he is at work", r"kelsakke hogid", r"horage hogidaare", r"bahar gaye hain", r"office colleague",
)
RELAY = _rx(
    r"(?:bata|bol|keh) dunga", r"bol dungi", r"bata dungi", r"i will tell (?:him|her)", r"let (?:him|her) know",
    r"heltini", r"message kodtini", r"unko bol", r"main bata", r"\bwill pass\b",
)
WRONG_NUMBER = _rx(r"wrong number", r"is naam ka koi nahi", r"doesn'?t know", r"no one by that name", r"yaaru gottilla")
LINK_ACCEPT = _rx(r"bhej do link", r"link bhej (?:do|dijiye)", r"send the link", r"link kalsi", r"\bbhej do\b", r"yes,? the link")
LINK_REFUSE = _rx(r"link mat bhejo", r"don'?t send a link", r"link beda", r"branch (?:jaake|ge hogi)", r"pay at the branch")
PRIOR_BREAK_EXCUSE = _rx(
    r"alag baat thi", r"that was different", r"adu bere", r"us time problem", r"was a problem then",
    r"aaga problem", r"pata hai,? sorry", r"\bsorry\b",
)
FORGOT = _rx(r"dhyan se nikal", r"i forgot", r"marethu hoytu", r"bhool gaya")
REFUSAL = _rx(r"abhi nahi ho payega", r"not possible", r"eega aagalla", r"nahi ho payega", r"\bcan'?t pay\b", r"nahi dunga")
DISPUTE = _rx(r"already paid", r"pehle hi pay", r"maine bhar diya", r"galat hai", r"not my loan", r"kattiddini")
MIN_ACK = re.compile(
    r"^(?:\W*(?:umm|hmm|haan|han|ha|ji|ok|okay|ok ok|theek|thik|hai|sari|aytu|yeah|yes|fine|achha|acha|"
    r"ho|jayega|haudu|alright|sure|kar|dunga|maadtini|aagutte|matlab|like|andre|so|right|bolo)\W*)+$",
    FLAGS,
)

# ------------------------------------------------------------------ agent / bot signals
AGENT_PROPOSE_DATE = _rx(
    r"\btoh .{0,20}\btak kar denge", r"tak ho jayega na", r"can you pay by", r"\bwill .{0,15} work\b",
    r"\bge aagutta\b", r"olage maadtira", r"\bso can you pay by", r"tak kar doge",
)
AGENT_OPEN_DATE_Q = _rx(
    r"kab tak", r"by when", r"yaavaga", r"by which date", r"kis tareekh", r"when will you", r"yaava dinaank",
)
AGENT_PRESSURE = _rx(
    r"cibil", r"credit score", r"escalate", r"last reminde", r"case aage", r"case munde", r"case moves ahead",
    r"\blegal\b", r"aaj hi karna padega", r"ivattu maadle beku", r"you need to pay today",
)
AGENT_RECORD = _rx(
    r"ptp note", r"record kar", r"noting a promise", r"recording it", r"confirm kar raha", r"confirming",
    r"confirm maadtiddini", r"note kar leta", r"note maadkoltini", r"make a note", r"note kar raha",
)
AGENT_PRIOR_BREAK = _rx(
    r"pehle bhi", r"pichli baar", r"last time also", r"you had promised", r"kaleda sala", r"modalu kooda",
)
AGENT_ASK_SALARY = _rx(r"salary kab", r"salary kis", r"when does your salary", r"salary yaava", r"salary yaavaga",
                      r"what date do you get your salary", r"kab credit hoti")
AGENT_PARTIAL_ASK = _rx(r"kam se kam", r"at least", r"kanishta", r"abhi .{0,15}kar sakte", r"can you do .* now",
                        r"eega .{0,20}maadtira", r"today and the rest", r"aaj .{0,20}kar dijiye,? baaki",
                        r"ivattu .{0,20}baaki")
AGENT_3P_MESSAGE = _rx(r"message de sakte", r"pass a message", r"bata dijiye ki", r"avrige .{0,40}heli",
                       r"tell him a call", r"message kodtira", r"call bandittu antha")

# ------------------------------------------------------------------ dates / amounts
MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
DATE_EXPLICIT = re.compile(
    r"\b(\d{1,2})\s*(?:st|nd|rd|th)?\s*(?:tareekh(?:u)?|ne tareekhu|tarik|" + MONTHS + r")\b"
    r"|\bthe\s*(\d{1,2})\s*(?:st|nd|rd|th)\b|\bon the (\d{1,2})"
    r"|\b(\d{1,2})\s*(?:tak|ko|ge|olage)\b",
    FLAGS,
)
DATE_RELATIVE = _rx(r"\bkal\b", r"\bnaale\b", r"tomorrow", r"\bparson\b", r"\bnaadidu\b",
                    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday",
                    r"somvar|mangalvar|budhvar|guruvar|shukravar|shanivar|ravivar")
DATE_TODAY = _rx(r"\baaj\b", r"\bivattu\b", r"\btoday\b", r"\babhi\b", r"\beega\b", r"right now")
SALARY_DAY = re.compile(
    r"salary\s*(\d{1,2})\s*(?:tak|ko)|(\d{1,2})\s*(?:tareekh ko aati|ne tareekhu barutte|ge barutte|ko aa jaati)"
    r"|comes on the\s*(\d{1,2})|every month on the\s*(\d{1,2})|har mahine\s*(\d{1,2})|prati tingalu\s*(\d{1,2})",
    FLAGS,
)
AMOUNT = re.compile(r"(?:₹|rs\.?\s?)\s?([\d,]{3,})|([\d,]{3,})\s*(?:rupaye|rupees|rs)\b", FLAGS)


def first_int(m):
    for g in m.groups():
        if g:
            try:
                return int(g)
            except ValueError:
                return None
    return None


def extract_days(text):
    """Day-of-month numbers explicitly mentioned (1..31)."""
    out = []
    for m in DATE_EXPLICIT.finditer(text):
        v = first_int(m)
        if v is not None and 1 <= v <= 31:
            out.append(v)
    return out


def extract_amounts(text):
    out = []
    for m in AMOUNT.finditer(text):
        g = m.group(1) or m.group(2)
        try:
            v = float(g.replace(",", ""))
        except ValueError:
            continue
        if v >= 100:
            out.append(v)
    return out


def extract_salary_day(text):
    m = SALARY_DAY.search(text)
    if m:
        v = first_int(m)
        if v and 1 <= v <= 31:
            return v
    return None


CUSTOMER_FAMILIES = dict(
    hedge=HEDGE, exit=EXIT, conditional=CONDITIONAL, vague_date=VAGUE_DATE, firm=FIRM,
    partial_offer=PARTIAL_OFFER, hardship=HARDSHIP, severe_hardship=SEVERE_HARDSHIP,
    third_party=THIRD_PARTY, relay=RELAY, wrong_number=WRONG_NUMBER, link_accept=LINK_ACCEPT,
    link_refuse=LINK_REFUSE, prior_break_excuse=PRIOR_BREAK_EXCUSE, forgot=FORGOT, refusal=REFUSAL,
    dispute=DISPUTE,
)
AGENT_FAMILIES = dict(
    propose_date=AGENT_PROPOSE_DATE, open_date_q=AGENT_OPEN_DATE_Q, pressure=AGENT_PRESSURE,
    record=AGENT_RECORD, prior_break=AGENT_PRIOR_BREAK, ask_salary=AGENT_ASK_SALARY,
    partial_ask=AGENT_PARTIAL_ASK, tp_message=AGENT_3P_MESSAGE,
)


def clean(text):
    return re.sub(r"\[(?:inaudible|name|lender)\]", " ", str(text), flags=FLAGS)


def is_min_ack(text):
    t = re.sub(r"[^\w\s]", " ", clean(text)).strip()
    return bool(t) and len(t.split()) <= 5 and bool(MIN_ACK.match(t))
