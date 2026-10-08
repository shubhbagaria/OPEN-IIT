# Live copilot replays (held-out test calls)

Each row is the copilot's state immediately after that utterance. `heard_enough` marks the first point where the score is stable and a commitment event has occurred.


## PTP000002 - ground truth `escape` - outcome: NOT KEPT (broken), lang=english

| # | speaker | utterance | p_keep | category (conf) | heard enough | copilot suggestion |
|---|---|---|---|---|---|---|
| 0 | bot | Hello, this is an automated call from [LENDER]. Am I speaking with [NAME]? | 0.10 | genuine_infeasible (0.80) |  |  |
| 1 | customer | Yes, speaking. | 0.12 | escape (0.84) |  | (Listen - ask the customer to state their own date: 'Which date works for you?') |
| 2 | bot | An amount of Rs 12,200 has been due on your account for 23 days. | 0.12 | escape (0.85) |  |  |
| 3 | customer | I forgot about it. | 0.10 | escape (0.90) |  | (Listen - ask the customer to state their own date: 'Which date works for you?') |
| 4 | bot | By which date can you make the payment? | 0.10 | escape (0.91) |  |  |
| 5 | customer | i will do it myself on next week, for sure. okay okay, i will pay, i have to go. | 0.05 | escape (0.98) | **YES** | Just one minute please. I am sending a ₹2,000 link right now; paying it now updates your account and reduces follow-up calls. The rest by 08 Apr. Is that okay? |
| 6 | bot | Okay, Rs 12,200 on 8 April, I am recording it. | 0.05 | escape (0.99) |  |  |
| 7 | bot | You will get a payment link on this number. | 0.05 | escape (0.99) |  |  |
| 8 | customer | Don't send a link I will do it myself | 0.06 | escape (0.98) |  | Just one minute please. I am sending a ₹2,000 link right now; paying it now updates your account and reduces follow-up calls. The rest by 08 Apr. Is that okay? |
| 9 | bot | Thanks for your time. | 0.06 | escape (0.98) |  |  |

## PTP000012 - ground truth `agent_recorded_or_pushed` - outcome: NOT KEPT (kept_late), lang=hinglish

| # | speaker | utterance | p_keep | category (conf) | heard enough | copilot suggestion |
|---|---|---|---|---|---|---|
| 0 | agent | umm Hello, [LENDER] collections team se bol rahi hoon. [NAME] ji bol rahe hain? | 0.39 | genuine_infeasible (0.63) |  |  |
| 1 | customer | Haan main hi hoon kya baat hai | 0.39 | genuine_feasible (0.66) |  | (Listen - customer se payment ki date unke shabdon mein poochhiye: 'Aap kis tareekh tak kar paayenge?') |
| 2 | agent | Aapke loan account mein ₹32,200 overdue hai, 58 din ho gaye hain. | 0.39 | genuine_infeasible (0.62) |  |  |
| 3 | agent | Aapne pehle bhi 5 February ka promise kiya tha, woh bhi miss ho gaya. | 0.38 | genuine_infeasible (0.60) |  |  |
| 4 | customer | Haan pata hai sorry | 0.38 | genuine_infeasible (0.62) |  | (Listen - customer se payment ki date unke shabdon mein poochhiye: 'Aap kis tareekh tak kar paayenge?') |
| 5 | customer | matlab Arre dhyan se nikal gaya. | 0.37 | genuine_infeasible (0.47) |  | (Listen - customer se payment ki date unke shabdon mein poochhiye: 'Aap kis tareekh tak kar paayenge?') |
| 6 | agent | Aapki salary kab credit hoti hai | 0.37 | genuine_infeasible (0.48) |  |  |
| 7 | customer | Har [inaudible] 28 ko aa jaati hai. | 0.36 | genuine_infeasible (0.51) |  | (Listen - customer se payment ki date unke shabdon mein poochhiye: 'Aap kis tareekh tak kar paayenge?') |
| 8 | agent | sir aaj hi karna padega, warna account aage escalate ho jayega. | 0.36 | agent_recorded_or_pushed (0.56) |  |  |
| 9 | agent | Aap payment kab tak kar payenge? | 0.34 | agent_recorded_or_pushed (0.55) |  |  |
| 10 | customer | umm salary aane ke baad ko pakka kar dunga. ₹16,100 de dunga, poora abhi nahi hoga. | 0.56 | agent_recorded_or_pushed (0.59) |  | Sir, main confirm karna chahta hoon - kya aap khud 26 Apr ko ₹16,100 pay kar payenge? Agar nahi, toh aap apni suvidha ki date bataiye. |
| 11 | agent | Aaj 16,100 rupaye kar dijiye, baaki 26 April tak. | 0.56 | agent_recorded_or_pushed (0.60) | **YES** |  |
| 12 | customer | 16,100 rupaye de dunga, poora abhi nahi hoga. | 0.56 | agent_recorded_or_pushed (0.60) |  | Sir, main confirm karna chahta hoon - kya aap khud 26 Apr ko ₹16,100 pay kar payenge? Agar nahi, toh aap apni suvidha ki date bataiye. |
| 13 | agent | Ok main follow up kar lunga | 0.55 | agent_recorded_or_pushed (0.60) |  |  |
| 14 | agent | Main aapko WhatsApp pe payment link bhej deta hoon. | 0.53 | agent_recorded_or_pushed (0.59) |  |  |
| 15 | customer | Nahi nahi, main branch jaake kar dunga. | 0.53 | agent_recorded_or_pushed (0.59) |  | Sir, main confirm karna chahta hoon - kya aap khud 26 Apr ko ₹16,100 pay kar payenge? Agar nahi, toh aap apni suvidha ki date bataiye. |
| 16 | agent | umm Shukriya, time dene ke liye. | 0.53 | agent_recorded_or_pushed (0.59) |  |  |
| 17 | customer | Haan, thank you. | 0.53 | agent_recorded_or_pushed (0.58) |  | Sir, main confirm karna chahta hoon - kya aap khud 26 Apr ko ₹16,100 pay kar payenge? Agar nahi, toh aap apni suvidha ki date bataiye. |

## PTP000146 - ground truth `third_party` - outcome: NOT KEPT (broken), lang=hinglish

| # | speaker | utterance | p_keep | category (conf) | heard enough | copilot suggestion |
|---|---|---|---|---|---|---|
| 0 | agent | Good morning sir main [LENDER] se call kar raha hoon kya [NAME] ji line pe hain | 0.15 | genuine_infeasible (0.35) |  |  |
| 1 | customer | Main [inaudible] mummy hoon woh kaam pe gaya hai | 0.19 | third_party (0.97) |  | Koi baat nahi. [NAME] ji se baat karne ka sahi samay kya rahega? Main tab dobara call kar loonga. Dhanyavaad. |
| 2 | agent | Kya aap unko message de sakte hain ki [LENDER] se call aaya tha? | 0.19 | third_party (0.93) | **YES** |  |
| 3 | customer | main bata dunga 6 tareekh tak ho jayega. | 0.18 | third_party (0.99) |  | Koi baat nahi. [NAME] ji se baat karne ka sahi samay kya rahega? Main tab dobara call kar loonga. Dhanyavaad. |
| 4 | agent | Confirm kar raha hoon, 6 April tak ₹30,900. | 0.18 | third_party (0.99) |  |  |
| 5 | agent | dhanyavaad, aapka shubh ho. | 0.18 | third_party (1.00) |  |  |

## PTP000101 - ground truth `genuine_infeasible` - outcome: NOT KEPT (broken), lang=hinglish

| # | speaker | utterance | p_keep | category (conf) | heard enough | copilot suggestion |
|---|---|---|---|---|---|---|
| 0 | agent | Hello [LENDER] collections team se bol rahi hoon [NAME] ji bol rahe hain | 0.41 | third_party (0.61) |  |  |
| 1 | customer | haan main hi hoon kya baat hai | 0.41 | third_party (0.66) |  | Koi baat nahi. [NAME] ji se baat karne ka sahi samay kya rahega? Main tab dobara call kar loonga. Dhanyavaad. |
| 2 | agent | Aapki EMI ₹46,950 bounce ho gayi thi, abhi tak payment nahi aaya. | 0.41 | third_party (0.64) |  |  |
| 3 | customer | Account mein balance nahi tha us din. | 0.41 | genuine_feasible (0.68) |  | (Listen - customer se payment ki date unke shabdon mein poochhiye: 'Aap kis tareekh tak kar paayenge?') |
| 4 | agent | Salary kis tareekh ko aati hai aapki? | 0.41 | third_party (0.64) |  |  |
| 5 | customer | Har mahine 30 ko aa jaati hai. | 0.41 | genuine_feasible (0.69) |  | (Listen - customer se payment ki date unke shabdon mein poochhiye: 'Aap kis tareekh tak kar paayenge?') |
| 6 | agent | Kab tak clear kar denge aap | 0.41 | genuine_feasible (0.69) |  |  |
| 7 | customer | 27 tareekh tak confirm ho jayega. poora ₹46,950 clear kar dunga. | 0.61 | genuine_feasible (0.91) |  | Bahut badhiya, 27 Apr ko ₹46,950 confirm hai. Main abhi WhatsApp pe payment link bhej raha hoon, ek din pehle reminder bhi aayega. |
| 8 | agent | poora nahi toh kam se kam 27,200 rupaye aaj kar dijiye. | 0.61 | genuine_feasible (0.91) | **YES** |  |
| 9 | customer | abhi 27,200 rupaye kar sakta hoon, baaki baad mein. | 0.61 | genuine_feasible (0.91) |  | Bahut badhiya, 27 Apr ko ₹46,950 confirm hai. Main abhi WhatsApp pe payment link bhej raha hoon, ek din pehle reminder bhi aayega. |
| 10 | agent | Ok, main follow up kar lunga. | 0.61 | genuine_feasible (0.91) |  |  |
| 11 | agent | main aapko whatsapp pe payment link bhej deta hoon. | 0.61 | genuine_feasible (0.91) |  |  |
| 12 | customer | haan bhej do link | 0.61 | genuine_feasible (0.92) |  | Bahut badhiya, 27 Apr ko ₹46,950 confirm hai. Main abhi WhatsApp pe payment link bhej raha hoon, ek din pehle reminder bhi aayega. |
| 13 | agent | achha Shukriya time dene ke liye | 0.61 | genuine_feasible (0.92) |  |  |
| 14 | customer | ok. | 0.60 | genuine_feasible (0.91) |  | Bahut badhiya, 27 Apr ko ₹46,950 confirm hai. Main abhi WhatsApp pe payment link bhej raha hoon, ek din pehle reminder bhi aayega. |

## PTP000324 - ground truth `repeat_promiser` - outcome: NOT KEPT (broken), lang=english

| # | speaker | utterance | p_keep | category (conf) | heard enough | copilot suggestion |
|---|---|---|---|---|---|---|
| 0 | agent | Good morning, this is the [LENDER] collections team. Is this [NAME]? | 0.21 | genuine_infeasible (0.59) |  |  |
| 1 | customer | Yes tell me | 0.23 | genuine_infeasible (0.58) |  | (Listen - ask the customer to state their own date: 'Which date works for you?') |
| 2 | agent | Your EMI of Rs 16,250 bounced and we have not received payment yet. | 0.23 | genuine_infeasible (0.58) |  |  |
| 3 | agent | When will you be able to make the payment? | 0.23 | genuine_infeasible (0.57) |  |  |
| 4 | customer | like i [inaudible] try something by the 12th. | 0.23 | genuine_infeasible (0.63) | **YES** | I understand this is a difficult time. Shall we move the date to 9th, after your salary, or would a small ₹2,000 today be manageable? I can also check a restructuring option for you. |
| 5 | agent | Confirming Rs 16,250 by 12 April. | 0.23 | genuine_infeasible (0.65) |  |  |
| 6 | agent | You will get a payment link on this number. | 0.22 | genuine_infeasible (0.72) |  |  |

## PTP000026 - ground truth `genuine_feasible` - outcome: KEPT (kept_on_time), lang=hinglish

| # | speaker | utterance | p_keep | category (conf) | heard enough | copilot suggestion |
|---|---|---|---|---|---|---|
| 0 | agent | Namaste main [LENDER] se baat kar raha hoon Kya meri baat [NAME] ji se ho rahi hai | 0.21 | genuine_infeasible (0.27) |  |  |
| 1 | customer | umm ji haan, boliye. | 0.32 | genuine_infeasible (0.29) |  | (Listen - customer se payment ki date unke shabdon mein poochhiye: 'Aap kis tareekh tak kar paayenge?') |
| 2 | agent | Aapke loan account mein 6,200 rupaye overdue hai, 46 din ho gaye hain. | 0.32 | genuine_infeasible (0.29) |  |  |
| 3 | agent | pichli baar bhi aapne 20 january ka bola tha, payment nahi aaya. | 0.31 | genuine_infeasible (0.28) |  |  |
| 4 | customer | Haan us time problem ho gayi thi. | 0.30 | genuine_infeasible (0.27) |  | (Listen - customer se payment ki date unke shabdon mein poochhiye: 'Aap kis tareekh tak kar paayenge?') |
| 5 | customer | Haan pata hai, ho jayega. | 0.30 | genuine_infeasible (0.33) |  | (Listen - customer se payment ki date unke shabdon mein poochhiye: 'Aap kis tareekh tak kar paayenge?') |
| 6 | agent | Aapka CIBIL kharab ho raha hai, aaj hi kar dijiye. | 0.30 | agent_recorded_or_pushed (0.92) |  |  |
| 7 | customer | Koshish | 0.21 | agent_recorded_or_pushed (0.94) |  | (Listen - customer se payment ki date unke shabdon mein poochhiye: 'Aap kis tareekh tak kar paayenge?') |
| 8 | agent | aapka cibil kharab ho raha hai aaj hi kar dijiye | 0.21 | agent_recorded_or_pushed (0.94) |  |  |
| 9 | agent | kab tak clear kar denge aap | 0.20 | agent_recorded_or_pushed (0.94) |  |  |
| 10 | customer | Meeting mein hoon, baad mein call karo. | 0.13 | agent_recorded_or_pushed (0.94) |  | (Listen - customer se payment ki date unke shabdon mein poochhiye: 'Aap kis tareekh tak kar paayenge?') |
| 11 | agent | 2 April tak ho jayega na? | 0.12 | agent_recorded_or_pushed (0.94) | **YES** |  |
| 12 | customer | Theek hai... dekhta hoon. | 0.09 | agent_recorded_or_pushed (0.98) |  | Sir, main confirm karna chahta hoon - kya aap khud 02 Apr ko ₹6,200 pay kar payenge? Agar nahi, toh aap apni suvidha ki date bataiye. |
| 13 | agent | main link bhej hoon abhi ₹1900 kar sakte hain kya | 0.09 | agent_recorded_or_pushed (0.98) |  |  |
| 14 | customer | abhi busy hoon, baad mein baat karte hain. | 0.09 | agent_recorded_or_pushed (0.98) |  | Sir, main confirm karna chahta hoon - kya aap khud 02 Apr ko ₹6,200 pay kar payenge? Agar nahi, toh aap apni suvidha ki date bataiye. |
| 15 | agent | Main link bhej deta hoon, abhi ₹1,900 kar sakte hain kya? | 0.09 | agent_recorded_or_pushed (0.98) |  |  |
| 16 | customer | Theek hai theek hai, dunga, rakhta hoon. | 0.09 | agent_recorded_or_pushed (0.98) |  | Sir, main confirm karna chahta hoon - kya aap khud 02 Apr ko ₹6,200 pay kar payenge? Agar nahi, toh aap apni suvidha ki date bataiye. |
| 17 | agent | Poora nahi toh kam se kam ₹1,900 aaj kar dijiye. | 0.09 | agent_recorded_or_pushed (0.98) |  |  |
| 18 | customer | haan koshish karunga. | 0.09 | agent_recorded_or_pushed (0.98) |  | Sir, main confirm karna chahta hoon - kya aap khud 02 Apr ko ₹6,200 pay kar payenge? Agar nahi, toh aap apni suvidha ki date bataiye. |
| 19 | agent | Toh main 2 April ke liye ₹6,200 ka PTP note kar raha hoon. | 0.09 | agent_recorded_or_pushed (0.98) |  |  |
| 20 | agent | Dhanyavaad, aapka din shubh ho. | 0.09 | agent_recorded_or_pushed (0.98) |  |  |
| 21 | customer | haan Ok | 0.09 | agent_recorded_or_pushed (0.98) |  | Sir, main confirm karna chahta hoon - kya aap khud 02 Apr ko ₹6,200 pay kar payenge? Agar nahi, toh aap apni suvidha ki date bataiye. |