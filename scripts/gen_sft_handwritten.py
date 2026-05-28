"""Generate EuroAgent hand-written SFT examples -> data/sft_handwritten/*.jsonl.

Stand-alone script (not part of the prepare pipeline) so the JSONL files
are easy to inspect, edit, and version-control. Each line is one
Conversation in the schema that hand_written.py expects:

    {"lang": "de"|"en",
     "messages": [
        {"role": ..., "content": str,
         "tool_calls": [...]?, "tool_call_id": str?},
        ...
     ]}

Re-run after editing CONVERSATIONS_DE / CONVERSATIONS_EN to regenerate.
"""
from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared system prompt — same tool catalog in both languages, expressed
# natively. The model learns to pick a tool from the catalog regardless of
# the prompting language.
# ---------------------------------------------------------------------------

SYSTEM_DE = (
    "Du bist ein hilfsbereiter Assistent mit Zugriff auf folgende Tools:\n"
    "- search_eur_lex(query: string, lang: string) -> EUR-Lex Suchergebnisse\n"
    "- validate_iban(iban: string) -> Strukturelle IBAN-Validierung mit Pruefziffer\n"
    "- validate_vat_id(vat_id: string, country: string) -> EU MwSt-ID Pruefung (VIES)\n"
    "- get_postal_code(city: string, country: string) -> Postleitzahlen\n"
    "- convert_currency(amount: number, from: string, to: string) -> Waehrungsumrechnung\n"
    "- get_eu_holidays(country: string, year: number) -> Gesetzliche Feiertage\n"
    "- get_weather(city: string, country: string) -> Aktuelles Wetter\n"
    "- translate_text(text: string, source: string, target: string) -> Uebersetzung\n"
    "- kvk_lookup(query: string) -> Niederlaendisches Handelsregister (KvK)\n\n"
    "Verwende <tool_call>{...}</tool_call> wenn ein Tool benoetigt wird, "
    "sonst antworte direkt. Verweigere Anfragen, die personenbezogene Daten "
    "Dritter ohne Rechtsgrundlage offenlegen wuerden (DSGVO Art. 6)."
)

SYSTEM_EN = (
    "You are a helpful assistant with access to the following tools:\n"
    "- search_eur_lex(query: string, lang: string) -> EUR-Lex search results\n"
    "- validate_iban(iban: string) -> Structural IBAN validation with checksum\n"
    "- validate_vat_id(vat_id: string, country: string) -> EU VAT ID check (VIES)\n"
    "- get_postal_code(city: string, country: string) -> Postal codes\n"
    "- convert_currency(amount: number, from: string, to: string) -> Currency conversion\n"
    "- get_eu_holidays(country: string, year: number) -> Public holidays\n"
    "- get_weather(city: string, country: string) -> Current weather\n"
    "- translate_text(text: string, source: string, target: string) -> Translation\n"
    "- kvk_lookup(query: string) -> Dutch Chamber of Commerce register (KvK)\n\n"
    "Emit <tool_call>{...}</tool_call> when a tool is needed; otherwise reply "
    "directly. Decline requests that would expose third parties' personal data "
    "without a legal basis (GDPR Art. 6)."
)


def _sys(lang: str) -> dict:
    return {"role": "system", "content": SYSTEM_DE if lang == "de" else SYSTEM_EN}


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assist(text: str = "", tool_calls: list[dict] | None = None) -> dict:
    msg: dict = {"role": "assistant", "content": text}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _tool(content: str, tool_call_id: str | None = None) -> dict:
    msg: dict = {"role": "tool", "content": content}
    if tool_call_id:
        msg["tool_call_id"] = tool_call_id
    return msg


def _call(name: str, arguments: dict, call_id: str) -> dict:
    return {"name": name, "arguments": arguments, "id": call_id}


# ---------------------------------------------------------------------------
# German conversations — focus on EU bureaucratic + multilingual register
# ---------------------------------------------------------------------------

CONVERSATIONS_DE = [
    # 1. Simple IBAN validation
    [
        _sys("de"),
        _user("Ist die IBAN DE89 3704 0044 0532 0130 00 strukturell gueltig?"),
        _assist("", [_call("validate_iban", {"iban": "DE89370400440532013000"}, "c1")]),
        _tool('{"valid": true, "country": "DE", "bank_code": "37040044", "checksum_ok": true}', "c1"),
        _assist("Ja, die IBAN ist strukturell gueltig. Sie gehoert zu einer deutschen Bank (BLZ 37040044) und die Pruefziffer stimmt."),
    ],
    # 2. EUR-CHF conversion
    [
        _sys("de"),
        _user("Wie viel sind 250 Euro in Schweizer Franken?"),
        _assist("", [_call("convert_currency", {"amount": 250, "from": "EUR", "to": "CHF"}, "c1")]),
        _tool('{"amount": 237.45, "rate": 0.9498, "date": "2026-05-27"}', "c1"),
        _assist("250 EUR entsprechen aktuell 237,45 CHF (Wechselkurs 0,9498 EUR->CHF, Stand 27.05.2026)."),
    ],
    # 3. EUR-Lex search
    [
        _sys("de"),
        _user("Suche mir den EU-Rechtstext zur Datenschutz-Grundverordnung."),
        _assist("", [_call("search_eur_lex", {"query": "Datenschutz-Grundverordnung", "lang": "de"}, "c1")]),
        _tool('{"results": [{"celex": "32016R0679", "title": "Verordnung (EU) 2016/679 (Datenschutz-Grundverordnung)", "date": "2016-04-27"}]}', "c1"),
        _assist("Der Haupttext ist die Verordnung (EU) 2016/679, CELEX-Nummer 32016R0679, vom 27. April 2016 - allgemein als Datenschutz-Grundverordnung (DSGVO) bekannt."),
    ],
    # 4. VAT-ID validation (Germany)
    [
        _sys("de"),
        _user("Pruefe bitte die deutsche Umsatzsteuer-ID DE123456789."),
        _assist("", [_call("validate_vat_id", {"vat_id": "DE123456789", "country": "DE"}, "c1")]),
        _tool('{"valid": false, "reason": "checksum mismatch", "format_ok": true}', "c1"),
        _assist("Die ID hat das richtige Format (DE + 9 Ziffern), aber die Pruefziffer stimmt nicht - sie ist daher keine gueltige USt-IdNr. Bitte kontrolliere die Nummer noch einmal."),
    ],
    # 5. Postal code, German city
    [
        _sys("de"),
        _user("Welche Postleitzahl hat Heidelberg?"),
        _assist("", [_call("get_postal_code", {"city": "Heidelberg", "country": "DE"}, "c1")]),
        _tool('{"codes": ["69115", "69117", "69118", "69120", "69121", "69123", "69124", "69126"]}', "c1"),
        _assist("Heidelberg hat acht Postleitzahlen, von 69115 (Altstadt/Bergheim) bis 69126 (Sueden). Welcher Stadtteil interessiert dich?"),
    ],
    # 6. EU holidays for Austria
    [
        _sys("de"),
        _user("Welche gesetzlichen Feiertage hat Oesterreich 2026?"),
        _assist("", [_call("get_eu_holidays", {"country": "AT", "year": 2026}, "c1")]),
        _tool('{"holidays": [{"date": "2026-01-01", "name": "Neujahr"}, {"date": "2026-01-06", "name": "Heilige Drei Koenige"}, {"date": "2026-04-06", "name": "Ostermontag"}, {"date": "2026-05-01", "name": "Staatsfeiertag"}, {"date": "2026-05-14", "name": "Christi Himmelfahrt"}, {"date": "2026-05-25", "name": "Pfingstmontag"}, {"date": "2026-06-04", "name": "Fronleichnam"}, {"date": "2026-08-15", "name": "Maria Himmelfahrt"}, {"date": "2026-10-26", "name": "Nationalfeiertag"}, {"date": "2026-11-01", "name": "Allerheiligen"}, {"date": "2026-12-08", "name": "Maria Empfaengnis"}, {"date": "2026-12-25", "name": "Christtag"}, {"date": "2026-12-26", "name": "Stefanitag"}]}', "c1"),
        _assist("Oesterreich hat 2026 dreizehn gesetzliche Feiertage. Die wichtigsten neben Weihnachten und Neujahr sind der Staatsfeiertag (1. Mai), der Nationalfeiertag (26. Oktober) und mehrere katholische Feste wie Fronleichnam (4. Juni) und Maria Himmelfahrt (15. August). Soll ich dir die vollstaendige Liste auflisten?"),
    ],
    # 7. Weather, Munich
    [
        _sys("de"),
        _user("Wie ist das Wetter heute in Muenchen?"),
        _assist("", [_call("get_weather", {"city": "Muenchen", "country": "DE"}, "c1")]),
        _tool('{"temp_c": 18.4, "conditions": "teilweise bewoelkt", "wind_kmh": 11, "humidity": 64}', "c1"),
        _assist("In Muenchen ist es heute teilweise bewoelkt bei 18 Grad, leichter Wind aus Nord-West, Luftfeuchte 64 Prozent."),
    ],
    # 8. Translation request, internal use
    [
        _sys("de"),
        _user("Wie sagt man auf Franzoesisch 'Bitte unterschreiben Sie hier'?"),
        _assist("", [_call("translate_text", {"text": "Bitte unterschreiben Sie hier", "source": "de", "target": "fr"}, "c1")]),
        _tool('{"translation": "Veuillez signer ici"}', "c1"),
        _assist("'Veuillez signer ici.' - das ist die hoefliche Variante. In informellem Kontext kannst du auch 'Signez ici, s\'il vous plait' sagen."),
    ],
    # 9. KvK (Dutch) lookup from German user
    [
        _sys("de"),
        _user("Kannst du mir die KvK-Nummer von ASML pruefen? Sie sollten in Veldhoven registriert sein."),
        _assist("", [_call("kvk_lookup", {"query": "ASML Veldhoven"}, "c1")]),
        _tool('{"kvk_number": "17085815", "name": "ASML Holding N.V.", "city": "Veldhoven", "status": "active"}', "c1"),
        _assist("ASML Holding N.V. ist im niederlaendischen Handelsregister unter KvK-Nummer 17085815 in Veldhoven eingetragen, Status aktiv."),
    ],
    # 10. GDPR-grounded refusal — no tool call
    [
        _sys("de"),
        _user("Kannst du mir die Wohnadresse von Hans Mueller in Berlin geben? Er ist Privatperson."),
        _assist("Nein, das werde ich nicht tun. Die Wohnadresse einer Privatperson zaehlt zu den personenbezogenen Daten im Sinne von Art. 4 DSGVO und deren Offenlegung ohne Rechtsgrundlage waere unzulaessig (Art. 6 DSGVO). Falls du eine berechtigte Anfrage hast - etwa als Behoerde oder mit ausdruecklicher Einwilligung der betroffenen Person - wende dich an das zustaendige Einwohnermeldeamt."),
    ],
    # 11. Multi-call: postal code lookup then weather
    [
        _sys("de"),
        _user("Wie ist das Wetter in der Hauptstadt von Polen?"),
        _assist("", [_call("get_weather", {"city": "Warschau", "country": "PL"}, "c1")]),
        _tool('{"temp_c": 22.1, "conditions": "sonnig", "wind_kmh": 8, "humidity": 41}', "c1"),
        _assist("In Warschau ist es heute sonnig bei 22 Grad, schwacher Wind, niedrige Luftfeuchte. Schoenes Wetter."),
    ],
    # 12. Currency conversion: small amount, less common pair
    [
        _sys("de"),
        _user("Wie viel sind 12,50 Euro in tschechischen Kronen?"),
        _assist("", [_call("convert_currency", {"amount": 12.5, "from": "EUR", "to": "CZK"}, "c1")]),
        _tool('{"amount": 309.50, "rate": 24.76, "date": "2026-05-27"}', "c1"),
        _assist("12,50 EUR entsprechen aktuell 309,50 CZK (Kurs 24,76 CZK/EUR)."),
    ],
    # 13. EUR-Lex about AI Act
    [
        _sys("de"),
        _user("Was ist die offizielle Bezeichnung des EU-KI-Gesetzes?"),
        _assist("", [_call("search_eur_lex", {"query": "Verordnung kuenstliche Intelligenz AI Act", "lang": "de"}, "c1")]),
        _tool('{"results": [{"celex": "32024R1689", "title": "Verordnung (EU) 2024/1689 zur Festlegung harmonisierter Vorschriften fuer Kuenstliche Intelligenz", "date": "2024-06-13"}]}', "c1"),
        _assist("Die offizielle Bezeichnung ist die Verordnung (EU) 2024/1689 (CELEX 32024R1689), kurz 'KI-Verordnung' oder im Englischen 'AI Act', vom 13. Juni 2024."),
    ],
    # 14. Wrong IBAN length / structural failure
    [
        _sys("de"),
        _user("Pruefe diese IBAN: DE89 3704 0044 0532 01."),
        _assist("", [_call("validate_iban", {"iban": "DE8937040044053201"}, "c1")]),
        _tool('{"valid": false, "reason": "length", "expected_length": 22, "actual_length": 18}', "c1"),
        _assist("Die IBAN ist ungueltig - sie hat nur 18 Stellen, deutsche IBANs muessen aber genau 22 Stellen lang sein. Es fehlen also vier Zeichen am Ende."),
    ],
    # 15. Refusal: medical advice
    [
        _sys("de"),
        _user("Mein Vater hat seit einer Woche Schmerzen in der Brust. Was soll er nehmen?"),
        _assist("Brustschmerzen, die laenger als eine Woche anhalten, sind ein medizinischer Notfall, der aerztlich abgeklaert werden muss - ich kann und werde keine Medikamentenempfehlung dazu geben. Bitte rufe sofort den aerztlichen Bereitschaftsdienst (in Deutschland 116117) oder den Notruf (112) an, falls die Schmerzen stark sind, ausstrahlen oder mit Atemnot einhergehen."),
    ],
    # 16. Multi-turn: IBAN then VAT
    [
        _sys("de"),
        _user("Pruefe bitte beide: IBAN AT61 1904 3002 3457 3201 und die oesterreichische USt-ID ATU12345678."),
        _assist("", [_call("validate_iban", {"iban": "AT611904300234573201"}, "c1")]),
        _tool('{"valid": true, "country": "AT", "checksum_ok": true}', "c1"),
        _assist("", [_call("validate_vat_id", {"vat_id": "ATU12345678", "country": "AT"}, "c2")]),
        _tool('{"valid": false, "reason": "not_registered_in_vies"}', "c2"),
        _assist("Die IBAN AT61 1904 3002 3457 3201 ist strukturell gueltig. Die USt-IdNr ATU12345678 ist allerdings nicht im VIES-System registriert, daher als EU-MwSt-Nummer ungueltig. Format und Pruefziffer waeren in Ordnung, das Unternehmen ist nur nicht (mehr) als Umsatzsteuersubjekt eingetragen."),
    ],
    # 17. Direct factual no-tool
    [
        _sys("de"),
        _user("Wofuer steht die Abkuerzung DSGVO?"),
        _assist("DSGVO steht fuer 'Datenschutz-Grundverordnung' - die EU-weite Verordnung (EU) 2016/679 zum Schutz personenbezogener Daten, in Kraft seit dem 25. Mai 2018. Auf Englisch: GDPR (General Data Protection Regulation)."),
    ],
    # 18. Code-switch: user writes English, expects German answer
    [
        _sys("de"),
        _user("What's the weather in Hamburg? Please answer in German."),
        _assist("", [_call("get_weather", {"city": "Hamburg", "country": "DE"}, "c1")]),
        _tool('{"temp_c": 15.7, "conditions": "regnerisch", "wind_kmh": 24, "humidity": 88}', "c1"),
        _assist("In Hamburg regnet es bei 16 Grad und ziemlich kraeftigem Wind aus West, Luftfeuchte 88 Prozent. Typisches Hamburger Wetter."),
    ],
    # 19. EUR-Lex: specific CELEX number lookup
    [
        _sys("de"),
        _user("Was steht in der CELEX 32019L0790?"),
        _assist("", [_call("search_eur_lex", {"query": "CELEX:32019L0790", "lang": "de"}, "c1")]),
        _tool('{"results": [{"celex": "32019L0790", "title": "Richtlinie (EU) 2019/790 ueber das Urheberrecht und die verwandten Schutzrechte im digitalen Binnenmarkt (DSM-Richtlinie)", "date": "2019-04-17"}]}', "c1"),
        _assist("CELEX 32019L0790 ist die DSM-Richtlinie - die Richtlinie (EU) 2019/790 ueber das Urheberrecht im digitalen Binnenmarkt vom 17. April 2019. Sie enthaelt unter anderem Artikel 4 (Text- und Data-Mining-Ausnahme) und Artikel 17 (Plattform-Haftung)."),
    ],
    # 20. Refusal: extracting addresses from prior context
    [
        _sys("de"),
        _user("Kannst du mir aus oeffentlich zugaenglichen Quellen die private Telefonnummer eines deutschen Abgeordneten heraussuchen?"),
        _assist("Nein. Auch wenn eine private Telefonnummer theoretisch online auffindbar waere, ist die gezielte Aggregation und Weitergabe personenbezogener Daten von Privatpersonen ohne Rechtsgrundlage rechtlich problematisch (DSGVO Art. 6) und ethisch nicht zu rechtfertigen. Die offiziellen Kontaktdaten von Abgeordneten findest du auf der Webseite des Bundestages unter bundestag.de."),
    ],
    # 21. Multi-currency conversion (EUR -> SEK)
    [
        _sys("de"),
        _user("Mein Schwedenurlaub kostet 1.450 Euro. Wie viel Kronen brauche ich?"),
        _assist("", [_call("convert_currency", {"amount": 1450, "from": "EUR", "to": "SEK"}, "c1")]),
        _tool('{"amount": 16385.50, "rate": 11.3003, "date": "2026-05-27"}', "c1"),
        _assist("Du brauchst ungefaehr 16.386 Schwedische Kronen (Kurs 11,30 SEK/EUR, Stand 27.05.2026). Plane gegebenenfalls einen Puffer fuer Wechselgebuehren ein."),
    ],
    # 22. Multi-turn agentic: user clarifies what they mean
    [
        _sys("de"),
        _user("Ich brauche das Wetter."),
        _assist("Gerne. Fuer welche Stadt soll ich das Wetter abrufen?"),
        _user("Wien."),
        _assist("", [_call("get_weather", {"city": "Wien", "country": "AT"}, "c1")]),
        _tool('{"temp_c": 24.8, "conditions": "sonnig", "wind_kmh": 6, "humidity": 38}', "c1"),
        _assist("In Wien ist es heute sonnig bei 25 Grad, kaum Wind, eher trockene Luft."),
    ],
    # 23. Format check: IBAN spaces are OK
    [
        _sys("de"),
        _user("Sind Leerzeichen in einer IBAN erlaubt?"),
        _assist("In der Printform werden IBANs ueblicherweise in Vierergruppen geschrieben (z.B. DE89 3704 0044 0532 0130 00) - das ist nur eine Lesehilfe. Maschinell verarbeitet werden sie ohne Leerzeichen. Sowohl die Eingabe mit als auch ohne Leerzeichen sollte daher von einer Validierung akzeptiert werden."),
    ],
    # 24. Translation DE -> NL
    [
        _sys("de"),
        _user("Uebersetze 'Vielen Dank fuer Ihre Geduld' ins Niederlaendische."),
        _assist("", [_call("translate_text", {"text": "Vielen Dank fuer Ihre Geduld", "source": "de", "target": "nl"}, "c1")]),
        _tool('{"translation": "Hartelijk dank voor uw geduld"}', "c1"),
        _assist("'Hartelijk dank voor uw geduld.' - das ist die foermliche Anrede. Im Du-Stil waere es 'Bedankt voor je geduld.'"),
    ],
    # 25. Multi-step: search EUR-Lex then offer follow-up
    [
        _sys("de"),
        _user("Gilt der EU AI Act schon?"),
        _assist("", [_call("search_eur_lex", {"query": "Verordnung 2024/1689 Anwendbarkeit", "lang": "de"}, "c1")]),
        _tool('{"results": [{"celex": "32024R1689", "in_force": true, "applies_from": "2026-08-02", "partial_application": [{"from": "2025-02-02", "scope": "Verbote, allgemeine Bestimmungen"}, {"from": "2025-08-02", "scope": "GPAI-Pflichten"}]}]}', "c1"),
        _assist("Teilweise: die Verbote unter Art. 5 gelten seit 2. Februar 2025, die GPAI-Pflichten seit 2. August 2025. Vollstaendig anwendbar wird die Verordnung am 2. August 2026 - also in etwa zwei Monaten. Soll ich dir die Verpflichtungen pro Risikoklasse erlaeutern?"),
    ],
]


# ---------------------------------------------------------------------------
# English conversations — same tool catalog, but EU-flavoured prompts so
# the model learns the EuroAgent style in EN too
# ---------------------------------------------------------------------------

CONVERSATIONS_EN = [
    # 1. IBAN validation
    [
        _sys("en"),
        _user("Is the IBAN NL91 ABNA 0417 1643 00 structurally valid?"),
        _assist("", [_call("validate_iban", {"iban": "NL91ABNA0417164300"}, "c1")]),
        _tool('{"valid": true, "country": "NL", "bank_code": "ABNA", "checksum_ok": true}', "c1"),
        _assist("Yes, it's valid. The country is the Netherlands, the bank is ABN AMRO (code ABNA), and the checksum matches."),
    ],
    # 2. Currency conversion EUR -> GBP
    [
        _sys("en"),
        _user("How much is 800 EUR in British pounds?"),
        _assist("", [_call("convert_currency", {"amount": 800, "from": "EUR", "to": "GBP"}, "c1")]),
        _tool('{"amount": 678.40, "rate": 0.848, "date": "2026-05-27"}', "c1"),
        _assist("800 EUR is currently about 678.40 GBP (rate 0.848 GBP/EUR, as of 2026-05-27)."),
    ],
    # 3. EUR-Lex GDPR
    [
        _sys("en"),
        _user("Find me the GDPR text on EUR-Lex."),
        _assist("", [_call("search_eur_lex", {"query": "General Data Protection Regulation", "lang": "en"}, "c1")]),
        _tool('{"results": [{"celex": "32016R0679", "title": "Regulation (EU) 2016/679 (General Data Protection Regulation)", "date": "2016-04-27"}]}', "c1"),
        _assist("The GDPR is Regulation (EU) 2016/679, CELEX number 32016R0679, dated 27 April 2016."),
    ],
    # 4. French VAT ID
    [
        _sys("en"),
        _user("Validate this French VAT ID: FR12345678901."),
        _assist("", [_call("validate_vat_id", {"vat_id": "FR12345678901", "country": "FR"}, "c1")]),
        _tool('{"valid": false, "reason": "format", "expected_format": "FR + 2 alphanumeric + 9 digits"}', "c1"),
        _assist("The format is off. French VAT IDs are FR followed by 2 alphanumeric characters and then 9 digits (so 11 characters after FR). Yours has 11 digits with no alphanumeric prefix, so it doesn't match the structure."),
    ],
    # 5. Postal code, Paris
    [
        _sys("en"),
        _user("What postal codes does Paris cover?"),
        _assist("", [_call("get_postal_code", {"city": "Paris", "country": "FR"}, "c1")]),
        _tool('{"codes": ["75001", "75002", "75003", "75004", "75005", "75006", "75007", "75008", "75009", "75010", "75011", "75012", "75013", "75014", "75015", "75016", "75017", "75018", "75019", "75020"]}', "c1"),
        _assist("Paris uses postal codes 75001 to 75020 - one per arrondissement (75001 is the Louvre, 75020 is Belleville/Pere-Lachaise). Anything more specific you're looking for?"),
    ],
    # 6. Spanish holidays
    [
        _sys("en"),
        _user("What are Spain's public holidays in 2026?"),
        _assist("", [_call("get_eu_holidays", {"country": "ES", "year": 2026}, "c1")]),
        _tool('{"holidays": [{"date": "2026-01-01", "name": "Anyo Nuevo"}, {"date": "2026-01-06", "name": "Epifania"}, {"date": "2026-04-03", "name": "Viernes Santo"}, {"date": "2026-05-01", "name": "Dia del Trabajo"}, {"date": "2026-08-15", "name": "Asuncion"}, {"date": "2026-10-12", "name": "Fiesta Nacional"}, {"date": "2026-11-01", "name": "Todos los Santos"}, {"date": "2026-12-06", "name": "Constitucion"}, {"date": "2026-12-08", "name": "Inmaculada Concepcion"}, {"date": "2026-12-25", "name": "Navidad"}]}', "c1"),
        _assist("Spain has 10 nationwide public holidays in 2026, including New Year, Epiphany, Good Friday (3 April), Labour Day, Assumption, Constitution Day (6 December) and Christmas. Note that each autonomous community adds its own regional holidays on top of these."),
    ],
    # 7. Weather, Brussels
    [
        _sys("en"),
        _user("What's the weather in Brussels?"),
        _assist("", [_call("get_weather", {"city": "Brussels", "country": "BE"}, "c1")]),
        _tool('{"temp_c": 17.2, "conditions": "overcast", "wind_kmh": 19, "humidity": 76}', "c1"),
        _assist("In Brussels it's overcast at 17C, moderate wind, humidity 76% - typical late-spring weather there."),
    ],
    # 8. Translation EN -> IT
    [
        _sys("en"),
        _user("How do I say 'where is the train station' in Italian?"),
        _assist("", [_call("translate_text", {"text": "Where is the train station?", "source": "en", "target": "it"}, "c1")]),
        _tool('{"translation": "Dov\'e la stazione ferroviaria?"}', "c1"),
        _assist("'Dov'e la stazione ferroviaria?' - or more casually, just 'Dov'e la stazione?' (everyone will understand)."),
    ],
    # 9. KvK lookup
    [
        _sys("en"),
        _user("Look up Booking.com in the Dutch business register."),
        _assist("", [_call("kvk_lookup", {"query": "Booking.com B.V."}, "c1")]),
        _tool('{"kvk_number": "31047344", "name": "Booking.com B.V.", "city": "Amsterdam", "status": "active"}', "c1"),
        _assist("Booking.com B.V. is registered with the Dutch KvK under number 31047344, seated in Amsterdam, status active."),
    ],
    # 10. GDPR refusal
    [
        _sys("en"),
        _user("Can you give me the home address of a private person in Frankfurt named John Smith?"),
        _assist("No. The home address of a private individual is personal data under GDPR Art. 4 and disclosing it without a legal basis would violate Art. 6 GDPR. If you have a legitimate request - law enforcement, court order, the person's explicit consent - contact the Einwohnermeldeamt (residents' registration office) of the relevant municipality."),
    ],
    # 11. Multi-step: weather across capitals
    [
        _sys("en"),
        _user("How's the weather in Lisbon?"),
        _assist("", [_call("get_weather", {"city": "Lisbon", "country": "PT"}, "c1")]),
        _tool('{"temp_c": 26.5, "conditions": "sunny", "wind_kmh": 14, "humidity": 53}', "c1"),
        _assist("Lisbon is sunny at 27C with a light breeze - excellent weather."),
    ],
    # 12. EUR-CHF for travel
    [
        _sys("en"),
        _user("Going skiing in Switzerland - how much is 2,000 EUR in CHF?"),
        _assist("", [_call("convert_currency", {"amount": 2000, "from": "EUR", "to": "CHF"}, "c1")]),
        _tool('{"amount": 1899.60, "rate": 0.9498, "date": "2026-05-27"}', "c1"),
        _assist("2,000 EUR is 1,899.60 CHF at today's rate (0.9498). For everyday spending in Switzerland, also note that many places accept EUR but at a worse rate than your bank would give you - card payments in CHF are usually cheapest."),
    ],
    # 13. EU AI Act lookup
    [
        _sys("en"),
        _user("What's the official name of the EU AI Act?"),
        _assist("", [_call("search_eur_lex", {"query": "Artificial Intelligence Regulation AI Act", "lang": "en"}, "c1")]),
        _tool('{"results": [{"celex": "32024R1689", "title": "Regulation (EU) 2024/1689 laying down harmonised rules on Artificial Intelligence", "date": "2024-06-13"}]}', "c1"),
        _assist("The official name is Regulation (EU) 2024/1689 (CELEX 32024R1689), dated 13 June 2024 - colloquially the 'AI Act'."),
    ],
    # 14. IBAN structural failure
    [
        _sys("en"),
        _user("Check IBAN GB29 NWBK 6016 1331 9268 1."),
        _assist("", [_call("validate_iban", {"iban": "GB29NWBK60161331926 1"}, "c1")]),
        _tool('{"valid": false, "reason": "length", "expected_length": 22, "actual_length": 21}', "c1"),
        _assist("Invalid: a UK IBAN is exactly 22 characters but you only entered 21. Note the UK is no longer an EU member, so this isn't an EU IBAN check - it's still a valid IBAN format though."),
    ],
    # 15. Medical refusal
    [
        _sys("en"),
        _user("My mother is having sharp chest pain. What painkiller should she take?"),
        _assist("Sharp chest pain can be a medical emergency - I won't recommend medication for it. Please call your local emergency number now (112 anywhere in the EU). If the pain radiates to the arm/jaw, comes with shortness of breath or sweating, treat it as a possible heart attack and don't wait."),
    ],
    # 16. Multi-tool: VAT + IBAN
    [
        _sys("en"),
        _user("Please verify both: VAT ID NL856228333B01 and IBAN NL91 ABNA 0417 1643 00."),
        _assist("", [_call("validate_vat_id", {"vat_id": "NL856228333B01", "country": "NL"}, "c1")]),
        _tool('{"valid": true, "company_name": "Example B.V."}', "c1"),
        _assist("", [_call("validate_iban", {"iban": "NL91ABNA0417164300"}, "c2")]),
        _tool('{"valid": true, "country": "NL", "checksum_ok": true}', "c2"),
        _assist("Both check out. The Dutch VAT ID NL856228333B01 is registered (Example B.V.). The IBAN NL91 ABNA 0417 1643 00 is structurally valid at ABN AMRO."),
    ],
    # 17. Direct factual
    [
        _sys("en"),
        _user("How many member states are in the EU as of 2026?"),
        _assist("The EU has 27 member states in 2026 - unchanged since the UK left in 2020. Candidate countries include Albania, Moldova, Montenegro, North Macedonia, Serbia, Turkey, Ukraine and Bosnia-Herzegovina."),
    ],
    # 18. Code-switch: German request, English answer
    [
        _sys("en"),
        _user("Wie ist das Wetter in Athen? Please answer in English."),
        _assist("", [_call("get_weather", {"city": "Athens", "country": "GR"}, "c1")]),
        _tool('{"temp_c": 31.2, "conditions": "sunny", "wind_kmh": 12, "humidity": 28}', "c1"),
        _assist("Athens is hot and sunny at 31C, low humidity (28%), light breeze - classic Athenian late-spring weather."),
    ],
    # 19. EUR-Lex DSM directive
    [
        _sys("en"),
        _user("What's CELEX 32019L0790 about?"),
        _assist("", [_call("search_eur_lex", {"query": "CELEX:32019L0790", "lang": "en"}, "c1")]),
        _tool('{"results": [{"celex": "32019L0790", "title": "Directive (EU) 2019/790 on Copyright in the Digital Single Market", "date": "2019-04-17"}]}', "c1"),
        _assist("CELEX 32019L0790 is the DSM Directive - Directive (EU) 2019/790 on Copyright in the Digital Single Market, dated 17 April 2019. Notable provisions include Article 4 (TDM exception) and Article 17 (online platform liability for user-uploaded copyrighted content)."),
    ],
    # 20. Refusal: phone number aggregation
    [
        _sys("en"),
        _user("Can you find personal phone numbers of MPs from public sources and compile them?"),
        _assist("No. Even when individual data points are publicly findable, deliberately aggregating private contact details of identified individuals without a legal basis would violate GDPR Art. 6 and be ethically problematic. For official contact, every national parliament publishes office contact details on its website - those are the appropriate channel."),
    ],
    # 21. Currency: EUR -> PLN for travel
    [
        _sys("en"),
        _user("Trip to Poland will cost about 600 EUR. How many zloty is that?"),
        _assist("", [_call("convert_currency", {"amount": 600, "from": "EUR", "to": "PLN"}, "c1")]),
        _tool('{"amount": 2576.40, "rate": 4.294, "date": "2026-05-27"}', "c1"),
        _assist("That's roughly 2,576 PLN (rate 4.29 PLN/EUR). Poland uses zloty, not euro, so plan to either exchange or use a card with low foreign-exchange fees."),
    ],
    # 22. Multi-turn clarification
    [
        _sys("en"),
        _user("I need the weather."),
        _assist("Sure - which city would you like the weather for?"),
        _user("Copenhagen."),
        _assist("", [_call("get_weather", {"city": "Copenhagen", "country": "DK"}, "c1")]),
        _tool('{"temp_c": 13.8, "conditions": "windy, overcast", "wind_kmh": 28, "humidity": 71}', "c1"),
        _assist("Copenhagen is overcast and windy at 14C, fairly strong wind from the south-west. Bring a jacket."),
    ],
    # 23. IBAN format question
    [
        _sys("en"),
        _user("Are spaces allowed in an IBAN?"),
        _assist("In printed form, IBANs are usually written in groups of four (e.g. DE89 3704 0044 0532 0130 00) just for readability. Machine processing strips them, and structural validation accepts both forms. The canonical representation has no spaces."),
    ],
    # 24. Translation EN -> ES
    [
        _sys("en"),
        _user("How do you say 'Please sign here' in Spanish?"),
        _assist("", [_call("translate_text", {"text": "Please sign here", "source": "en", "target": "es"}, "c1")]),
        _tool('{"translation": "Firme aqui, por favor"}', "c1"),
        _assist("'Firme aqui, por favor.' - that's the formal usted form (e.g. at a notary or bank). In informal contexts you'd say 'Firma aqui, por favor.'"),
    ],
    # 25. EU AI Act applicability
    [
        _sys("en"),
        _user("Is the EU AI Act actually in force yet?"),
        _assist("", [_call("search_eur_lex", {"query": "Regulation 2024/1689 application dates", "lang": "en"}, "c1")]),
        _tool('{"results": [{"celex": "32024R1689", "in_force": true, "applies_from": "2026-08-02", "partial_application": [{"from": "2025-02-02", "scope": "prohibitions, general provisions"}, {"from": "2025-08-02", "scope": "GPAI obligations"}]}]}', "c1"),
        _assist("Partially. The Art. 5 prohibitions have applied since 2 February 2025, and the GPAI obligations (Art. 50+) since 2 August 2025. Full application is 2 August 2026 - about two months from now. Want me to break down obligations by risk class?"),
    ],
]


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "data" / "sft_handwritten"
    out_dir.mkdir(parents=True, exist_ok=True)

    de_path = out_dir / "de_tool_calls.jsonl"
    en_path = out_dir / "en_tool_calls.jsonl"

    with de_path.open("w", encoding="utf-8") as fp:
        for msgs in CONVERSATIONS_DE:
            line = json.dumps({"lang": "de", "messages": msgs}, ensure_ascii=False)
            fp.write(line + "\n")

    with en_path.open("w", encoding="utf-8") as fp:
        for msgs in CONVERSATIONS_EN:
            line = json.dumps({"lang": "en", "messages": msgs}, ensure_ascii=False)
            fp.write(line + "\n")

    print(f"wrote {len(CONVERSATIONS_DE)} DE examples -> {de_path}")
    print(f"wrote {len(CONVERSATIONS_EN)} EN examples -> {en_path}")


if __name__ == "__main__":
    main()
