import re
import json
import pdfplumber
from pathlib import Path

# Paths
EN_PDF = "data/raw/ess11_en.pdf"
DE_PDF = "data/raw/ess11_de.pdf"
OUT_PATH = "data/processed/ess11_items.json"

# Mapping questions to item types based on section
SECTION_TO_TYPE = {
    "A": "behavioral",        # Media use, internet
    "B": "attitudinal",       # Politics, trust, immigration
    "C": "attitudinal",       # Wellbeing, discrimination, religion
    "D": "behavioral",        # Social inequalities in health
    "E": "attitudinal",       # Gender in contemporary Europe
    "F": "sociodemographic",  # Socio-demographic profile
    "H": "likert",            # Human values scale
    "I": "attitudinal",       # Test questions
    "K": "attitudinal",       # COVID-19
    "R": "attitudinal",       # Recontact
}

# Words parts of the instructions for interviewer, not questions
SKIP_KEYWORDS = ["INTERVIEWER", "SHOW CARD", "CODE ALL", "DO NOT READ", 
                 "GO TO", "ASK IF", "READ OUT", "TYPE IN", "IF NECESSARY",
                 "STILL CARD", "DISPLAY", "ASK ALL"]


def extract_text_from_pdf(pdf_path):
    """
    Extract full text from PDF page by page.
    Returns one big string with all pages joined.
    """
    print(f"Extracting text from: {pdf_path}")
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}")
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append(text)

    full_text = "\n".join(pages)
    print(f"Extracted {len(full_text)} characters total")
    return full_text

def clean_question_text(text):

    # remove CARD and STILL CARD references
    text = re.sub(r'(STILL\s+)?CARD\s+\d+', '', text)

    # remove LISTE references (German)
    text = re.sub(r'LISTE\s+\d+\s*', '', text)

    # remove scale number rows like "0 0 0 0 0 0 0 0 0 0 1 7 88"
    text = re.sub(r'(\b0\s+){3,}\d+(\s+\d+)*', '', text)

    # remove response option lines like "Not at all 1 Very little 2..."
    text = re.sub(
        r'\s+\d{1,2}\s+(Very|Not|Some|A lot|Never|Always|Strongly|'
        r'Don\'t|Yes|No|Conservative|Labour|Überhaupt|Ein|Ziemlich|'
        r'Sehr|Voll|Stark).+$',
        '', text, flags=re.DOTALL | re.IGNORECASE
    )

    # remove footnote blocks like "2 'Can't be too careful': ..."
    text = re.sub(r"\d+\s*'[^']+'.+?(?=\d+\s*'|\Z)", '', text, flags=re.DOTALL)

    # remove (Refusal) (Don't know) (Antwort verweigert) (Weiß nicht)
    text = re.sub(
        r'\(Refusal\)|\(Don\'t know\)|\(Antwort verweigert\)|\(Weiß nicht\)',
        '', text, flags=re.IGNORECASE
    )

    # remove footnote superscript numbers attached to words e.g. "affairs1?"
    text = re.sub(r'(\w)(\d{1,2})(\?|\s|,)', r'\1\3', text)

    # cut after question mark — everything after is response options or instructions
    if '?' in text:
        text = text[:text.index('?') + 1]

    # remove section headers like "SECTION B Now we want to ask..."
    text = re.sub(r'SECTION\s+[A-Z].+$', '', text, flags=re.DOTALL)

    # remove END DATE markers
    text = re.sub(r'\[END.+?\]', '', text)

    # collapse whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def skip(text):
    """ Return True if this text looks like an interviewer instruction."""
    text_upper = text.upper()
    return any(kw in text_upper for kw in SKIP_KEYWORDS)


def parse_questions(full_text):
    """
    Parse questions from ESS questionnaire text.

    ESS format examples:
        B14 Which party did you vote for in that election?
    
    Extracting question ID (e.g. B14) and question text (e.g. "Which party did you vote for in that election?")

    Returns dict of {question_id: question_text}.

    """
    questions = {}

    # Pattern: start of line, capital letter, 1-2 digits, optional lowercase
    q_id_pattern = re.compile(r'(?m)^([A-Z]\d{1,2}[a-z]?)\s+(.+?)(?=^[A-Z]\d{1,2}[a-z]?\s|\Z)', re.DOTALL)

    matches = list(q_id_pattern.finditer(full_text))
    print(f"Raw pattern matches: {len(matches)}")

    for match in matches:
        q_id   = match.group(1).strip()
        q_text = match.group(2).strip()

        q_text = clean_question_text(q_text)

        if len(q_text) < 15:
            continue

        if skip(q_text):
            continue

        if re.match(r'^\d+\s', q_text): # starts with numbers — likely response options or scale rows
            continue

        questions[q_id] = q_text

    print(f"Valid questions found: {len(questions)}")
    return questions


def get_item_type(q_id):
    """Get item type from question ID prefix letter."""
    letter = q_id[0] if q_id else "B"
    return SECTION_TO_TYPE.get(letter, "attitudinal")


def extract_scale(q_text):
    """Scale left empty and I am manually going to filled during data correction."""
    return ""


def match_and_build(en_questions, de_questions):
    """Match EN and DE questions by ID. Build final dataset."""
    dataset   = []
    matched   = 0
    unmatched = []

    for q_id, source_en in en_questions.items():
        trans_de = de_questions.get(q_id, "")

        if trans_de:
            matched += 1
        else:
            unmatched.append(q_id)

        item = {
            "id": f"ESS11_{q_id}",
            "question_id": q_id,
            "item_type": get_item_type(q_id),
            "source_en": source_en,
            "trans_de": trans_de,
            "scale": extract_scale(source_en),
            "round": "11",
            "source": "ESS"
        }
        dataset.append(item)

    print(f"Matched EN-DE pairs: {matched}")
    print(f"EN only (no DE match): {len(unmatched)}")
    if unmatched[:5]:
        print(f"Sample unmatched IDs: {unmatched[:5]}")

    return dataset


def filter_dataset(dataset, min_length=20, require_trans=True):
    """Remove low quality items."""
    filtered = []
    for item in dataset:
        if len(item['source_en']) < min_length:
            continue
        if require_trans and not item['trans_de']:
            continue
        filtered.append(item)

    print(f"Before filtering: {len(dataset)}")
    print(f"After filtering: {len(filtered)}")
    return filtered


def save_dataset(dataset, output_path):
    """Save to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"  Saved to: {output_path}")


def print_sample(dataset, n=5):
    """Print sample items for inspection."""
    print(f"\n{'='*60}")
    print(f"SAMPLE ITEMS (first {n})")
    print(f"{'='*60}")
    for item in dataset[:n]:
        print(f"\nID: {item['id']}")
        print(f"Type: {item['item_type']}")
        print(f"EN: {item['source_en'][:120]}")
        print(f"DE: {item['trans_de'][:120] if item['trans_de'] else 'NO MATCH'}")
        print("-" * 40)


def print_stats(dataset):
    """Print dataset statistics."""
    from collections import Counter
    type_counts = Counter(item['item_type'] for item in dataset)
    has_trans    = sum(1 for item in dataset if item['trans_de'])

    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total items: {len(dataset)}")
    print(f"With translation DE: {has_trans}")
    print(f"Without translation DE: {len(dataset) - has_trans}")
    print(f"\nBy item type:")
    for t, c in type_counts.most_common():
        print(f"  {t:<20} {c}")


def main():
    print("\n" + "="*60)
    print("ESS Round 11 — Questionnaire Extraction")
    print("="*60)

    print("\n[Step 1] English PDF...")
    en_text = extract_text_from_pdf(EN_PDF)

    print("\n[Step 2] German PDF...")
    de_text = extract_text_from_pdf(DE_PDF)

    print("\n[Step 3] Parsing English questions...")
    en_questions = parse_questions(en_text)

    print("\n[Step 4] Parsing German questions...")
    de_questions = parse_questions(de_text)

    print("\n[Step 5] Matching EN-DE pairs...")
    dataset = match_and_build(en_questions, de_questions)

    print("\n[Step 6] Filtering...")
    dataset = filter_dataset(dataset, min_length=20, require_trans=True)

    print("\n[Step 7] Saving...")
    save_dataset(dataset, OUT_PATH)

    print_stats(dataset)
    print_sample(dataset)

    print("\nDone! Next: review the JSON and check quality.")


if __name__ == "__main__":
    main()

# Have used this to translate scale 
SCALE_TRANSLATIONS = {
    "Yes / No":                          "Ja / Nein",
    "Yes / No / Not eligible to vote":   "Ja / Nein / Nicht wahlberechtigt",
    "Yes / No / Not applicable":         "Ja / Nein / Nicht zutreffend",
    "Yes / No / Prefer not to say": "Ja / Nein / Keine Angabe",
    "Yes a lot / Yes to some extent / No": "Ja, stark / Ja, bis zu einem gewissen Grad / Nein",
    "Yes, currently / Yes, previously / No": "Ja, heute noch / Ja, früher einmal / Nein, nie",
    "0-10 (Extremely bad - Extremely good)":                    "0-10 (Äußerst schlecht - Äußerst gut)",
    "0-10 (Extremely dissatisfied - Extremely satisfied)":      "0-10 (Äußerst unzufrieden - Äußerst zufrieden)",
    "0-10 (Left - Right)":                                      "0-10 (Links - Rechts)",
    "0-10 (Most people would try to take advantage of me - Most people would try to be fair )": "0-10 (Die meisten Menschen versuchen, mich auszunutzen - Die meisten Menschen versuchen, sich fair zu verhalten)",
    "0-10 (People mostly look out for themselves - People mostly try to be helpful)": "0-10(Die Menschen sind meistens auf den eigenen Vorteil bedacht - Die Menschen versuchen, hilfsbereit zu sein)",
    "0-10 (You can't be too careful - Most people can be trusted)": "0-10(Man kann nicht vorsichtig genug sein - Den meisten Menschen kann mann vertrauen)",
    "0-10(Extremely unhappy - Extremely happy)":                "0-10(Äußerst unglücklich - Äußerst glücklich)",
    "0-10(Not at all emotionally attached - Very emotionally attached)": "0-10(Gefühlsmäßig überhaupt nicht verbunden - Gefühlsmäßig sehr verbunden)",
    "0-10(Not at all religious - Very religious)":              "0-10(Überhaupt nicht religiös - Sehr religiös)",
    "0-10(Worse place to live - Better place to live)":         "0-10(Wird zu einem schlechteren Ort zum Leben - Wird zu einem besseren Ort zum Leben)",
    "0-6(None - 10 or more)":                                   "0-6(Keinen - 10 oder mehr)",
    "0-6(Not at all - Completely)":                             "0-6(Überhaupt nicht - Voll und ganz)",
    "0-6(Not at all feminine - Very feminine)":                 "0-6(Überhaupt nicht weiblich - Sehr weiblich)",
    "0-6(Not at all masculine - Very masculine)":               "0-6(Überhaupt nicht männlich - Sehr männlich)",
    "0-6(Very bad for businesses in [country] - Very good for businesses in [country])": "0-6(Sehr schlecht für Unternehmen in Deutschland - Sehr gut für Unternehmen in Deutschland)",
    "0-6(Very bad for family life in [country] - Very good for family life in [country])": "0-6(Sehr schlecht für das Familienleben in Deutschland - Sehr gut für das Familienleben in Deutschland)",
    "0-6(Very bad for politics in [country] - Very good for politics in [country])": "0-6(Sehr schlecht für die Politik in Deutschland - Sehr gut für die Politik in Deutschland)",
    "0-6(Very bad for the strength of the economy in [country] - Very good for the strength of the economy in [country])": "0-6(Sehr schlecht für die Stärke der Wirtschaft in Deutschland - Sehr gut für die Stärke der Wirtschaft in Deutschland)",
    "1-10 (Bad for the economy - Good for the economy)":        "1-10 (Schlecht für die Wirtschaft - Gut für die Wirtschaft)",
    "1-10(Cultural life undermined - Cultural life enriched)":  "1-10(Kulturelles Leben wird untergraben - Kulturelles Leben wird bereichert)",
    "1-10(No control at all - Complete control)":               "1-10(Überhaupt keine Kontrolle - Vollständige Kontrolle)",
    "1-10(Not at all- A great deal)":                           "1-10(Überhaupt nicht - Sehr stark)",
    "1-4(Allow many to come and live here - Allow none)":       "1-4(vielen erlauben, herzukommen und hier zu leben - oder niemandem erlauben)",
    "1-4(very safe - very unsafe)":                             "1-4(sehr sicher - sehr unsicher)",
    "1-5 (Agree strongly - Disagree strongly)":                 "1-5 (Stimme stark zu - Lehne stark ab)",
    "1-5 (Agree strongly- Disagree strongly)":                  "1-5 (Stimme stark zu- Lehne stark ab)",
    "1-5 (Never - Always)":                                     "1-5 (Nie - Immer)",
    "1-5 (Not at all able - Completely able)":                  "1-5 (Überhaupt nicht fähig - Voll und ganz fähig)",
    "1-5 (Not at all confident - Completely confident)":        "1-5 (Vertraue meinen Fähigkeiten überhaupt nicht - Vertraue meinen Fähigkeiten voll und ganz)",
    "1-5 (Strongly in favour - Strongly against)":              "1-5 (Sehr dafür - Sehr dagegen)",
    "1-5(Always - Never)":                                      "1-5(Immer - Nie)",
    "1-5(Entirely by natural processes - Entirely by human activity)": "1-5(Nur durch natürliche Prozesse - Nur durch menschliches Handeln)",
    "1-5(Much less than most - Much more than most)":           "1-5(Viel seltener als die meisten - Viel häufiger als die meisten)",
    "1-5(Not at all worried - Extremely worried)":              "1-5(Überhaupt nicht besorgt - Äußerst besorgt)",
    "1-5(very good - very bad)":                                "1-5(sehr gut - sehr schlecht)",
    "1-7(Every day - Never)":                                   "1-7(Täglich - Nie)",
    "1-7(Never - Every day)":                                   "1-7(Nie - Täglich)",
    "1-7(Three times or more a day - Never)":                   "1-7(Dreimal pro Tag oder öfter - Nie)",
    "A man/ A woman /Other (TYPE IN) / Prefer not to say":      "Mann / Frau / Andere Bezeichnung (EINTRAGEN)",
    "Duration (hours and minutes)": "Dauer (Stunden und Minuten)",
    "Employee / Self-employed / Not working / Father dead or absent when you were 14": "Abhängig Beschäftigter / Selbständig / Keine bezahlte Tätigkeit / Vater bereits verstorben/lebte nicht im Haushalt als Befragte(r) 14 war",
    "Employee / Self-employed / Not working / Mother dead or absent when you were 14": "Abhängig Beschäftigte / Selbständig / Keine bezahlte Tätigkeit / Mutter bereits verstorben/lebte nicht im Haushalt als Befragte(r) 14 war",
    "Legally married / In a legally registered civil union / Legally separated / Widowed / None of these (Never married)": "Verheiratet / In einer eingetragenen Lebenspartnerschaft / Getrennt lebend / Verwitwet / Nichts davon (Noch nie verheiratet)",
    "Never / Only occasionally / A few times a week / Most days / Every day": "Nie / Nur ab und zu / Ein paarmal pro Woche / An den meisten Tagen / Jeden Tag",
    "Number (in cm)": "Zahl (in cm)",
    "very close / quite close / not close / not at all close": "Sehr nahe / Ziemlich nahe / nicht besonders nahe / Überhaupt nicht nahe",
    "0-10 (Unification has already gone too far - Unification should go further)": "0-10 (Einigung ist schon zu weit gegangen - Einigung sollte weitergehen)",
    "Number of days (0-7)": "Anzahl der Tage (0-7)",
    "Numeric (Hour)": "Numerisch (Stunde)",
    "Numeric (Year)": "Numerisch (Jahr)",
    "Numeric Count": "Numerische Anzahl",
    "Numeric Year": "Numerisches Jahr",
    "Open ended (up to 2 languages)": "Offene Antwort (bis zu 2 Sprachen)",
    "Text": "Text",
    "Text (up to 2 ancestries)": "Text (bis zu 2 Herkunftsangaben)",
    "The police treat women less fairly than men / The police treat men less fairly than women / Women and men are treated equally fairly": "Die Polizei behandelt Frauen weniger gerecht als Männer / Die Polizei behandelt Männer weniger gerecht als Frauen / Frauen und Männer werden gleichermaßen gerecht behandelt",
    "Wages or salaries / Income from self-employment (excluding farming) / Income from farming / Pensions / Unemployment/redundancy benefits / Any other social benefit / Income from investment, savings, insurance or property / Income from other source": "Löhne oder Gehälter / Einkommen aus selbständiger oder freiberuflicher Tätigkeit (ausgenommen Landwirtschaft) / Einkommen aus Landwirtschaft / Renten oder Pensionen / Arbeitslosengeld / Bürgergeld oder Abfindungen / Andere Sozialleistungen (Sozialhilfe, Bafög usw.) / Einkommen aus Vermögensanlagen, Ersparnissen, Versicherungen, Grundbesitz oder Immobilien / Einkommen aus anderen Quellen",
    "Women are treated less fairly than men / Men are treated less fairly than women / Women and men are treated equally fairly": "Frauen werden weniger gerecht behandelt als Männer / Männer werden weniger gerecht behandelt als Frauen / Frauen und Männer werden gleichermaßen gerecht behandelt",
    "in paid work / in education / unemployed and actively looking for work / unemployed and not actively looking for work / Permanently sick or disabled / retired / doing housework, looking after children or other persons / in military service / other": "Bezahlte Tätigkeit / Schule/Ausbildung / Arbeitslos und auf aktiver Suche nach einem Arbeitsplatz /Arbeitslos , keine aktive Suche / Chronisch krank oder behindert / m Vorruhestand/Ruhestand/Frührente/Rente / Hausarbeit, Betreuung von Kindern oder anderen Personen / Sonstiges",
}