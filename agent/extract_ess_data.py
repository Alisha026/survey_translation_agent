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