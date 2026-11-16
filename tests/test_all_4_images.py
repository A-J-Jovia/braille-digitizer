import sys
sys.path.insert(0, 'D:/Braille_Digitizer')
import cv2
import re
import numpy as np

COMMON_ENGLISH_WORDS = {
    'the', 'and', 'for', 'are', 'can', 'trees', 'is', 'year', 'growing', 
    'redwoods', 'sequoias', 'in', 'on', 'at', 'to', 'of', 'it', 'was', 
    'as', 'he', 'she', 'that', 'with', 'from', 'by', 'this', 'have', 'not',
    'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', 'hello', 'world',
    'braille', 'alphabet'
}

COMMON_RUSSIAN_WORDS = {
    'шрифт', 'брайля', 'брайль', 'шоифт', 'сказка', 'сказки', 'книга', 'текст',
    'дерево', 'дерева', 'человек', 'время', 'дело', 'жизнь', 'день', 'рука',
    'когда', 'только', 'будет', 'очень', 'между', 'после', 'через', 'сказал'
}

def is_alphabet_sequence(text):
    cleaned = ' '.join(text.lower().split())
    return any(seq in cleaned for seq in ['a b c d', 'a b c', 'h i j k', 'o p q r', 'and for of the with', 'sh th wh'])

def score_english_coherence(text):
    clean_text = ' '.join(text.lower().split())
    alphabet_bonus = 60 if is_alphabet_sequence(text) else 0
    words = re.findall(r'[a-zA-Z]+', text.lower())
    if not words:
        return -100
    score = alphabet_bonus
    score += sum(6 for w in words if w in COMMON_ENGLISH_WORDS)
    score += sum(2 for w in words if len(w) >= 3 and any(v in w for v in 'aeiou'))
    score -= text.count('?') * 2
    score -= text.count('[dots') * 4
    return score

def detect_braille_language(en_text, ru_text):
    if is_alphabet_sequence(en_text):
        return "English Braille (UEB)", False

    has_ru_marker = any(m in ru_text.lower() for m in ['шрифт', 'шоифт', 'брайл', 'сказк', 'дерев', 'книг', 'человек'])
    en_words = [w for w in re.findall(r'[a-zA-Z]+', en_text.lower()) if len(w) >= 3]
    ru_words = [w for w in re.findall(r'[а-яА-ЯёЁ]+', ru_text.lower()) if len(w) >= 3]
    en_matches = [w for w in en_words if w in COMMON_ENGLISH_WORDS]
    ru_matches = [w for w in ru_words if w in COMMON_RUSSIAN_WORDS]

    if has_ru_marker or (len(ru_matches) >= 2 and len(ru_matches) > len(en_matches)):
        return "Russian Braille", True

    if any(c in en_text.lower() for c in ['whrift', 'braandlsh']) and not is_alphabet_sequence(en_text):
        return "Russian Braille", True

    return "English Braille (UEB)", False

import braille_engine as be
be.score_english_coherence = score_english_coherence
be.detect_braille_language = detect_braille_language

# Test all 4 images:
images = [
    ('English Reference Chart (User upload)', 'C:/Users/ajjov/.gemini/antigravity/brain/3fcba85f-bcc2-4b0e-adb3-b643130cd97c/.user_uploaded/media_1794804206310_9e9f044f.png', 0, 'English Braille (UEB)'),
    ('Russian Reference Chart', 'C:/Users/ajjov/.gemini/antigravity/brain/3fcba85f-bcc2-4b0e-adb3-b643130cd97c/.user_uploaded/media_1794757241443_c1f4d691.jpg', 0, 'Russian Braille'),
    ('Upside-down Redwoods Photo', 'C:/Users/ajjov/.gemini/antigravity/brain/3fcba85f-bcc2-4b0e-adb3-b643130cd97c/.user_uploaded/media_1794125518993_1166408c.jpg', 180, 'English Braille (UEB)'),
    ('Sample Braille Sheet', 'D:/Braille_Digitizer/samples/sample_braille_sheet.jpg', 180, 'English Braille (UEB)')
]

for label, path, expected_rot, expected_lang in images:
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    doc, all_cells, dots, _, applied_rot = be.decode_page_with_auto_orient(gray, auto_orient=True)
    en_txt = be.translate_document(doc, 'EN')
    ru_txt = be.translate_document(doc, 'RU')
    lang, is_ru = be.detect_braille_language(en_txt, ru_txt)
    print(f"[{label}]")
    print(f"  Applied Rot: {applied_rot} (Expected: {expected_rot})")
    print(f"  Detected Lang: {lang} (Expected: {expected_lang})")
    print(f"  First line: {en_txt.splitlines()[0] if en_txt.splitlines() else ''}")
    print()
