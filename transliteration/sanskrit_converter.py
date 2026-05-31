from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

def main():
    slp1_text = "suKinaH"
    devanagari_text = transliterate(slp1_text, sanscript.SLP1, sanscript.DEVANAGARI)

    with open("sanskrit_output.txt", "w", encoding="utf-8") as f:
        f.write(f"Original SLP1: {slp1_text}\n")
        f.write(f"Devanagari: {devanagari_text}\n")

if __name__ == "__main__":
    main()