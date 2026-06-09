from pathlib import Path
from markitdown import MarkItDown

INPUT_DIR = Path(__file__).parent / "input"
OUTPUT_DIR = Path(__file__).parent / "output"

def convert_all():
    pdfs = list(INPUT_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDF files found in the input folder.")
        return

    md = MarkItDown()
    print(f"Found {len(pdfs)} PDF(s). Converting...\n")

    for pdf in pdfs:
        out_file = OUTPUT_DIR / (pdf.stem + ".md")
        try:
            result = md.convert(str(pdf))
            out_file.write_text(result.text_content, encoding="utf-8")
            print(f"  [OK] {pdf.name} -> output/{out_file.name}")
        except Exception as e:
            print(f"  [FAIL] {pdf.name}: {e}")

    print("\nDone.")

if __name__ == "__main__":
    convert_all()
