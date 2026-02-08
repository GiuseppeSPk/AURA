
import sys

try:
    import pypdf
    print("pypdf is installed")
    reader = pypdf.PdfReader(r"C:\Users\spicc\Desktop\Multimodal\AURA-Advanced-Understanding-and-Reporting-of-Aggression (1).pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    print("---EXTRACTED TEXT START---")
    print(text[:2000]) # First 2000 chars
    print("---EXTRACTED TEXT END---")
except ImportError:
    print("pypdf NOT installed")
    try:
        import fitz
        print("PyMuPDF (fitz) is installed")
        doc = fitz.open(r"C:\Users\spicc\Desktop\Multimodal\AURA-Advanced-Understanding-and-Reporting-of-Aggression (1).pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        print("---EXTRACTED TEXT START---")
        print(text[:2000])
        print("---EXTRACTED TEXT END---")
    except ImportError:
        print("PyMuPDF NOT installed")
except Exception as e:
    print(f"Error: {e}")
