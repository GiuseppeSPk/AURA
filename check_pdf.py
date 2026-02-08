
import sys
import os

# Try to use a common tool if available, or just a simple check.
# Since I cannot install new packages easily, I will try to use the browser tool to "read" the PDF if possible, 
# but first let's see if I can find any existing pdftotext or similar in the system.
# Alternatively, I can try to read the first few bytes to see if it's a valid PDF.

path = r"C:\Users\spicc\Desktop\Multimodal\AURA-Advanced-Understanding-and-Reporting-of-Aggression (1).pdf"

if os.path.exists(path):
    print(f"File found: {path}")
    print(f"Size: {os.path.getsize(path)} bytes")
else:
    print("File NOT found.")
