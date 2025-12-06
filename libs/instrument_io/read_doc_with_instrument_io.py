
import sys
from pathlib import Path

# Add the instrument_io library to the python path
sys.path.append(str(Path(__file__).parent / 'src'))

from instrument_io.readers.docx import DOCXReader

def read_and_summarize_docx(file_path_str):
    reader = DOCXReader()
    file_path = Path(file_path_str)

    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    try:
        paragraphs = reader.read_paragraphs(file_path)
        headings = reader.read_headings(file_path)
        print(f"--- Summary of {file_path.name} (using instrument_io) ---")
        
        # Print first few paragraphs
        print("\nFirst few paragraphs:")
        if paragraphs:
            for i, para_text in enumerate(paragraphs):
                if i >= 5: # Limit to 5 paragraphs
                    break
                text = para_text.strip()
                if text:
                    print(f"- {text}")
        else:
            print("No paragraphs found.")
            
        # Print first few headings
        print("\nFirst few headings:")
        if headings:
            for i, (level, heading_text) in enumerate(headings):
                if i >= 5: # Limit to 5 headings
                    break
                text = heading_text.strip()
                if text:
                    print(f"- {text} (Level: {level})")
        else:
            print("No explicit headings found.")
            
    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")

if __name__ == "__main__":
    file_to_summarize = r"C:\Users\austi\PROJECTS\UC Irvine\Celia Louise Braun Faiola - FaiolaLab\Notebooks\Juan Flores Lab Notebook\BVOC_Sample_Collection_Protocol.docx"
    read_and_summarize_docx(file_to_summarize)
