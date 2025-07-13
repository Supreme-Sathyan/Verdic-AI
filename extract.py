import PyPDF2

def extract_pdf_as_one_paragraph(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + " "
    # Remove excessive newlines and extra spaces
    one_paragraph = ' '.join(text.split())
    return one_paragraph