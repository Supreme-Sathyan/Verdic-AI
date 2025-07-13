import os
import json
from fpdf import FPDF

# Step 1: Save JSON from string
def save_json_from_string(json_string, output_file="witness_summon.json"):
    json_string = json_string[7:-3]  # Remove ```json and ```
    try:
        data = json.loads(json_string)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"✅ JSON saved successfully at {output_file}")
        return output_file
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Step 2: Generate PDF from JSON
def generate_witness_summon_pdf(json_data, output_path="witness_summon.pdf"):
    try:
        data = json_data["witness_summon"]

        # Extract fields
        witness_name = data["witness_name"]
        witness_address = data["witness_address"]
        court_name = data["court_name"]
        court_address = data["court_address"]
        hearing_date = data["hearing_date"]
        hearing_time = data["hearing_time"]
        case_name = data["case_name"]
        case_number = data["case_number"]
        testimony_purpose = data["testimony_required_for"]
        issued_date = data["issued_date"]
        issued_by = data["issued_by"]
        contact_name = data["contact_person_for_rescheduling"]["name"]

        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "WITNESS SUMMONS", ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, f"To: {witness_name}\n{witness_address}\n[City, State, Zip Code]")
        pdf.ln(5)

        pdf.multi_cell(0, 10,
            f"YOU ARE HEREBY SUMMONED to appear in the {court_name}, located at {court_address}, "
            f"on {hearing_date} at {hearing_time} to testify as a witness in the case of {case_name}, "
            f"Case Number {case_number}."
        )
        pdf.ln(5)

        pdf.multi_cell(0, 10,
            f"Your testimony is required in connection with the {testimony_purpose}"
        )
        pdf.ln(5)

        pdf.multi_cell(0, 10,
            "Failure to comply with this summons may result in a contempt of court charge and potential legal consequences, including fines or arrest."
        )
        pdf.ln(5)

        pdf.multi_cell(0, 10, f"Issued on: {issued_date}")
        pdf.multi_cell(0, 10, f"Issued by: {issued_by}")
        pdf.multi_cell(0, 10, f"Attorney for: Plaintiff")
        pdf.ln(10)

        pdf.cell(0, 10, "Signature of the Clerk or Issuer: ___________________", ln=True)
        pdf.cell(0, 10, "Clerk's Name or Title: ___________________", ln=True)
        pdf.ln(15)

        # Certificate of Service
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Certificate of Service", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10,
            f"I, {contact_name}, hereby certify that on {issued_date}, I served a copy of this witness summons on {witness_name} "
            f"by [method of service, e.g., personal delivery, mail, email, etc.], at {witness_address}."
        )
        pdf.ln(10)
        pdf.cell(0, 10, "Signature of Server: ___________________", ln=True)
        pdf.cell(0, 10, "Date: ___________________", ln=True)

        pdf.output(output_path)
        print(f"✅ PDF saved at: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return None

# Step 3: Run Both
data_string = '''```json
{
  "witness_summon": {
    "case_name": "John Doe v. Jane Smith",
    "case_number": "12345-2024",
    "court_name": "Superior Court of [County], [State]",
    "court_address": "123 Court Street, City, State, ZIP",
    "hearing_date": "May 15, 2024",
    "hearing_time": "10:00 AM",
    "witness_name": "[Witness Name - Not explicitly stated in the provided text]",
    "witness_address": "[Witness Address - Not explicitly stated in the provided text]",
    "testimony_required_for": "Key facts related to a contract dispute between the plaintiff (John Doe) and defendant (Jane Smith), specifically regarding the signed agreement dated March 1, 2024, and any subsequent communications or meetings relevant to the dispute.",
    "required_documents_or_evidence": "Any documents, photographs, or other evidence relevant to the case.",       
    "submission_instructions_for_evidence": "Bring to court or submit to Alice Johnson at Law Firm Name, 456 Legal Lane, City, State, ZIP prior to the hearing.",
    "consequences_of_failure_to_appear": "Serious legal consequences, including contempt of court, fines, or arrest.",
    "contact_person_for_rescheduling": {
      "name": "Alice Johnson",
      "title": "Attorney for the Plaintiff",
      "phone": "(555) 123-4567",
      "email": "alice.johnson@lawfirm.com"
    },
    "issued_date": "May 1, 2024",
    "issued_by": "Alice Johnson"
  }
}
```'''

json_file = save_json_from_string(data_string)
if json_file:
    with open(json_file) as f:
        json_data = json.load(f)
    generate_witness_summon_pdf(json_data)
