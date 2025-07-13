import os
import json
from fpdf import FPDF

#Legal Brief

def save_json_from_string(json_string, output_file="legal_brief.json"):
    json_string=json_string[7:-3]
    try:
        # Parse the JSON string into a Python dictionary
        data = json.loads(json_string)

        # Save the parsed JSON data to a file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"✅ JSON saved successfully at {output_file}")
        return output_file
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def generate_legal_brief_pdf_from_json(data, output_dir="output"):
    # Ensure the JSON string is valid and saved properly
    filepath = save_json_from_string(data)
    
    if not filepath:
        print("❌ Failed to save JSON data. Exiting PDF generation.")
        return None
    
    try:
        # Open the saved JSON file and load the data
        with open(filepath, 'r') as file:
            data = json.load(file)
        
        # Convert JSON string to dict if needed
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON input: {e}")

        brief = data.get("Legal Brief", {})
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "LEGAL BRIEF", ln=True, align='C')
        pdf.ln(10)

        def write_field(label, value):
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(40, 10, f"{label}:")
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 10, value)

        # Define fields to include
        fields = [
            ("Case Name", "case_name"),
            ("Case Number", "case_number"),
            ("Plaintiff", "plaintiff"),
            ("Defendant", "defendant"),
            ("Court Name", "court_name"),
            ("Date Filed", "date_filed"),
            ("Date of Order", "date_of_order"),
            ("Summary of Facts", "summary_of_facts"),
            ("Legal Arguments", "legal_arguments"),
            ("Conclusion", "conclusion")
        ]

        # Write the fields to the PDF
        for label, key in fields:
            value = brief.get(key, "N/A")
            pdf.ln(2)
            if label in ["Summary of Facts", "Legal Arguments", "Conclusion"]:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, f"{label}:", ln=True)
                pdf.set_font("Arial", '', 12)
                pdf.multi_cell(0, 10, value)
            else:
                write_field(label, value)

        # Save the PDF to the output directory
        case_num = brief.get("case_number", "unknown").replace("/", "_")
        filename = f"legal_brief_{case_num}.pdf"
        pdf_output_path = os.path.join(output_dir, filename)
        pdf.output(pdf_output_path)

        print(f"✅ PDF saved successfully at {pdf_output_path}")
        return pdf_output_path

    except Exception as e:
        print(f"❌ Error in PDF generation: {e}")
        return None

#Affidavit


def save_json_from_stringa(json_string, output_file="affidavit.json"):
    
    print(json_string)
    try:
        # Parse the JSON string into a Python dictionary
        data = json.loads(json_string)

        # Save the parsed JSON data to a file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"✅ JSON saved successfully at {output_file}")
        return output_file
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON input: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def generate_affidavit_pdf_from_json(data, output_dir="output"):
    
    # Ensure the JSON string is valid and saved properly
    filepath = r"C:\Users\supre\OneDrive\Desktop\New folder (2)\affidavit.json"
    
    if not filepath:
        print("❌ Failed to save JSON data. Exiting PDF generation.")
        return None
    
    try:
        # Open the saved JSON file and load the data
        with open(filepath, 'r') as file:
            data = json.load(file)
        
        # Extract details from the JSON structure
        full_name = data["affiant_details"]["full_name"]
        parent_name = data["affiant_details"]["parent_name"]
        address = data["affiant_details"]["address"]
        statement_of_competence = data["statement_of_competence"]
        declaration = data["declaration"]
        purpose_of_affidavit = data["purpose_of_affidavit"]
        affirmation_of_truth = data["affirmation_of_truth"]
        place_of_signing = data["place_of_signing"]
        date_of_signing = data["date_of_signing"]

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "AFFIDAVIT", ln=True, align='C')
        pdf.ln(10)

        # Add content
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, f"I, {full_name}, son/daughter of {parent_name}, residing at {address}, do hereby solemnly affirm and state as follows:")
        pdf.ln(5)

        pdf.multi_cell(0, 10, f"That I am the deponent herein and I am fully competent to swear this affidavit.")
        pdf.ln(5)

        pdf.multi_cell(0, 10, f"That the facts stated herein are true and correct to the best of my knowledge and belief.")
        pdf.ln(5)

        pdf.multi_cell(0, 10, f"That I am making this affidavit to declare {declaration['subject']}.")
        pdf.multi_cell(0, 10, f"Details of the declaration:")
        pdf.multi_cell(0, 10, f"  - Lost Item: {declaration['lost_item']}")
        pdf.multi_cell(0, 10, f"  - Item Identifier: {declaration['item_identifier']}")
        pdf.multi_cell(0, 10, f"  - Date of Loss: {declaration['date_of_loss']}")
        pdf.multi_cell(0, 10, f"  - Location of Loss: {declaration['location_of_loss']}")
        pdf.multi_cell(0, 10, f"  - Recovery Efforts Outcome: {declaration['recovery_efforts_outcome']}")
        pdf.ln(5)

        pdf.multi_cell(0, 10, f"That I understand this affidavit may be used in legal or official purposes.")
        pdf.ln(5)

        pdf.multi_cell(0, 10, affirmation_of_truth)
        pdf.ln(10)

        # Date and Place
        pdf.cell(0, 10, f"Date: {date_of_signing}", ln=True)
        pdf.cell(0, 10, f"Place: {place_of_signing}", ln=True)
        pdf.ln(10)

        # Signature Line
        pdf.cell(0, 10, "Deponent's Signature: ____________________", ln=True)
        pdf.cell(0, 10, f"Name: {full_name}", ln=True)

        # Save the PDF to the output directory
        filename = f"affidavit_{full_name.replace(' ', '_')}.pdf"
        pdf_output_path = os.path.join(output_dir, filename)
        pdf.output(pdf_output_path)

        print(f"✅ PDF saved successfully at {pdf_output_path}")
        return pdf_output_path

    except Exception as e:
        print(f"❌ Error in PDF generation: {e}")
        return None
    
#witness summons

import json
from fpdf import FPDF

def save_json_from_stringq(json_string, output_file="witness_summon.json"):
    try:
        json_string = json_string.strip()[7:-3]  # Strip ```json and ```
        data = json.loads(json_string)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"✅ JSON saved to {output_file}")
        return output_file
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")
        return None

def generate_witness_summon_pdf(json_string, output_pdf="witness_summon.pdf"):
    json_file = r"C:\Users\supre\OneDrive\Desktop\New folder (2)\witness_summon.json"
    if not json_file:
        return None

    try:
        with open(json_file) as f:
            data = json.load(f)["witness_summon"]

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Header
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "WITNESS SUMMONS", ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, f"To: {data['witness_name']}\n{data['witness_address']}\n[City, State, Zip Code]")
        pdf.ln(5)

        pdf.multi_cell(0, 10,
            f"YOU ARE HEREBY SUMMONED to appear in the {data['court_name']}, located at {data['court_address']}, "
            f"on {data['hearing_date']} at {data['hearing_time']} to testify as a witness in the case of {data['case_name']}, "
            f"Case Number {data['case_number']}."
        )
        pdf.ln(5)

        pdf.multi_cell(0, 10,
            f"Your testimony is required in connection with the {data['testimony_purpose']}"
        )
        pdf.ln(5)

        pdf.multi_cell(0, 10,
            "Failure to comply with this summons may result in a contempt of court charge and potential legal consequences, including fines or arrest."
        )
        pdf.ln(5)

        pdf.multi_cell(0, 10, f"Issued on: {data['issued_date']}")
        pdf.multi_cell(0, 10, f"Issued by: {data['issued_by']}")
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
            f"I, {data['contact_person_for_rescheduling']['name']}, hereby certify that on {data['issued_date']}, "
            f"I served a copy of this witness summons on {data['witness_name']} by [method of service], "
            f"at {data['witness_address']}."
        )
        pdf.ln(10)
        pdf.cell(0, 10, "Signature of Server: ___________________", ln=True)
        pdf.cell(0, 10, "Date: ___________________", ln=True)

        pdf.output(output_pdf)
        print(f"✅ PDF saved at: {output_pdf}")
        return output_pdf
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return None
