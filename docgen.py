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