import gradio as gr
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ----------------------------
# LOAD MODEL
# ----------------------------
model_path = "Skin model"

processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path)
model.eval()

# ----------------------------
# KEEP YOUR ORIGINAL DATA
# ----------------------------
disease_info = {
    # 🔴 HIGH RISK
"Melanoma Skin Cancer Nevi and Moles": {
    "severity": "HIGH",
    "risk": "Malignant (life-threatening)",
    "tests": "Excisional biopsy, dermoscopy, sentinel lymph node biopsy",
    "treatment": "Wide excision, oncology referral",
    "note": "ABCDE rule, rapid evolution, high metastasis risk"
},

"Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": {
    "severity": "HIGH",
    "risk": "Pre-cancer / cancer",
    "tests": "Biopsy",
    "treatment": "Cryotherapy, topical 5-FU, Mohs surgery",
    "note": "UV-induced lesions, progression risk"
},

"Lupus and other Connective Tissue diseases": {
    "severity": "HIGH",
    "risk": "Autoimmune systemic",
    "tests": "ANA, dsDNA, biopsy",
    "treatment": "Steroids, hydroxychloroquine",
    "note": "Photosensitive rash, systemic involvement"
},

"Vasculitis Photos": {
    "severity": "HIGH",
    "risk": "Systemic inflammation",
    "tests": "Biopsy, urinalysis, blood tests",
    "treatment": "Steroids, treat underlying cause",
    "note": "Palpable purpura, organ involvement"
},

"Cellulitis Impetigo and other Bacterial Infections": {
    "severity": "HIGH",
    "risk": "Infection (systemic spread)",
    "tests": "Clinical ± culture",
    "treatment": "Antibiotics (cephalexin, clindamycin)",
    "note": "Rapid progression, fever possible"
},

"Monkeypox": {
    "severity": "HIGH",
    "risk": "Contagious viral",
    "tests": "PCR",
    "treatment": "Supportive, tecovirimat",
    "note": "Isolation required"
},

"Systemic Disease": {
    "severity": "HIGH",
    "risk": "Underlying systemic pathology",
    "tests": "CBC, metabolic panel, imaging",
    "treatment": "Treat underlying disease",
    "note": "May indicate internal disease"
},

# 🟠 MODERATE
"Psoriasis pictures Lichen Planus and related diseases": {
    "severity": "MODERATE",
    "risk": "Chronic inflammatory",
    "tests": "Clinical ± biopsy",
    "treatment": "Topical steroids, biologics",
    "note": "Check for arthritis"
},

"Eczema Photos": {
    "severity": "MODERATE",
    "risk": "Inflammatory",
    "tests": "Clinical",
    "treatment": "Moisturizers, steroids",
    "note": "Barrier dysfunction"
},

"Atopic Dermatitis Photos": {
    "severity": "MODERATE",
    "risk": "Allergic/inflammatory",
    "tests": "Clinical",
    "treatment": "Emollients, steroids",
    "note": "Associated with asthma"
},

"Exanthems and Drug Eruptions": {
    "severity": "MODERATE",
    "risk": "Drug reaction",
    "tests": "Clinical ± biopsy",
    "treatment": "Stop drug, antihistamines",
    "note": "Watch for SJS/TEN"
},

"Herpes HPV and other STDs Photos": {
    "severity": "MODERATE",
    "risk": "Infectious",
    "tests": "PCR, serology",
    "treatment": "Antivirals",
    "note": "Check HIV if severe"
},

"Tinea Ringworm Candidiasis and other Fungal Infections": {
    "severity": "MODERATE",
    "risk": "Infectious",
    "tests": "KOH scraping",
    "treatment": "Topical/oral antifungals",
    "note": "Common recurrence"
},

"Scabies Lyme Disease and other Infestations and Bites": {
    "severity": "MODERATE",
    "risk": "Parasitic",
    "tests": "Clinical",
    "treatment": "Permethrin, ivermectin",
    "note": "Treat contacts"
},

"Bullous Disease Photos": {
    "severity": "HIGH",
    "risk": "Autoimmune blistering",
    "tests": "Biopsy + immunofluorescence",
    "treatment": "Steroids, immunosuppressants",
    "note": "Can be life-threatening"
},

# 🟡 LOW–MODERATE
"Seborrheic Keratoses and other Benign Tumors": {
    "severity": "LOW",
    "risk": "Benign",
    "tests": "Clinical",
    "treatment": "None / cosmetic removal",
    "note": "Biopsy if atypical"
},

"Vascular Tumors": {
    "severity": "LOW",
    "risk": "Usually benign",
    "tests": "Clinical",
    "treatment": "Observation / propranolol",
    "note": "Monitor growth"
},

"Vasculitis Photos": {
    
    "severity": "VARIABLE (Mild to Life-threatening)",
    "risk": "Organ damage / Aneurysm",
    "tests": "Biopsy / Blood tests (ANCA, CRP, ESR)",
    "treatment": "Corticosteroids / Immunosuppressants",
    "note": "Requires long-term monitoring for flares"
},

"Light Diseases and Disorders of Pigmentation": {
    "severity": "LOW",
    "risk": "Cosmetic",
    "tests": "Clinical",
    "treatment": "Topicals, photoprotection",
    "note": "Rule out melanoma if suspicious"
},

# 🟢 LOW
"Acne and Rosacea Photos": {
    "severity": "LOW",
    "risk": "Inflammatory",
    "tests": "None",
    "treatment": "Topicals, antibiotics",
    "note": "Avoid triggers"
},

"Hair Loss Photos Alopecia and other Hair Diseases": {
    "severity": "LOW",
    "risk": "Autoimmune/cosmetic",
    "tests": "Thyroid, iron (if needed)",
    "treatment": "Minoxidil, steroids",
    "note": "Check systemic causes"
},

"Nail Fungus and other Nail Disease": {
    "severity": "LOW",
    "risk": "Fungal",
    "tests": "KOH test",
    "treatment": "Terbinafine",
    "note": "Slow response"
},

"Warts Molluscum and other Viral Infections": {
    "severity": "LOW",
    "risk": "Viral",
    "tests": "Clinical",
    "treatment": "Cryotherapy, salicylic acid",
    "note": "Self-limited"
},

"normal skin": {
    "severity": "NONE",
    "risk": "Healthy",
    "tests": "None",
    "treatment": "None",
    "note": "Routine monitoring"
}
}

description_data = {

    "Melanoma Skin Cancer Nevi and Moles": "Aggressive malignant tumor with high metastasis.",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": "Pre-cancerous or cancerous lesions caused by sun exposure.",
    "Lupus and other Connective Tissue diseases": "Autoimmune disorders affecting skin and internal organs.",
    "Vasculitis Photos": "Inflammation of blood vessels causing skin lesions.",
    "Cellulitis Impetigo and other Bacterial Infections": "Bacterial infections causing redness, swelling, and pain.",
    "Monkeypox": "Contagious viral infection with characteristic skin lesions.",
    "Systemic Disease": "Skin manifestations of underlying systemic conditions.",

    "Psoriasis pictures Lichen Planus and related diseases": "Autoimmune skin disease causing rapid cell turnover.",
    "Eczema Photos": "Inflammatory condition causing dry, itchy, and irritated skin.",
    "Atopic Dermatitis Photos": "Chronic allergic skin condition linked with asthma.",
    "Exanthems and Drug Eruptions": "Skin reactions caused by infections or medications.",
    "Herpes HPV and other STDs Photos": "Viral infections affecting skin and mucous membranes.",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "Fungal infections affecting skin, hair, or nails.",
    "Scabies Lyme Disease and other Infestations and Bites": "Parasitic infestations causing itching and irritation.",
    "Bullous Disease Photos": "Autoimmune blistering disorders affecting skin layers.",

    "Seborrheic Keratoses and other Benign Tumors": "Non-cancerous skin growths with no serious risk.",
    "Vascular Tumors": "Abnormal growth of blood vessels, usually benign.",
    "Light Diseases and Disorders of Pigmentation": "Conditions affecting skin pigmentation and color.",

    "Acne and Rosacea Photos": "Common inflammatory conditions affecting facial skin.",
    "Hair Loss Photos Alopecia and other Hair Diseases": "Hair loss due to autoimmune, hormonal, or genetic causes.",
    "Nail Fungus and other Nail Disease": "Fungal infections causing nail discoloration and thickening.",
    "Warts Molluscum and other Viral Infections": "Benign viral infections causing skin growths.",

    "normal skin": "Healthy skin without any visible disease."
}


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))  # only resize, no crop
    return image

# ----------------------------
# PREDICT FUNCTION
# ----------------------------
def predict(image):
    image = preprocess_image(image)
    inputs = processor(images=image, return_tensors="pt")
   

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.nn.functional.softmax(logits, dim=1)[0]
    labels = model.config.id2label

    results = {labels[i]: float(probs[i]) for i in range(len(probs))}
    top3 = sorted(results.items(), key=lambda x: x[1], reverse=True)[:3]

    primary, conf = top3[0]
    conf_percent = conf * 100

    info = disease_info.get(primary, {})

    severity = info.get("severity", "Unknown")
    tests = info.get("tests", "")
    treatment = info.get("treatment", "")
    note = info.get("note", "")

    if severity == "HIGH":
        risk_color = "#fb9696"
    elif severity == "MODERATE":
        risk_color = "#fde68a"
    else:
        risk_color = "#bbf7d0"

    # ----------------------------
    # SMALL PIE CHART
    # ----------------------------
    fig, ax = plt.subplots()
    ax.pie(
        [x[1] for x in top3],
        labels=[x[0] for x in top3],
        autopct='%1.1f%%',
        colors=["#22c55e", "#38bdf8", "#facc15"]
    )
    
    

    # ----------------------------
    # UI BOXES
    # ----------------------------
    summary = f"<div style='padding:20px;background:#bfdbfe;border-radius:10px'><b>{primary}</b><br>{conf_percent:.2f}%</div>"

    risk = f"<div style='padding:20px;background:{risk_color};border-radius:10px'><b>Risk Level:</b><br>{severity}</div>"

    details = "<div style='padding:15px;background:#fef3c7;border-radius:10px'>"
    for name, score in top3:
        details += f"<div style='margin-bottom:8px'><b>{name}</b><br>{score*100:.2f}%</div>"
    details += "</div>"

    clinical = f"<div style='padding:62px;background:#d9f99d;border-radius:10px'><b>Clinical Insight:</b><br>{note}</div>"

    tests_box = f"<div style='padding:20px;background:#e0f2fe;border-radius:10px'><b>Tests Required:</b><br>{tests}</div>"

    recommendation = f"<div style='padding:15px;background:#dcfce7;border-radius:10px'><b>Treatment:</b><br>{treatment}</div>"

    description_box = "<div style='padding:10px;background:#ede9fe;border-radius:10px'><b>Disease Description:</b><br><br>"

    for name, _ in top3:
        desc = description_data.get(name, "")
        description_box += f"<b>{name}</b><br>{desc}<br><br>"
    description_box += "</div>"

    return (
        results, fig,
        summary, risk,
        details, clinical,
        tests_box, recommendation,
        description_box
    )

# ----------------------------
# CSS (FIX EVERYTHING CLEANLY)
# ----------------------------
css = """
/* compact layout */
.gr-row {
    gap: 10px !important;
}

/* NORMAL IMAGE BOX */
.gradio-container .image-container {
    border-radius: 10px !important;
}

button {
    background: linear-gradient(90deg,#22c55e,#38bdf8);
    color:white;
    padding:10px 20px;
    border-radius:10px;
}

/* INPUT TOOL BUTTONS */
button[aria-label] {
    background: #22c55e !important;
    color: white !important;
    border-radius: 40% !important;
}

"""

# ----------------------------
# UI
# ----------------------------
with gr.Blocks() as demo:


    gr.Markdown("""<h1 style='text-align:center; color:#22c55e; font-size:32px'>🏥 Clinical Skin Analysis Dashboard</h1>""")

    with gr.Row():

        # LEFT SIDE
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Skin Image",
                interactive=True
            )

            btn = gr.Button("Analyze", size="sm")

            description_box = gr.Markdown()

        # RIGHT SIDE
        with gr.Column(scale=2):

            with gr.Row():
                label_output = gr.Label(num_top_classes=3)
                chart_output = gr.Plot()

            with gr.Row():
                summary_box = gr.Markdown()
                risk_box = gr.Markdown()

            with gr.Column():

                with gr.Row():
                    details_box = gr.Markdown()
                    clinical_box = gr.Markdown()
                    with gr.Column(): 
                        tests_box = gr.Markdown()
                        recommendation_box = gr.Markdown()   

    btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[
            label_output,
            chart_output,
            summary_box,
            risk_box,
            details_box,
            clinical_box,
            tests_box,
            recommendation_box,
            description_box
        ]
    )

demo.launch()