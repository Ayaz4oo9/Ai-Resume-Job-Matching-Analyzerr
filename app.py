import os
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from pypdf import PdfReader
import docx2txt

MODEL_ID = "Ayaz4oo9/resume-fit-classifier"
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    low_cpu_mem_usage=True,
    dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model.eval()

labels = ["No Fit", "Potential Fit", "Good Fit"]

def extract_text(file_path):
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        return " ".join(page.extract_text() or "" for page in reader.pages)
    elif file_path.lower().endswith(".docx"):
        return docx2txt.process(file_path)
    else:
        with open(file_path, "r", errors="ignore") as f:
            return f.read()

def predict(resume_file, job_description):
    if resume_file is None:
        return {"Error": 1.0}
    resume_text = extract_text(resume_file.name)
    text = resume_text + " [SEP] " + job_description
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    return {labels[i]: float(probs[i]) for i in range(3)}

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.File(label="Upload Resume (PDF or DOCX)", file_types=[".pdf", ".docx"]),
        gr.Textbox(lines=8, label="Job Description")
    ],
    outputs=gr.Label(num_top_classes=3, label="Fit Classification"),
    title="Resume–Job Fit Classifier",
    description="Upload a resume and paste a job description to see how well they match."
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
