import os
import gradio as gr
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from pypdf import PdfReader
import docx2txt

MODEL_ID = "Ayaz4oo9/resume-fit-classifier-onnx"
model = ORTModelForSequenceClassification.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

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
    inputs = tokenizer(text, return_tensors="np", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits[0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
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

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))demo = gr.Interface(
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
