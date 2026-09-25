import os
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_ID = "Ayaz4oo9/resume-fit-classifier"
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model.eval()

labels = ["No Fit", "Potential Fit", "Good Fit"]

def predict(resume_text, job_description):
    text = resume_text + " [SEP] " + job_description
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    return {labels[i]: float(probs[i]) for i in range(3)}

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Textbox(lines=8, label="Resume"), gr.Textbox(lines=8, label="Job Description")],
    outputs=gr.Label(num_top_classes=3, label="Fit Classification"),
    title="Resume–Job Fit Classifier",
    description="Paste a resume and a job description to see how well they match."
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
