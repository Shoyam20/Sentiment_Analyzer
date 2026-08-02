import os
import gradio as gr
from transformers import pipeline

classifier =pipeline(
                    "sentiment-analysis"
                     )

def check(text):
  result=classifier(text)
  score=f"{result[0]['score']:.4f}"
  return result[0]["label"],score

app=gr.Interface(
    fn=check,
    inputs=gr.Textbox(
        label="Enter sentiment",
        placeholder="Enter the sentence...",
    ),
        
    outputs=[
        gr.Textbox(label="Analyzed sentiment"),
        gr.Textbox(label="Analyzed Score")
        
    ],

    title="Sentiment Analyzer"

)

app.launch()
