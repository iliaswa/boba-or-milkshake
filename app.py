import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('milkshake_model.pkl')

categories = learn.dls.vocab

def classify_img(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
title = "Is it Boba or a Milkshake?"
description = "A 'Boba vs. Milkshake' classifier trained via convolutional neural network. Transfer learning based on Resnet50 with fine-tuned parameters."
examples = ["boba2.jpg", "milkshake3.jpg", "boba1.png"]


demo = gr.Interface(fn=classify_img, inputs="image", outputs="label", title=title, description=description, examples=examples)
demo.launch()