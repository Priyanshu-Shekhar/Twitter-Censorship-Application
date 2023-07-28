
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tkinter as tk
import customtkinter as ctk
import numpy as np


def analyze_tweet():
    tweet = entry.get()

    # preprocess tweet
    tweet_words = []

    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)

    tweet_proc = " ".join(tweet_words)

    # load model and tokenizer
    roberta = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    labels = ['Negative', 'Neutral', 'Positive']

    # sentiment analysis
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    output = model(**encoded_tweet)

    scores = output.logits.detach().numpy()[0]
    scores = np.exp(scores) / np.sum(np.exp(scores))

    # display results
    result = ''
    for i in range(len(scores)):
        l = labels[i]
        s = scores[i]
        result += f"{l}: {s*100:.2f}%\n"
    print(" ")
    print(tweet) 
    print(result)
        
    sentiment_idx = np.argmax(scores)
    
    sentiment = labels[sentiment_idx]
    if (sentiment == "Positive"):
        label_result.configure(text=sentiment, text_color = "#2cb322")

    elif (sentiment == "Negative"):
        label_result.configure(text=sentiment, text_color = "#d61818")
    
    else:
       label_result.configure(text=sentiment, text_color = "#d6d618") 
    


window = ctk.CTk()
window.title('Tweet Classifier')
window.geometry('590x400')

label = ctk.CTkLabel(window, text="Tweet/Post Moderation", padx=25, pady=35, text_color="#0da9d9", font=("", 30))
label.pack()

entry = ctk.CTkEntry(window, width=450, height=65, text_color="#ffffff", placeholder_text="Enter tweet",
                      placeholder_text_color="grey", corner_radius=250)
entry.pack()

label_space = ctk.CTkLabel(window, text="", padx=3, pady=3)
label_space.pack()

button = ctk.CTkButton(window, text="Analyze", width=25, height=35, corner_radius=450, fg_color="transparent",
                       text_color="#ffffff", border_color="#c1c2c0", border_width=1, command=analyze_tweet)
button.pack()

label_space2 = ctk.CTkLabel(window, text="", padx=3, pady=3)
label_space2.pack()


label_result = ctk.CTkLabel(window, text="", font=("San Serif", 30))
label_result.pack()

window.mainloop()

