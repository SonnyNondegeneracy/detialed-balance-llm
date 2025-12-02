from pywheels import get_answer
import numpy as np
prefix = "If the 26 English letters A B C D E F G H I J K L M N O P Q R S T U V W X Y Z correspond to the numbers 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 respectively, then ~ Knowledge K + N + O + W + L + E + D + G + E = 11 + 14 + 15 + 23 + 12 + 5 + 4 +7 +5 = 96% Workhard W + O + R + K + H + A + R + D = 23 + 15 + 18 + 11 + 8 + 1 +18 + 4 = 98% This means that knowledge and hard work can contribute up to 96% and 98% to our lives. Is it Luck? L + U + C+ K = 12 + 21 + 3 + 11 = 47% Is it Love? L + O+ V + E = 12 +15+ 22 + 5 = 54% It seems that these things we usually consider important do not play the most crucial role. So, what can determine our life at 100%? Is it Money? M + O + N + E + Y = 13 +15 +14 + 5 + 25 = 72% It seems not."
# Use AI to find a word that is exactly 100%, get feedback for each result, and maintain a dictionary until found.
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import json

def calculate_percentage(word):
    # Define the mapping from letters to numbers
    letter_to_value = {chr(ord('A') + i): i + 1 for i in range(26)}
    
    # Calculate the sum of the values of each letter in the word
    total_value = sum(letter_to_value.get(letter.upper(), 0) for letter in word)
    
    # Calculate the percentage
    percentage = total_value
    
    return percentage

def is_word(word):
    return word.isalpha()

database = {"BUZZY": []}
lock = threading.Lock()
num_tasks = 20000
max_workers = 100
model = ['gpt5-nano', 'gemini-2.5-flash-nothinking', 'claude-4'][3]# choose one model

def worker(_):
    try:
        # pick a random existing word in a thread-safe way
        with lock:
            keys = list(database.keys())
        words = random.choice(keys)

        prompt = (
            f"{prefix}"
            f"An example word is {words}, whose letters add up to 100%. "
            "Please give me a new word whose letters add up to 100%. "
            "Only provide the word without any additional explanation."
        )

        suggestion = get_answer(prompt, model=model)
        if not isinstance(suggestion, str):
            suggestion = str(suggestion)
        suggestion = suggestion.strip().upper()

        if is_word(suggestion) and suggestion != words:
            with lock:
                if calculate_percentage(suggestion) == 100:
                    pass
                else:
                    suggestion = 'INVALID00'
                if suggestion not in database:
                    database[suggestion] = []
                database[words].append(suggestion)

            return f"Found Edges: {words} -> {suggestion}"
        else:
            return f"Rejected suggestion: {suggestion} from {words}"
    except Exception as e:
        return f"Error in worker: {e}"

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(worker, i) for i in range(num_tasks)]
    for fut in as_completed(futures):
        print(fut.result())

data_path = f"./data/word_count_{model}.json"
with open(data_path, 'w', encoding='utf-8') as f:
    json.dump(database, f, indent=4, ensure_ascii=False)

print(f"Final Database saved in {data_path}")
