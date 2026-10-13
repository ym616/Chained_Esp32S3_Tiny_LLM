import os

with open('llama2.c/data/tok2048.vocab', 'r', encoding='utf-8') as f:
    vocab = [line.strip().split('\t')[0].replace('\u2581', ' ') for line in f if line.strip()]

insults = [
    'My gran could code better than this, and she is dead!',
    'You donkey! What is this absolute garbage?',
    'This code is so raw it is still running in the pasture!',
    'Are you having a laugh? This is an absolute disgrace!',
    'Shut it down! This memory leak is going to burn the whole server down!',
    'It is RAW! Your variables are not even initialized, you idiot sandwich!',
    'Where is the error handling? IT IS MISSING! WHERE IS IT?!',
    'Brownout triggered? More like brain-out triggered! You donkey!'
]

for ins in insults:
    words = ins.split()
    first = words[0]
    matches = [(i, repr(v)) for i, v in enumerate(vocab) if v.strip() == first]
    print(first, '->', matches)

