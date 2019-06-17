import numpy as np
import random
import json
import re


def generate_random_start(model, graph, seed_length=50,new_words=50,diversity=1,return_output=False,n_gen=1):
    """Generate `new_words` words of output from a trained model and format into HTML."""

    word_idx = json.load(open('data/word-index.json'))
    idx_word = {idx: word for word, idx in word_idx.items()}

    sequences = json.load(open('data/sequences.json'))

    # Choose a random sequence
    seq = random.choice(sequences)

    # Choose a random starting point
    seed_idx = random.randint(0, len(seq) - seed_length - 10)
    # Ending index for seed
    end_idx = seed_idx + seed_length

    gen_list = []
    with graph.as_default():
        for n in range(n_gen):
            # Extract the seed sequence
            seed = seq[seed_idx:end_idx]
            original_sequence = [idx_word[i] for i in seed]
            generated = seed[:] + ['#']

            # Find the actual entire sequence
            actual = generated[:] + seq[end_idx:end_idx + new_words]

            # Keep adding new words
            for i in range(new_words):

                # Make a prediction from the seed
                preds = model.predict(np.array(seed).reshape(1, -1))[0].astype(np.float64)

                # Diversify
                preds = np.log(preds) / diversity
                exp_preds = np.exp(preds)

                # Softmax
                preds = exp_preds / sum(exp_preds)

                # Choose the next word
                probas = np.random.multinomial(1, preds, 1)[0]

                next_idx = np.argmax(probas)

                # New seed adds on old word
                #             seed = seed[1:] + [next_idx]
                seed += [next_idx]
                generated.append(next_idx)

        # Showing generated and actual abstract
        n = []

        for i in generated:
            n.append(idx_word.get(i, '==='))

        gen_list.append(n)

    a = []

    for i in actual:
        a.append(idx_word.get(i, '==='))

    a = a[seed_length:]

    gen_list = [gen[seed_length:seed_length + len(a)] for gen in gen_list]

    if return_output:
        return original_sequence, gen_list, a
	
    # HTML formatting
    seed_html = ''
    seed_html = addContent(seed_html, header(
        'Seed Sequence', color='darkblue'))
    seed_html = addContent(seed_html,box(remove_spaces(' '.join(original_sequence))))

    gen_html = ''
    gen_html = addContent(gen_html, header('RNN Generated', color='darkred'))
    gen_html = addContent(gen_html, box(remove_spaces(' '.join(gen_list[0]))))

    a_html = ''
    a_html = addContent(a_html, header('Actual', color='darkgreen'))
    a_html = addContent(a_html, box(remove_spaces(' '.join(a))))
	
    st = "<div>" + seed_html + "</div><div>" + gen_html + "</div><div>" + a_html + "</div>" 
    #return f"<div>{seed_html}</div><div>{gen_html}</div><div>{a_html}</div>"
    return st

def generate_from_seed(model, graph, seed,new_words=50, diversity=0.75):
    """Generate output from a sequence"""

    # Mapping of words to integers
    word_idx = json.load(open('data/word-index.json')) 
    idx_word = {idx: word for word, idx in word_idx.items()}

    # Original formated text
    start = format_sequence(seed).split()
    gen = []
    s = start[:]

    with graph.as_default():

        # Generate output
        for _ in range(new_words):
            # Conver to array
            x = np.array([word_idx.get(word, 0)
                          for word in s]).reshape((1, -1))

            # Make predictions
            preds = model.predict(x)[0].astype(float)

            # Diversify
            preds = np.log(preds) / diversity
            exp_preds = np.exp(preds)
            # Softmax
            preds = exp_preds / np.sum(exp_preds)

            # Pick next index
            next_idx = np.argmax(np.random.multinomial(1, preds, size=1))
            s.append(idx_word[next_idx])
            gen.append(idx_word[next_idx])

    # Formatting in html
    start = remove_spaces(' '.join(start)) + ' '
    gen = remove_spaces(' '.join(gen))
    html = ''
    html = addContent(html, header(
        'Input Seed ', color='black', gen_text='Network Output'))
    html = addContent(html, box(start, gen))
    st = "<div>"+html+"</div>"
    return st


def header(text, color='black', gen_text=None):
    """Create an HTML header"""

    if gen_text:
        raw_html = '<h1 style="margin-top:16px;color: {color};font-size:54px"><center>' + str(
            text) + '<span style="color: red">' + str(gen_text) + '</center></h1>'
    else:
        raw_html = '<h1 style="margin-top:12px;color: {color};font-size:54px"><center>' + str(
            text) + '</center></h1>'
    return raw_html


def box(text, gen_text=None):
    """Create an HTML box of text"""

    if gen_text:
        raw_html = '<div style="padding:8px;font-size:28px;margin-top:28px;margin-bottom:14px;">' + str(
            text) + '<span style="color: red">' + str(gen_text) + '</div>'

    else:
        raw_html = '<div style="border-bottom:1px inset black;border-top:1px inset black;padding:8px;font-size: 28px;">' + str(
            text) + '</div>'
    return raw_html


def addContent(old_html, raw_html):
    """Add html content together"""

    old_html += raw_html
    return old_html


def format_sequence(s):
    """Add spaces around punctuation and remove references to images/citations."""

    # Add spaces around punctuation
    s = re.sub(r'(?<=[^\s0-9])(?=[.,;?])', r' ', s)

    # Remove references to figures
    s = re.sub(r'\((\d+)\)', r'', s)

    # Remove double spaces
    s = re.sub(r'\s\s', ' ', s)
    return s


def remove_spaces(s):
    """Remove spaces around punctuation"""

    s = re.sub(r'\s+([.,;?])', r'\1', s)

    return s