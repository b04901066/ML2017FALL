import difflib
import numpy as np

def rule_base(context, question, window_size=30):
    context = context.replace('\n', 'N')

    answer_score = []
    question = question.replace('\n', 'N')
    
    for ans in range(len(context)-window_size):
        answer_score.append(difflib.SequenceMatcher(None,
                                                    context[ans : ans + window_size],
                                                    question).ratio())

    start_point = np.argmax(np.asarray(answer_score, dtype=float))
    output = ''
    output_text = ''
    for out in range(start_point, start_point + window_size):
        if question.find(context[out]) == (-1):
            output += str(out)
            output += ' '
            output_text += context[out]

    output = output[:-1]

    return output, output_text
