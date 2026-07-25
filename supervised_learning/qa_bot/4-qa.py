#!/usr/bin/env python3
"""Interactive question answering loop over a corpus of documents."""
 
 
find_answer = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search
 
 
def question_answer(corpus_path):
    """Answer questions from multiple reference documents.
 
    Args:
        corpus_path (str): path to the corpus of reference documents
    """
    exit_words = {'exit', 'quit', 'goodbye', 'bye'}
 
    while True:
        question = input('Q: ')
 
        if question.strip().lower() in exit_words:
            print('A: Goodbye')
            break
 
        reference = semantic_search(corpus_path, question)
        answer = find_answer(question, reference)
 
        if answer is None:
            answer = 'Sorry, I do not understand your question.'
 
        print('A: {}'.format(answer))
 
