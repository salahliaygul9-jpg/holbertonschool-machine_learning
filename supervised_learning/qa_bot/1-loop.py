#!/usr/bin/env python3
"""Interactive loop that echoes user input under a Q/A prompt."""
 
 
def main():
    """Prompt the user with Q: and respond with A:.
 
    Exits and prints 'A: Goodbye' when the user enters one of
    'exit', 'quit', 'goodbye', or 'bye' (case insensitive).
    """
    exit_words = ('exit', 'quit', 'goodbye', 'bye')
 
    while True:
        question = input('Q: ')
 
        if question.lower() in exit_words:
            print('A: Goodbye')
            break
 
        print('A:')
 
 
if __name__ == '__main__':
    main()
 
