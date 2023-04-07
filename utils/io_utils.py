"""Some utils for input prompts and output generations"""
import sys
from termcolor import colored

def get_input():
    """Gets multiline input until EOF"""
    line = ""
    user_input = ""
    try:
        while True:
            line = input()
            if len(user_input) > 0:
                line = "\n" + line
            user_input += line
    except EOFError:
        pass
    return user_input


def delete_last_line():
    "Use this function to delete the last line in the STDOUT"
    #cursor up one line
    sys.stdout.write('\x1b[1A')
    #delete last line
    sys.stdout.write('\x1b[2K')

def labeled_input(label):
    """Print colored label, then ask for input"""
    print(colored(label, 'green'))
    return get_input().strip()

def wait_and_answer(generation_function, title='Answer:'):
    """Prints generating while generating, then prints an answer"""
    print(colored(title, 'green'))
    print(colored('[generating]', 'yellow'))
    result = generation_function()
    delete_last_line()
    print(colored(result, 'blue'))
