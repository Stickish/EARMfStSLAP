import re
from nltk.tokenize import word_tokenize


# None of these five are used - redundant?
comma_to_whitespace = lambda x : x.replace(',', ' ')
whitespace_to_underscore = lambda x : x.replace(' ', '_')
ampersand_to_and = lambda x : x.replace('&', 'och')
hyphen_to_whitespace = lambda x : x.replace('-', ' ')
slash_to_whitespace = lambda x : x.replace('/', ' ')


def prune_insurance(x):
    try:
        idx  = x.index('(Li')
    except ValueError as e:
        idx = len(x)
    
    x = x[:idx]
    
    return x


def standardize_units(x):
    # Find times where we have <d,d> e.x 1,2 and ignore them, otherwise remove comma
    # Också kolla om det finns <d,dd> e.x 0,25
    x = x.replace('.', ',')
    x = x.replace('-', ' ') # Might want to change to whitespace
    x = x.replace('/', ' ')
    
    one_decimal = re.compile('[\d][,][\d][mcWAkgVl\"]') # 1,2A
    two_decimal = re.compile('[\d][,][\d][\d][mcAWkgVl\"]') #0,25m
    one_letter = re.compile('[\d][ ][mcWkgVl\"][ gm]') # sets within sets? kr doesnt get caught currently
    decimal_no_letter = re.compile('[\d][,][\d]')
    
    iterator_1l = one_letter.finditer(x)
    
    for match in iterator_1l:
        s = match.group()
        r = s[0] + s[2:]
        x = x.replace(s, r)
    
    iterator_1d = one_decimal.finditer(x)
    
    for match in iterator_1d:
        s = match.group()
        r = s[0] + s[2] + s[3]
        x = x.replace(s, r)
    
    iterator_2d = two_decimal.finditer(x)
    
    for match in iterator_2d:
        s = match.group()
        r = s[0] + s[2] + s[3] + s[4]
        x = x.replace(s, r)
    
    iterator_decimal = decimal_no_letter.finditer(x)
    
    for match in iterator_decimal:
        s = match.group()
        r = s[0] + s[2]
        x = x.replace(s, r)
    
    x = x.replace(',', '')
    x = x.replace('&', ' ')
    return x


def add_underscore(x):
    pattern = re.compile('[ ][\w][ ]')
    iterator = pattern.finditer(x)
    
    for match in iterator:
        s = match.group()
        r = ' _' + s[1] + '_ '
        x = x.replace(s, r)
        
        
    return x


def remove_words(x):
    # words to remove: [med, i, för, till]
    pattern_1 = re.compile('[ ][m][e][d][ ]')
    pattern_2 = re.compile('[ ][f][ö][r][ ]')
    pattern_3 = re.compile('[ ][i][ ]')
    pattern_4 = re.compile('[ ][t][i][l][l][ ]')
    patterns = [pattern_1, pattern_2, pattern_3, pattern_4]
    
    for p in patterns:
        iterator = p.finditer(x)
        
        for match in iterator:
            s = match.group()
            x = x.replace(s, ' ')

    return x


# Important to call after lower()
def combine_words(x):
    patterns = []
    patterns.append(re.compile('[u][s][b][ ][c]'))
    
    for p in patterns:
        iterator = p.finditer(x)
        for match in iterator:
            s = match.group()
            r = 'usbc'
            x = x.replace(s, r)
    
    return x
    

def prune_whitespace(x):
    s = word_tokenize(x)
    tmp = ' '.join(s)
    
    return tmp

    
def get_preprocessing_functions():
    """
    Creates and returns a list of functions that are to be applied in order to the article names
    """
    preprocessing_functions = [
                               lambda x : x + ' ',              # Add whitespace at the end
                               standardize_units,               # Standardize 50A and 50 A to 50A
                               prune_insurance,                 # Removes mathces to <livst...>
                               lambda x : x.lower(),            # Lowercases the word
                               lambda x : x.replace('(', ' '),  # Removes parenthesis
                               lambda x : x.replace(')', ' '),  # Removes parenthesis
                               combine_words,                   # Combine usb c to usbc
                               remove_words,                    # Removes common words that doesn't add meaning
                               add_underscore,                  # adds _ _ to numbers and letters and separates them
                               add_underscore,                  # adds _ _ to numbers and letters and separates them
                               prune_whitespace                 # Removes extra whitespaces
                              ]      
    return preprocessing_functions