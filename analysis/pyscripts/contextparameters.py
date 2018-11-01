'''
This module contains parameters to be used
with contextcount.ContextCount.

The parameters are packaged into dictionaries and
delivered under the variable of 'parameters' in the form 
of an iterable of all the dictionaries. 
Each dictionary contains all the parameters needed to query and process a particular syntactic context.

The parameters consist of a search string (template),
tokenizer functions for indexed results from running the
search templates, and optional arguments for formatting
or analyzing the templates. 

The following keys are expected for each parameter:

template - a TF search template (string)
target - the index of the target word
bases - tuple of basis indexes
target_tokenizer - a function to construct target tokens, requires index of target
basis_tokenizer - a function to construct basis tokens, requires index of basis
*kwargs - optional dict of keyword arguments for formatting the template (if any)
*sets - optional dict containing sets arguments for TF.search
*collapse_instances - optional boolean on whether to collapse multiple instances of a 
                      basis element at the result level; default is False
'''

parameters = list()

def token_lex(target):
    '''
    Builds simple lexeme token
    '''
    return F.lex_utf8.v(target)

def tokenize_verb(verb_node):
    '''
    Constructs a stem+lemma verb token.
    '''
    verb_stem = F.vs.v(basis)
    verb_lex = F.lex_utf8.v(basis)
    return f'{verb_stem}.{verb_lex}'




# -- clause constituent relation searches --


ccr = '''

clause kind=VC
/without/
    phrase function=PreC
/-/
    phrase function=Pred|PreO|PreS
        -heads> word pdp=verb lex#HJH[
        
    phrase typ#PP
        -heads> target
        
# target=4
# basis=2
'''

def token_cc_rela(basis, target):
    # Fnct-> stem.lex
    target_funct = F.function.v(L.u(target, 'phrase'))
    verb = tokenize_verb(basis)
    return f'{target_funct}.-> {verb}'

parameters.append({'template': ccr, 
                   'target': 4, 
                   'bases': (2,), 
                   'target_tokenizer': token_lex, 
                   'basis_tokenizer': token_cc_rela})

ccr_PP = '''

clause kind=VC
/without/
    phrase function=PreC
/-/
    phrase function=Pred|PreO|PreS
        -heads> word pdp=verb lex#HJH[
        
    phrase typ=PP
        -heads> word
        -prep_obj> target
        
# target=5
# basis=2
'''

def token_cc_rela_PP(basis, target):
    # Fnct.prep-> stem.lex
    target_funct = F.function.v(L.u(target, 'phrase'))
    prep = F.lex_utf8.v(E.heads.t(basis)[0])
    verb = tokenize_verb(basis)
    return f'{target_funct}.{prep}-> {basis_token}'

parameters.append({'template': ccr_PP, 
                   'target': 5, 
                   'bases': (2,), 
                   'target_tokenizer': token_lex, 
                   'basis_tokenizer': token_cc_rela_PP})


ccr_subj_preC = '''

clause
    phrase function=Subj
        -heads> target
    phrase function=PreC
        -heads> word

# target=2
# basis=4
'''

def token_subj_preC(basis, target):
    # PreC.lex-> Subj
    preC = token_lex(basis)
    return f'PreC.{preC}-> Subj.'

parameters.append({'template': ccr_subj_preC, 
                   'target': 2, 
                   'bases': (4,), 
                   'target_tokenizer': token_lex, 
                   'basis_tokenizer': token_subj_preC})


ccr_preC_subj = '''

clause
    phrase function=Subj
        -heads> word
    phrase function=PreC
        -heads> target

# target=4
# basis=2
'''

def token_PreC_subj(basis, target):
    # PreC-> Subj.lex
    subj = token_lex(basis)
    return f'PreC-> Subj.{subj}'

parameters.append({'template': ccr_preC_subj, 
                   'target': 4, 
                   'bases': (2,), 
                   'target_tokenizer': token_lex, 
                   'basis_tokenizer': token_PreC_subj})




# -- subphrase relation searches --


# parallel relations:
# this pattern requires additional restrictions 
# in order to ensure that no nomen-rectum terms 
# are accidentally captured

sp_parallel = '''

clause
    subphrase
        w1:target
        /without/
        subphrase rela=rec
            w1
        /-/
    <mother- subphrase rela=par
        w2:word ls#card|ordn pdp#art|prep|conj lex#KL/
        /without/
        subphrase rela=rec
            w2
        /-/
        
# target=2
# basis=4
'''

def token_sp_parallel(basis, target):
    # par.lex-> .
    par = token_lex(basis)
    return f'par.{par}-> .'

parameters.append({'template': sp_parallel, 
                   'target': 2, 
                   'bases': (4,), 
                   'target_tokenizer': token_lex, 
                   'basis_tokenizer': token_sp_parallel})