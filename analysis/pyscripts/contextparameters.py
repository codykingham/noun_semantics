'''
This module contains parameters to be used
with contextcount.ContextCounter. The class DeliverParameters
simply allows the parameters to be configured with a 
supplied list of target terms.

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

from textwrap import dedent # removes identations from triple-quote strings to align them with function
import re

def deliver_params(target_nouns, tf=None):
    '''
    A function that simply
    umbrellas all of the parameters
    and allows a target noun list to 
    be fed to the params without import
    statements. Returns a list of dictionaries
    of parameters. TF is an instance of Text-Fabric.
    '''
    F, L = tf.F, tf.L # text-fabric feature class
    parameters = list()
    sets = {'target': target_nouns}

    # <><> universal tokenizers (i.e. used throughout) <><>

    def token_lex(target):
        '''
        Builds simple lexeme token 
        with disambiguation if necessary.
        '''
        lex_trans = F.lex.v(target)
        #disambig = re.sub('[A-Z><]', '', lex_trans)
        #utf8 = F.lex_utf8.v(target)
        return lex_trans      

    def token_verb(verb_node):
        '''
        Constructs a stem+lemma verb token.
        '''
        verb_stem = F.vs.v(verb_node)
        #verb_lex = F.lex_utf8.v(verb_node)
        verb_lex = token_lex(verb_node)
        return f'{verb_stem}.{verb_lex}'


    # <><> clause constituent relation searches <><>


    # target in a non-prepositional function relating to a predicate lexeme
    clause_function = '''

    clause kind=VC
    /without/
        phrase function=PreC
    /-/
        phrase function=Pred|PreO|PreS
            -heads> word pdp=verb lex#HJH[
        phrase
        /with/
        typ#PP
        /or/
        typ=PP function=Objc
        /-/
            -noun_heads> target

    % target=4
    % bases=(2, 3)
    '''

    def token_cc_rela(bases, target):
        # funct.-> st.verb.lex
        target_funct = F.function.v(bases[1])
        verb = token_verb(bases[0])
        return f'{target_funct}.-> {verb}'

    parameters.append({'template': dedent(clause_function), 
                       'target': 4, 
                       'bases': (2,3), 
                       'target_tokenizer': token_lex, 
                       'basis_tokenizer': token_cc_rela,
                       'sets': sets,
                       'name': 'funct.-> st.verb.lex'
                      })


    # target in a prepositional function relating to a predicate lexeme
    clause_function_PP = '''

    clause kind=VC
    /without/
        phrase function=PreC
    /-/
        phrase function=Pred|PreO|PreS
            -heads> word pdp=verb lex#HJH[

        phrase typ=PP function#Objc
            -heads> word
            -prep_obj> target

    % target=5
    % bases=(2,3,4)
    '''

    def token_cc_rela_PP(bases, target):
        # funct.prep-> st.verb.lex
        verb, prep = token_verb(bases[0]), F.lex.v(bases[2])
        target_funct = F.function.v(bases[1])    
        return f'{target_funct}.{prep}-> {verb}'

    parameters.append({'template': dedent(clause_function_PP), 
                       'target': 5, 
                       'bases': (2,3,4), 
                       'target_tokenizer': token_lex, 
                       'basis_tokenizer': token_cc_rela_PP,
                       'sets': sets,
                       'name': 'funct.prep-> st.verb.lex'
                      })
    
    
    # target is subject to a complement lexeme in a predicate-complement relation 
    ccr_subj_preC = '''

    clause
        phrase function=Subj
            -heads> target
        phrase function=PreC typ#PP
            -heads> word pdp=subs|nmpr|verb|adjv|advb lex#KL/ ls#card|ordn

    % target=2
    % bases=(4,)
    '''

    def token_subj_preC(bases, target):
        # PreC.lex-> Subj.
        compliment = token_lex(bases[0])
        return f'PreC.{compliment}-> Subj.'

    parameters.append({'template': dedent(ccr_subj_preC), 
                       'target': 2, 
                       'bases': (4,), 
                       'target_tokenizer': token_lex, 
                       'basis_tokenizer': token_subj_preC,
                       'sets': sets,
                       'name': 'PreC.lex-> Subj.'
                      })
    
    
    # target is subject to a prepositional complement lexeme in a predicate-complement relation
    ccr_subj_preC_PP = '''

    clause
        phrase function=Subj
            -heads> target
        phrase function=PreC typ=PP
            -heads> word
            -prep_obj> word pdp=subs|nmpr|verb|adjv|advb lex#KL/ ls#card|ordn

    % target=2
    % bases=(4, 5)
    '''

    def token_subj_preC_PP(bases, target):
        # PreC.prep.lex-> Subj.
        prep, complement = token_lex(bases[0]), token_lex(bases[1])
        compliment = token_lex(bases[0])
        return f'PreC.{compliment}-> Subj.'

    parameters.append({'template': dedent(ccr_subj_preC_PP), 
                       'target': 2, 
                       'bases': (4,5), 
                       'target_tokenizer': token_lex, 
                       'basis_tokenizer': token_subj_preC_PP,
                       'sets': sets,
                       'name': 'PreC.prep.lex-> Subj.'
                      })

    
    # target is complement to a subject lexeme in predicate-complement relation
    ccr_preC_subj = '''

    clause
        phrase function=Subj typ#PP
            -heads> word pdp=subs|nmpr|verb|adjv|advb lex#KL/ ls#card|ordn
        phrase function=PreC
            -heads> target

    % target=4
    % bases=(2,)
    '''

    def token_PreC_subj(bases, target):
        # PreC.-> Subj.lex
        subj = token_lex(bases[0])
        return f'PreC.-> Subj.{subj}'

    parameters.append({'template': dedent(ccr_preC_subj), 
                       'target': 4, 
                       'bases': (2,), 
                       'target_tokenizer': token_lex, 
                       'basis_tokenizer': token_PreC_subj,
                       'sets': sets,
                       'name': 'PreC.-> Subj.lex'
                     })

    
    # target is complement to a prepositional subject lexeme in predicate-complement relation
    ccr_preC_subj_PP = '''

    clause
        phrase function=Subj typ=PP
            -heads> word
            -prep_obj> word pdp=subs|nmpr|verb|adjv|advb lex#KL/ ls#card|ordn
        phrase function=PreC
            -heads> target

    % target=5
    % bases=(2,3)
    '''

    def token_PreC_subj_PP(bases, target):
        # PreC.-> Subj.prep.lex
        prep, subj = token_lex(bases[0]), token_lex(bases[1])
        return f'PreC.-> Subj.{prep}.{subj}'

    parameters.append({'template': dedent(ccr_preC_subj_PP), 
                       'target': 5, 
                       'bases': (2,3), 
                       'target_tokenizer': token_lex, 
                       'basis_tokenizer': token_PreC_subj_PP,
                       'sets': sets,
                       'name': 'PreC.-> Subj.prep.lex'
                     })

    # <><> subphrase relation searches <><>


    #  a lexeme connected to target through a conjunction
    sp_parallel = '''

    clause
        p1:phrase
            subphrase
                w1:target
            <mother- subphrase rela=par
                w2:word lex#KL/ ls#card|ordn


    p1 -noun_heads> w1
    p1 -noun_heads> w2

    % target=3
    % bases=(5,)
    '''

    def token_sp_parallel(bases, target):
        # par.lex-> .
        par = token_lex(bases[0])
        return f'par.{par}-> .'

    parameters.append({'template': dedent(sp_parallel), 
                       'target': 3, 
                       'bases': (5,), 
                       'target_tokenizer': token_lex, 
                       'basis_tokenizer': token_sp_parallel,
                       'sets': sets,
                       'name': 'par.lex-> .'
                      })


    #  target connected to a lexeme through a conjunction
    sp_parallel_rela = '''

    clause
        p1:phrase
            subphrase
                w1:word lex#KL/ ls#card|ordn
            <mother- subphrase rela=par
                w2:target


    p1 -noun_heads> w1
    p1 -noun_heads> w2

    % target=5
    % bases=(3,)
    '''

    def token_sp_parallel_rela(bases, target):
        # par.-> lex
        par = token_lex(bases[0])
        return f'par.-> {par}'

    parameters.append({'template': dedent(sp_parallel_rela), 
                       'target': 5, 
                       'bases': (3,), 
                       'target_tokenizer': token_lex, 
                       'basis_tokenizer': token_sp_parallel_rela,
                       'sets': sets,
                       'name': 'par.-> lex'
                      })


    # target is described adjectivally by a lexeme 
    sp_adjective = '''

    clause
        p1:phrase
            subphrase
                w1:target
            <mother- subphrase rela=atr
            /without/
                word ls=card|ordn
            /-/
                word pdp=adjv lex#KL/

    p1 -noun_heads> w1

    % target=3
    % bases=(5,)
    '''

    def token_sp_adjective(bases, target):
        # atr.lex -> .
        adjv = token_lex(bases[0])
        return f'atr.{adjv}-> .'

    parameters.append({'template': dedent(sp_adjective), 
                       'target': 3, 
                       'bases': (5,), 
                       'target_tokenizer': token_lex, 
                       'basis_tokenizer': token_sp_adjective,
                       'sets': sets,
                       'name': 'atr.lex -> .'
                      })


    # a lexeme is construct with target
    sp_construct = '''

    clause
        p1:phrase
            subphrase
                w1:target
                <mother- subphrase rela=rec
                    word pdp=subs ls#card|ordn lex#KL/

    p1 -noun_heads> w1

    % target=3
    % bases=(5,)
    '''

    def token_sp_construct(bases, target):
        # rec.lex -> .
        constr = token_lex(bases[0])
        return f'rec.{constr}-> .'

    parameters.append({'template': dedent(sp_construct), 
                       'target': 3, 
                       'bases': (5,), 
                       'target_tokenizer': token_lex, 
                       'basis_tokenizer': token_sp_construct,
                       'sets': sets,
                       'name': 'rec.lex -> .'
                      })


    # target is construct with a lexeme
    sp_construct_rela = '''

    clause
        p1:phrase
            subphrase
                w1:word ls#card|ordn lex#KL/
                <mother- subphrase rela=rec
                    target

    p1 -noun_heads> w1

    % target=5
    % bases=(3,)
    '''

    def token_sp_construct_rela(bases, target):
        # rec.-> lex
        absolute = token_lex(bases[0])
        return f'rec.-> {absolute}'

    parameters.append({'template': dedent(sp_construct_rela), 
                       'target': 5, 
                       'bases': (3,), 
                       'target_tokenizer': token_lex, 
                       'basis_tokenizer': token_sp_construct_rela,
                       'sets': sets,
                       'name': 'rec.-> lex'
                      })


    # <><> phrase atom relation searches <><>


    # a lexeme is connected through conjunction to target (in another phrase_atom)
    pa_parallel = '''

    clause
        phrase_atom
            -noun_heads> target
        <mother- phrase_atom rela=Para
            -noun_heads> word pdp=subs lex#KL/ ls#card|ordn

    % target=2
    % bases=(4,)
    '''

    def token_pa_parallel(bases, target):
        # Para.lex-> .
        para = token_lex(bases[0])
        return f'Para.{para}-> .'

    parameters.append({'template': dedent(pa_parallel), 
                       'target': 2, 
                       'bases': (4,), 
                       'target_tokenizer': token_lex, 
                       'basis_tokenizer': token_pa_parallel,
                       'sets': sets,
                       'name': 'Para.lex-> .'
                      })


    # target is connected through conjunction to a lexeme
    pa_parallel_rela = '''

    clause
        phrase_atom
            -noun_heads> word pdp=subs lex#KL/ ls#card|ordn
        <mother- phrase_atom rela=Para
            -noun_heads> target

    % target=4
    % bases=(2,)
    '''

    def token_pa_parallel_rela(bases, target):
        # Para.-> lex
        paralleled = token_lex(bases[0])
        return f'Para.-> {paralleled}'

    parameters.append({'template': dedent(pa_parallel_rela), 
                       'target': 4, 
                       'bases': (2,), 
                       'target_tokenizer': token_lex, 
                       'basis_tokenizer': token_pa_parallel_rela,
                       'sets': sets,
                       'name': 'Para.-> lex'
                      })


    # a lexeme is in apposition to target
    pa_apposition = '''

    clause
        phrase_atom
            -noun_heads> target
        <mother- phrase_atom rela=Appo
            -noun_heads> word ls#card|ordn lex#KL/

    % target=2
    % bases=(4,)
    '''

    def token_pa_apposition(bases, target):
        # Appo.lex-> .
        appo = token_lex(bases[0])
        return f'Appo.{appo}-> .'

    parameters.append({'template': dedent(pa_apposition), 
                       'target': 2, 
                       'bases': (4,), 
                       'target_tokenizer': token_lex, 
                       'basis_tokenizer': token_pa_apposition,
                       'sets': sets,
                       'name': 'Appo.lex-> .'
                      })


    # target is in apposition to a lexeme
    pa_apposition_rela = '''

    clause
        phrase_atom
            -noun_heads> word ls#card|ordn lex#KL/
        <mother- phrase_atom rela=Appo
            -noun_heads> target

    % target=4
    % bases=(2,)
    '''

    def token_pa_apposition_rela(bases, token):
        # Appo.-> lex
        appo_rela = token_lex(bases[0])
        return f'Appo.-> {appo_rela}'

    parameters.append({'template': dedent(pa_apposition_rela), 
                       'target': 4, 
                       'bases': (2,), 
                       'target_tokenizer': token_lex, 
                       'basis_tokenizer': token_pa_apposition_rela,
                       'sets': sets,
                       'name': 'Appo.-> lex'
                      })
    
    return parameters