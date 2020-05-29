# this module takes in a set of search parameters and tokenizers and delivers 
# a list of contexts for analysis

import numpy as np
from .feature_formatting import book2sbl, simplified_functions

def deliver_data(parameters, tf, report=True):
    """Run queries with parameters and return results as a list"""

    # set up TF methods
    F, L, S, T = tf.api.F, tf.api.L, tf.api.S, tf.api.T

    # place all data here, to be rows in a table
    data = []

    # run the search templates and tokenize results
    for param in parameters:    
        search_template = param['template']
        target_i = param['target']
        bases_i = param['bases']
        target_tokener = param['target_tokenizer']
        basis_tokener = param['basis_tokenizer']
        tmpl_kwargs = param.get('kwargs', None)
        tmpl_sets = param.get('sets', None)
        name = param.get('name', '\n'+search_template+'\n')

        # run search query on template
        if report:
            print(f'running query on template [ {name} ]...')
        search_template = search_template if not tmpl_kwargs else search_template.format(**tmpl_kwargs)    
        search = sorted(S.search(search_template, sets=tmpl_sets))
        if report:
            print(f'\t{len(search)} results found.')

        # make target token
        for specimen in search:
        
            # get phrase function for dataset
            function = np.nan
            if 'funct' in name:
                phrases = [n for n in specimen if F.otype.v(n) == 'phrase']
                funct_phrase = phrases[-1] if phrases else 0
                function = F.function.v(funct_phrase) or np.nan
                function = simplified_functions.get(function, function)

            # get clause for clause-token mapping
            book,chapter,verse = T.sectionFromNode(specimen[0])
            book = book2sbl[book]
            clause = specimen[0] if F.otype.v(specimen[0]) == 'clause'\
                         else next(r for r in specimen if F.otype.v(r) == 'clause')
            target = specimen[target_i]
            target_token = target_tokener(target)
            verse_node = L.u(target, 'verse')[0]

            # make basis token, map to clause
            bases_nodes = tuple(specimen[i] for i in bases_i)
            basis_token = basis_tokener(bases_nodes, target)
            # so tokenizer can return multiple tokens or just one:
            basis_tokens = (basis_token,) if type(basis_token) == str else basis_token 

            # add data to the dataset
            for basis_tag in basis_tokens:
                data.append({
                    'clause': clause,
                    'book': book,
                    'ref': f'{book} {chapter}:{verse}',
                    'clause_text': T.text(clause),
                    'target': target_token,
                    'target_node': target,
                    'basis': basis_tag,
                    'basis_nodes': bases_nodes, 
                    'verse_text': T.text(verse_node),
                    'context_type': name,
                    'function': function,
                }) 

    return data 
