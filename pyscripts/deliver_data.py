# this module takes in a set of search parameters and tokenizers and delivers 
# a list of contexts for analysis

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

            # get clause for clause-token mapping
            book,chapter,verse = T.sectionFromNode(specimen[0])
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
                    'clause_text': T.text(clause),
                    'ref': f'{book} {chapter}:{verse}',
                    'target': target_token,
                    'target_node': target,
                    'basis': basis_tag,
                    'basis_nodes': bases_nodes, 
                    'verse_text': T.text(verse_node),
                }) 

    return data 
