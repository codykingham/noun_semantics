#!/usr/bin/env python
# coding: utf-8

# Best viewed locally in a Jupyter notebook or online in <a href="https://nbviewer.jupyter.org/github/codykingham/noun_semantics/blob/master/analysis.ipynb">Jupyter Notebook Viewer</a>
# 
# # Analysis of Noun Semantics in the Hebrew Bible
# ## Cody Kingham
# 
# In this notebook, I compare the syntactic contexts of the top 200 most frequent nouns in the Hebrew Bible. This notebook essentially walks through my process and includes limited commentary throughout. Full descriptions borrowed from the paper will soon be transferred to here as well.

# In[1]:


get_ipython().system(' echo "last updated:"; date')


# In[3]:


# ETCBC's BHSA data
from tf.fabric import Fabric
from tf.app import use

# stats & data-containers
import collections, math, re, random, csv
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from kneed import KneeLocator # https://github.com/arvkevi/kneed

# data visualizations
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.serif'] = ['Times New Roman']

from IPython.display import HTML, display, Image
from adjustText import adjust_text # fixes overlapping scatterplot annotations

# custom modules
#from pyscripts.contextcount import ContextCounter, ContextTester
from pyscripts.contextparameters import deliver_params
from pyscripts.deliver_data import deliver_data

# prep the Hebrew syntax data

name = 'noun_semantics'
hebrew_data = ['~/github/etcbc/{}/tf/c'.format(direc) for direc in ('bhsa','lingo/heads', 'heads', 'phono')] # data dirs
load_features = '''
typ phono lex_utf8 lex
voc_lex_utf8 voc_lex gloss
freq_lex pdp sp ls
language
rela number function
vs vt
code label

head obj_prep sem_set nhead
heads noun_heads
''' 
# TF load statements
TF = Fabric(locations=hebrew_data)
api = TF.load(load_features)
B = use('bhsa', api=api, hoist=globals(), silent=True) # Bhsa functions for search and visualizing text


# In[4]:


def reverse_hb(heb_text):
    '''
    Reverses order of left-to-right text 
    for good matplotlib formatting.
    '''
    return ''.join(reversed(heb_text))

def show_word_list(word_nodes, joiner='&nbsp;&nbsp;|', title=''):
    '''
    Displays Hebrew for a pipe-separated list of word nodes
    Good for seeing lexemes without taking up screen space.
    '''
    formatted = joiner.join(T.text(node) for node in word_nodes)
    display(HTML(formatted))
    
def show_subphrases(phrase, direction=L.d):
    '''
    A simple function to print subphrases
    and their relations to each other.
    '''
    for sp in direction(phrase, 'subphrase'):
        
        mother = E.mother.f(sp)[0] if E.mother.f(sp) else ''
        mother_text = T.text(mother)
        
        print('-'*7 + str(sp) + '-'*16)
        print()
        print(f'{T.text(sp)} -{F.rela.v(sp)}-> {mother_text}')
        print(f'nodes:  {sp} -{F.rela.v(sp)}-> {mother}')
        print(f'slots:  {L.d(sp, "word")} -{F.rela.v(sp)}-> {L.d(mother or 0, "word")}')
        print('-'*30)


# ## Corpus Size
# 
# Below is the number of words included in the corpus of BHSA.

# In[5]:


len(list(F.otype.s('word')))


# ## Demonstrating the Collocational Principle
# 
# Here is a query for all nouns that serve as the object to the verb אכל "to eat". This query demonstrates how the collocation patterns of syntactic context can be informative for semantic meaning. This is the driving principle behind this project.

# In[6]:


eat_obj = '''

clause
    phrase function=Pred
        word pdp=verb lex=>KL[
    phrase function=Objc
        <head- w1:word pdp=subs
        
lex
    w2:word
    
w1 = w2
'''

eat_obj = B.search(eat_obj)
eaten_lexs = collections.Counter(T.text(r[5]) for r in eat_obj)

for word, count in eaten_lexs.most_common(10):
    print(f'{count}\t{word}')


# ## Define a Target Noun Set
# 
# *Insert discussion about the semantic relationship between iconicity and frequency with regards to the most frequent noun lexemes in the HB.*

# In[7]:


raw_search = '''

lex language=Hebrew sp=subs

'''

raw_nouns = B.search(raw_search)


# Now we order the results on the basis of lexeme frequency.

# In[8]:


raw_terms_ordered = sorted(((F.freq_lex.v(res[0]), res[0]) for res in raw_nouns), reverse=True)


# Below we have a look at the top 50 terms from the selected set. Pay attention to the feature `ls`, i.e. "lexical set." This feature gives us some rudimentary semantic information about the nouns and their usual functions, and it suggests that some additional restrictions are necessary for the noun selection procedure. Note especially that several of these nouns are used in adjectival or prepositional roles (e.g.  כל ,אחד, אין, תחת).

# In[9]:


raw_nnodes = [res[1] for res in raw_terms_ordered] # isolate the word nodes of the sample
B.displaySetup(extraFeatures={'ls', 'freq_lex'}) # config B to display ls and freq_lex

# display lexeme data
# for i, node in enumerate(raw_nnodes[:50]):
#     B.prettyTuple((node,), seq=i)


# Based on the nouns that are present, we should make some key exclusions. Many substantives have more functional or adjectival roles. Undesirable categories include copulative nouns (`nmcp`, e.g. אין), cardinal numbers (`card`), potential prepositions (`ppre`, e.g. תחת). The `ls` category of potential adverb (`padv`) contains desirable nouns like יום, but also more functionally adverbial-nouns like עוד. Thus we can see that there is a range of adverbial tendencies found in this category. Due to the potentially interesting possibility of seeing these tendencies play out in the data, we can decide to keep these instances. 
# 
# To be sure, the very phenomenon of "functional" versus "nominal" is worthy of further, quantitative investigation. The `ls` feature is an experimental and incomplete feature in the ETCBC, and this is precisely the kind of shortcoming this present work seeks to address. Nouns and adverbs likely sit along a sliding scale of adverbial tendencies, with adverbs nearly always functioning in such a role, and nouns exhibiting various statistical tendencies. But due to the scope of this investigation, we limit ourselves to mainly nominal words with a small inclusion of some adverbial-like substantives.
# 
# We can eliminate more functional nouns by restricting the possible lexical set (`ls`) values. Below we apply those restrictions to the search template. In the case of certain quantifiers such as כל there is an `ls` feature of distributive noun (`nmdi`), yet this feature is likewise applied to nouns such as אח ("brother"). So it is undesirable to exclude all of these cases. Thus we depend, instead, on an additional filter list that excludes quantifiers.
# 
# A few terms such as דרך and עבר are eliminated because the ETCBC labels it as a potential preposition. This is a speculative classification. So we define a seperate parameter in the template that saves this instance.

# In[10]:


exclude = '|'.join(('KL/', 'M<V/', 'JTR/', 'M<FR/', 'XYJ/')) # exclude quantifiers
include = '|'.join(('padv', 'nmdi'))  # ok ls features
keep = '|'.join(('DRK/', '<BR/'))

'''
Below is a TF search query for three cases:
One is a lexeme with included ls features.
The second is a lexeme with a null ls feature.
The third is lexemes we want to prevent from being excluded.
For all cases we exclude excluded lexemes.
'''

select_noun_search = f'''

lex language=Hebrew
/with/
sp=subs ls={include} lex#{exclude}
/or/
sp=subs ls# lex#{exclude}
/or/
sp=subs lex={keep}
/-/

'''

select_nouns = B.search(select_noun_search)
noun_dat_ordered = sorted(((F.freq_lex.v(res[0]), res[0]) for res in select_nouns), reverse=True)
nnodes_ordered = list(noun_dat[1] for noun_dat in noun_dat_ordered)
filtered_lexs = list(node for node in raw_nnodes if node not in nnodes_ordered)

print(f'\t{len(raw_nouns) - len(select_nouns)} results filtered out of raw noun list...')
print('\tfiltered lexemes shown below:')
show_word_list(filtered_lexs)


# ### Plot the Nouns in Order of Frequency
# 
# Now that we have obtained a filtered noun-set, we must decide a cut-off point at which to limit the present analysis. Below we plot the attested nouns and their respective frequencies.

# In[14]:


# plot data
y_freqs = [lex_data[0] for lex_data in noun_dat_ordered]
x_rank = [i+1 for i in range(0, len(y_freqs))]
title = 'Noun Frequencies in the Hebrew Bible'
xlabel = 'Noun Rank'
ylabel = 'Noun Frequency'

# first plot
plt.figure(figsize=(8, 4))
plt.plot(x_rank, y_freqs, color='black', linewidth=1)
plt.title(title + f' (ranks 1-{len(x_rank)})', size=10)
plt.xlabel(xlabel, size=10)
plt.ylabel(ylabel, size=10)
plt.plot()
plt.show()


# We zoom in closer to view ranks 1-1000...
# 
# *Consider using a subplot here with 4 different zooms*

# In[16]:


# second plot
plt.figure(figsize=(8, 4))
plt.plot(x_rank[:1000], y_freqs[:1000], color='black', linewidth=1)
plt.xlabel(xlabel, size=10)
plt.ylabel(ylabel, size=10)
plt.axvline(200, color='red', linewidth=0.8, linestyle='--')
plt.savefig('results/plots/noun_frequencies1-1000.svg', format='svg', bbox_inches='tight') # save the plot (without title)
plt.title(title + f' (ranks 1-1000)', size=10)
plt.show()


# This curve is typical of Zipf's law:
# 
# > Zipf's law states that given some corpus of natural language utterances, the frequency of any word is inversely proportional to its rank in the frequency table ([wikipedia](https://en.wikipedia.org/wiki/Zipf%27s_law))
# 
# The curve sharply "elbows" at around rank 15. Between ranks 50-100 there is still an appreciable drop-off. The curve starts to significantly flatten after 200. We thus decide an arbitrary cut-off point at rank 200, based on the fact that the curve does not show any significant leveling after this point.

# In[17]:


target_nouns = nnodes_ordered[:200]
tnoun_instances = set(word for lex in target_nouns for word in L.d(lex, 'word'))

show_word_list(target_nouns) # temporary comment out while bug is fixed
print(f'\n{len(tnoun_instances)} nouns ready for searches')


# In[18]:


nouns_text_freqs = sorted(
    ((F.voc_lex_utf8.v(L.d(noun,'word')[0]), F.freq_lex.v(noun)) 
         for noun in target_nouns), key=lambda k: k[-1], reverse=True
)


# In[19]:


', '.join(f'{noun}' for noun, freq in nouns_text_freqs)


# ## Strategy for Context Selection
# 
# See [pyscripts/contextparameters.py](pyscripts/contextparameters.py) for the full delineation of these patterns and to see how they've been selected and tokenized. 

# In[20]:


contexts = deliver_params(tnoun_instances, tf=api)


# In[21]:


data = deliver_data(contexts, tf=TF)


# Let's have a look at the first example...

# In[22]:


data[0]


# Now we put the data into a dataframe. We also export the dataframe for reference.

# In[23]:


data_df = pd.DataFrame(data)
data_df.set_index('clause', inplace=True)
data_df.to_csv('dataset.csv') # export dataset
data_df.head()


# #### Random Samples of the Data

# In[15]:


# randomized = [r for r in counts.search2result['T.const→ lex (with article separation)']]

# random.shuffle(randomized)


# In[16]:


# B.show(randomized, end=50, condenseType='phrase', withNodes=True, extraFeatures={'sem_set'})


# <hr>
# 
# ### Excursus: Checking Context Tags and Gathering Examples
# 
# In this section I will inspect the tokens that are generated and counted, as well as pull out some examples and their counts for the presentation.

# In[17]:


# patterns = {'funct.-> st.verb.lex': '\D*\.-> \D*\.\D*\[',
#             'funct.prep-> st.verb.lex': '\D*\.\D+\-> \D*\.\D*\['}

# token_examps = collections.defaultdict(list)

# for token in counts.data.index:
#     for query, pattern in patterns.items():
#         if re.match(pattern, token):
#             token_examps[query].append(token)

# for query in token_examps:
#     random.shuffle(token_examps[query])
#     examples = token_examps[query][:10]
#     targets = list()
    
#     # get example target nouns
#     for ex in examples:
#         ex_target = counts.data.loc[ex].sort_values(ascending=False).index[0]
#         targets.append(ex_target)
        
#     show_random = [f'target: {target} \t {ex}' for target, ex in zip(targets, examples)]
    
#     print('QUERY: ', query)
#     print('-'*5)
#     print('\n'.join(show_random))
#     print('-'*20, '\n')


# Now some more specific counts...

# In[18]:


counts.data['לב.n1']['T.Objc→ זכה.v1.piel'].sum()


# In[19]:


counts.data['פתח.n1']['T.Cmpl→ עמד.v1.qal'].sum()


# In[20]:


counts.data['אישׁ.n1']['T.Subj→ פקד.v1.hit'].sum()


# In[21]:


counts.data['שׁער.n1']['T.Loca→ שׁית.v1.qal'].sum()


# In[22]:


counts.data['גוי.n1']['T.ב.Adju→ אמר.v1.qal'].sum()


# In[23]:


counts.data['יד.n1']['T.מן.Cmpl→ ישׁע.v1.hif'].sum()


# In[24]:


counts.data['עת.n1']['T.ב.Time→ נתן.v1.nif'].sum()


# In[25]:


counts.data['דרך.n1']['T.ל.Cmpl→ פנה.v1.qal'].sum()


# <hr>

# #### Examining the Dataset
# 
# Below we look at the number of dimensions in the data:

# In[26]:


counts.data.shape


# And a sample of the data is below, sorted on the results of אלהים in order to bring up interesting examples.

# In[27]:


counts.data.sort_values(ascending=False, by='אלהים.n1').head(10)


# Next we look at a few example counts:

# In[28]:


pd.DataFrame(counts.data['אלהים.n1'][counts.data['אלהים.n1'] > 0].sort_values(ascending=False)).head(15)


# This gives a good idea of the content of the co-occurrence counts.

# #### Various Tag Searches Below
# 
# Below I isolate a few tags of interest to serve as examples in the paper. 
# 
# **TODO:** Extract and display all the exact examples.

# In[29]:


prec = [tag for tag in counts.data.index if 'PreC' in tag and 'אישׁ.n1' in tag]

prec


# In[30]:


target = 'עלה.n1'

target_counts = counts.data[target][counts.data[target]>0].sort_values(ascending=False)

prec_contexts = target_counts[target_counts.index.str.contains('ל.PreC')]

prec_contexts


# ## Adjusting the Counts
# 
# We will apply two primary adjustments:
# 
# 1. We drop co-occurrences that are unique to a noun. The dropped observations will thus be considered outliers. While these items are useful for describing the uniqueness of a given lexeme, they are unhelpful for drawing comparisons between our sets. 
# 2. We convert the counts into a measure of statistical significance. For this we use Fisher's exact test, which is ideal for datasets that have counts that are less than 5. Our matrix is likely to have many such counts. The resulting p-values, of which <0.05 represents a statistically significant colexeme, will be log-transformed. Values that fall below expected frequencies will be negatively transformed.

# ### Remove Co-occurrence Outliers
# 
# We will remove colexemes/bases that occur with only one target noun. This is done by subtracting the row total from each item in the row. Any 0 value in a row means that that row has a unique colexeme that only occurs with one target noun (we will call that a `hapax_colex` here). We will remove these rows further down.

# In[31]:


colex_counts = counts.data.sum(1)
remaining_counts = counts.data.sub(colex_counts, axis=0) # subtract colex_counts
hapax_colex = remaining_counts[(remaining_counts == 0).any(1)] # select rows that have a 0 value anywhere


# Below is an example just to make sure we've selected the right indices. The value has been manually chosen from `hapax_colex`.

# In[32]:


counts.data.loc['T.Adju→ אכל.v1.pual'].sort_values(ascending=False).head()


# Indeed this context tag is only attested with חרב, thus it is not useful for drawing meaningful comparisons to this noun. Below we see that there are `8191` other such basis elements. We remove these data points in the next cell and name the new dataset `data`.

# In[33]:


hapax_colex.shape


# In[34]:


data = counts.data.drop(labels=hapax_colex.index, axis=0)

print(f'New data dimensions: {data.shape}')
print(f'New total observations: {data.sum().sum()}')
print(f'Observations removed: {counts.data.sum().sum() - data.sum().sum()}')


# Random example to make sure there are no unique colexemes in the new dataset:

# In[35]:


data.loc['T.Adju→ בוא.v1.hif'].sort_values(ascending=False).head(5)


# #### Check for Orphaned Target Nouns
# 
# I want to see if any target nouns in the dataset now have 0 basis observations (i.e. "orphaned") as a result of our data pruning. The test below shows that there is no columns in the table with a sum of 0.

# In[36]:


data.loc[:, (data == 0).all(0)].shape


# ### How many zero counts are there?
# 
# The raw count matrix has a lot of sparsity. Here's how many zeros there are. We also count other values.

# In[37]:


unique_values, value_counts = np.unique(data.values, return_counts=True)
unique_counts = pd.DataFrame.from_dict(dict(zip(unique_values, value_counts)), orient='index', columns=['count'])
display(HTML('<h5>Top 10 Unique Values and Their Counts in Dataset</h5>'))
unique_counts.head(10)


# In[38]:


zero = unique_counts.loc[0.0][0]
non_zero = unique_counts[unique_counts.index > 0].sum()[0]
non_zero_ratio, zero_ratio = non_zero / (non_zero+zero), zero / (non_zero+zero)

print(f'Number of zero count variables: {zero} ({round(zero_ratio, 2)})')
print(f'Number of non-zero count variables: {non_zero} ({round(non_zero_ratio, 2)})')


# Below the number of observed counts is given:

# In[39]:


data.sum().sum()


# ### Apply Fisher's Exact Test
# 
# Now we apply the Fisher's exact test to the data set. This involves supplying values to a 2x2 contingency table that is fed to `scipy.stats.fisher_exact`

# #### Number of Datapoints To Iterate Over
# 
# The Fisher's exact test takes some time to run. That is because it must iterate over a lot of pairs. The number is printed below.

# In[40]:


print(data.shape[0]*data.shape[1])


# #### Apply the Tests
# 
# The whole run takes 5.5-6.0 minutes on a 2017 Macbook pro.

# In[41]:


# data for contingency tables
target_obs = data.apply(lambda col: col.sum(), axis=0, result_type='broadcast') # total target lexeme observations
colex_obs = data.apply(lambda col: col.sum(), axis=1, result_type='broadcast') # total colexeme/basis observations
total_obs = data.sum().sum() # total observations

# preprocess parts of contingency formula; 
# NB: a_matrix = data
b_matrix = target_obs.sub(data)
c_matrix = colex_obs.sub(data)
d_matrix = pd.DataFrame.copy(data, deep=True)
d_matrix[:] = total_obs
d_matrix = d_matrix.sub(data+b_matrix+c_matrix)

fisher_transformed = collections.defaultdict(lambda: collections.defaultdict())

i = 0 # counter for messages
indent(reset=True) # TF utility for timed messages
info('applying Fisher\'s test to dataset...')
indent(level=1, reset=True)

for lex in data.columns:
    for colex in data.index:
        a = data[lex][colex]
        b = b_matrix[lex][colex]
        c = c_matrix[lex][colex]
        d = d_matrix[lex][colex]
        contingency = np.matrix([[a, b], [c, d]])
        oddsratio, pvalue = stats.fisher_exact(contingency)
        fisher_transformed[lex][colex] = pvalue
        i += 1
        if i % 100000 == 0: # update message every 100,000 iterations
            info(f'finished iteration {i}...')
            
indent(level=0)
info(f'DONE at iteration {i}!')

fisherdata = pd.DataFrame(fisher_transformed)


# In[42]:


fisherdata.head(10)


# ### log10 transformation

# In[43]:


expectedfreqs = (data+b_matrix) * (data+c_matrix) / (data+b_matrix+c_matrix+d_matrix)
fishertransf = collections.defaultdict(lambda: collections.defaultdict())

indent(reset=True)
info('applying log10 transformation to Fisher\'s data...')

for lex in data.columns:
    for colex in data.index:
        observed_freq = data[lex][colex]
        exp_freq = expectedfreqs[lex][colex]
        pvalue = fisherdata[lex][colex]
        if observed_freq < exp_freq:
            logv = np.log10(pvalue)
            fishertransf[lex][colex] = logv
        else:
            logv = -np.log10(pvalue)
            fishertransf[lex][colex] = logv
    
info('finished transformations!')
            
fishertransf = pd.DataFrame(fishertransf)


# The Fisher's test has produced zero values, indicating a very high degree of attraction between lexemes and a colexemes. A log-transformed zero equals `infinity`. Below those values are isolated.

# In[44]:


display(HTML('<h5>contexts x nouns with a p-value of 0 :</h5>'))
inf_nouns = fishertransf.columns[(fishertransf == np.inf).any()]
inf_data = [] # inf data contains column/index information needed to assign the new values
for inf_noun in inf_nouns:
    inf_noun2context = pd.DataFrame(fishertransf[inf_noun][fishertransf[inf_noun] == np.inf])
    inf_data.append(inf_noun2context)
    display(inf_noun2context)


# In this case the Fisher's has returned a zero value. A p-value of 0 means that the likelihood אלהים and יהוה are *not* dependent variables is essentially null. We can thus reject the null hypothesis that the two values are not related. There is, rather, a maximum level of confidence that these two values *are* interrelated. The `np.inf` value that resulted from `log10(0)` is not viable for calculating vector distances. Thus, we need to substitute an arbitrary, but appropriate value. Below we access the lowest non-zero p-values in the dataset.

# In[45]:


minimum_pvalues = fisherdata.min()[fisherdata.min() > 0].sort_values()
minmin_noun = minimum_pvalues.index[0]
minmin_context = fisherdata[minimum_pvalues.index[0]].sort_values().index[0]
minimum_pvalues.head(10)


# The minimum noun x context score is shown below.

# In[46]:


minmin_noun


# In[47]:


minmin_context


# The small pvalue listed above is used to substitute the infinitive values below.

# In[48]:


# make the substitutions 
for inf_dat in inf_data:
    for noun in inf_dat.columns:
        for context in inf_dat.index:
            print(f'adjusting infinite score for {noun}')
            new_pvalue, new_transf = fisherdata[minmin_noun][minmin_context], fishertransf[minmin_noun][minmin_context]
            fisherdata[noun][context] = new_pvalue
            print(f'\tpvalue updated to {new_pvalue}')
            fishertransf[noun][context] = new_transf
            print(f'\ttransformed pvalue updated to {new_transf}')


# Below we double to check to ensure that all infinitive values have been removed. The test should read `False`.

# In[49]:


# infinites in dataset?
bool(len(fishertransf[(fishertransf == np.inf).any(1)].index))


# ### Comparing Raw and Adjusted Counts
# 
# What kinds of counts are "upvoted" and "downvoted" in the adjusted numbers? This information is helpful for gaining insight into the adjustment process and the efficacy of its results.
# 
# Below I isolate and compare counts for a set of key lexemes: מלך "king", עיר "city", and חכמה "wisdom". The counts are analyzed by comparing context tag rankings and looking for those contexts which are most affected (i.e. have the most absolute differences) by the changes. 

# In[50]:


examine_nouns = ['מלך.n1', 'עיר.n1', 'חכמה.n1']

context_rankings = {}

# gather context rankings into dataframes
for noun in examine_nouns:
    
    # make raw context DF, sorted, with columns count and rank
    rawcounts = pd.DataFrame(data[noun].values, 
                             columns=['count'], 
                             index=data.index).sort_values(ascending=False, by='count')
    rawcounts['rank'] = np.arange(len(rawcounts))+1 # add column "rank"
    
    # make adjusted context DF, sorted, with columns count and rank
    adjcounts = pd.DataFrame(fishertransf[noun].values, 
                             columns=['count'], 
                             index=fishertransf.index).sort_values(ascending=False, by='count')
    adjcounts['rank'] = np.arange(len(adjcounts))+1
    
    # put both DFs into dict mapped to noun
    context_rankings[noun]={'raw':rawcounts, 'adj':adjcounts}
    
    
# print for each noun a report on top up/downgrades
for noun, rankset in context_rankings.items():
    raw, adj = rankset['raw'], rankset['adj']
    upgrades = pd.DataFrame((raw['rank']-adj['rank']).sort_values(ascending=False))
    downgrades = pd.DataFrame((raw['rank']-adj['rank']).sort_values())
    upgrades.columns, downgrades.columns = [['difference']]*2
    upgrades['previous rank'], downgrades['previous rank'] = [raw['rank']]*2
    upgrades['new rank'], downgrades['new rank'] = [adj['rank']]*2

    display(HTML(f'<h3>{noun}</h3>'))
    print('top 10 raw counts:')
    display(raw.head(10))
    print('top 10 adjusted counts:')
    display(adj.head(10))
    print('top 10 rank upgrades')
    display(upgrades.head(10))
    print('top 10 rank downgrades')
    display(downgrades.head(10))
    print('-'*40)
    print()


# #### Export Data for מלך for Paper

# In[51]:


context_rankings['מלך.n1']['raw'].head(10).to_csv('spreadsheets/king_top10_raw.csv')
round(context_rankings['מלך.n1']['adj'].head(10), 2).to_csv('spreadsheets/king_top10_adj.csv')


# #### Extracting Specific Examples for the Paper (on מלך) to Illustrate Count Adjustments
# 
# Below the four separate parts of the contingency table are extracted for מלך "king". These were previously calculated above 

# In[52]:


data['מלך.n1']['T.Objc→ נתן.v1.qal'] # A


# In[53]:


b_matrix['מלך.n1']['T.Objc→ נתן.v1.qal'] # B


# In[54]:


c_matrix['מלך.n1']['T.Objc→ נתן.v1.qal'] # C


# In[55]:


d_matrix['מלך.n1']['T.Objc→ נתן.v1.qal'] # D


# Where do the 10 cases happen?

# In[56]:


passages = []
for res in counts.target2basis2result['מלך.n1']['T.Objc→ נתן.v1.qal']:
    passages.append('{} {}:{}'.format(*T.sectionFromNode(res[0])))
print('; '.join(passages))


# What is the result of the Fisher's test?

# In[57]:


round(fisherdata['מלך.n1']['T.Objc→ נתן.v1.qal'], 4)


# What is the value of the expected count?

# In[58]:


round(expectedfreqs['מלך.n1']['T.Objc→ נתן.v1.qal'], 2)


# In[59]:


round(fishertransf['מלך.n1']['T.Objc→ נתן.v1.qal'], 2)


# How has the rank changed?

# In[60]:


context_rankings['מלך.n1']['raw'].loc['T.Objc→ נתן.v1.qal']


# In[61]:


context_rankings['מלך.n1']['adj'].loc['T.Objc→ נתן.v1.qal']


# <hr>
# 
# #### Excursus: A Random Sample Examined
# 
# We saw that the model seems to be succeeding at isolating intuitive associations with קול. Let's look at another example at random, in this case the noun ארץ ("land"). Below are the transformed p-values for that noun.

# In[62]:


fishertransf['ארץ.n1'].sort_values(ascending=False).head(10)


# The most associated variables include cases where ארץ is an object to the verb ירשׁ, where ארץ serves as the complement from which something is brought (hifil of יצא and hifil of עלה), frequently in construct to עם "people"), the participle of ישב "inhabitant(s)"), and ממלכה, "kingdom", as well as other satisfying and expected occasions of use. These examples show that the model is working well.
# <hr>

# ## Comparing the Nouns
# 
# The nouns are now ready to be compared. I will do so in two ways.
# 
# 1. Principle Component Analysis — We have a semantic space with 4,218 dimensions. That is a lot of potential angles from which to compare the vectors. One method that is commonly used in semantic space analysis is principle component analysis or **PCA**. PCA is a dimensionality reduction method that reduce a multi-dimensional vector to the two points in an imagined space that show the most distance between the nouns. We can visualize said space by plotting the two points on an X and Y axis.
# 2. Cosine Similarity — This measure allows us to compare the vectors on the basis of their trajectories. This method is particularly well-suited for semantic spaces because it ignores differences in frequency and compares, rather, the closeness of relationship between two sets of frequencies. 

# ### PCA Analysis
# 
# We want to apply PCA in order to plot nouns in an imaginary space. The goal is to use the visualization to identify patterns and groups amongst the 199 target nouns. Nouns that are more similar should fall within the same general areas relative to the origin (0, 0). PCA seeks to identify the maximum variance amongst the vector spaces. 

# In[63]:


pca = PCA(10) # PCA with 3 principal components
noun_fit = pca.fit(fishertransf.T.values) # get coordinates
pca_nouns = noun_fit.transform(fishertransf.T.values)

plt.figure(figsize=(8, 6))
sns.barplot(x=np.arange(10)+1, y=noun_fit.explained_variance_ratio_[:10])
plt.xlabel('Principle Component', size=20)
plt.ylabel('Raio of Explained Variance', size=20)
plt.title('Ratio of Explained Variance for Principle Components 1-10 (Scree Plot)', size=20)
plt.show()


# Variance accounted for by PC1 and PC2:

# In[64]:


noun_fit.explained_variance_ratio_[0]+noun_fit.explained_variance_ratio_[1]


# The plot above, also called a scree plot, tells us that the first two principle components only account for 12% of the total variance in the dataset. Thus the PCA noun space is rather noisy. This may be explained by the fact that we are combining many different kinds of syntactic contexts into one dataset. And it may also be due to the rather spread out nature of lexical data. 
# 
# Below we extract the top 25 features which are most influential for the first two principal components.

# In[65]:


loadings = noun_fit.components_.T * np.sqrt(noun_fit.explained_variance_)
loadings = pd.DataFrame(loadings.T, index=np.arange(10)+1, columns=data.index)


# In[66]:


pc1_loadings = pd.DataFrame(loadings.loc[1].sort_values(ascending=False))
pc2_loadings = pd.DataFrame(loadings.loc[2].sort_values(ascending=False))

pc1_loadings_above0 = pc1_loadings[pc1_loadings[1] > 0.1] # isolate loadings > 0

# automatically detect elbow in graph:
elbow = KneeLocator(x=np.arange(pc1_loadings_above0.shape[0]), 
                    y=pc1_loadings_above0[1].values, 
                    curve='convex', 
                    direction='decreasing').knee

# plot it all
plt.figure(figsize=(8, 6))
plt.plot(pc1_loadings_above0.values)
plt.title('Loading Scores >0 by Rank for Principle Component 1', size=20)
plt.ylabel('Loading Score', size=20)
plt.xlabel('Rank', size=20)
plt.xticks(np.arange(pc1_loadings_above0.shape[0], step=20), size=20)
plt.yticks(size=20)
plt.axvline(elbow, color='red') # plot elbow with red line
plt.show()


# #### Top PCX Loadings and Scores (for data exploration)

# In[67]:


# pcx_loadings = pd.DataFrame(loadings.loc[4].sort_values(ascending=False)) # for experiments

# pcx_loadings.head(25)


# #### Top 25 PC1 Loadings and Scores

# In[68]:


pc1_loadings.round(2).head(25).to_csv('spreadsheets/PC1_loadings.csv')
pc1_loadings.head(25)


# #### PC1 Verb Contexts and Loadings 

# In[69]:


pc1_loadings[pc1_loadings.index.str.contains('v1')].round(2).head(15).to_csv('spreadsheets/top15_animate_verbs.csv')

top_pc1_loadings = pc1_loadings[pc1_loadings[1] >= 0.30]

pc1_loadings[pc1_loadings.index.str.contains('v1')].head(15)


# #### Looking at T.ל.Cmpl→ לקח.v1.qal
# 
# This is an interesting top verbal context. Is it related to marriage situations?

# In[70]:


take_contexts = [r for r in counts.basis2result['T.ל.Cmpl→ לקח.v1.qal']]
random.seed(213214) # shuffle random, preserve state
random.shuffle(take_contexts)
B.show(take_contexts, condenseType='clause', withNodes=True, end=5)
display(HTML(f'<h4>...{len(take_contexts)-5} other results cutoff...'))


# In[71]:


'; '.join(['{} {}:{}'.format(*T.sectionFromNode(r[0])) for r in sorted(take_contexts)])


# In[72]:


len(take_contexts)


# #### PC2 Loadings, top 25

# In[73]:


pc2_loadings.head(25)


# In[74]:


def plot_PCA(pca_nouns, 
             zoom=tuple(), 
             noun_xy_dict=False, 
             save='', 
             annotate=True, 
             title='', 
             components=(pca_nouns[:,0], pca_nouns[:,1])):
    '''
    Plots a PCA noun space.
    Function is useful for presenting various zooms on the data.
    '''
    
    x, y = components
    
    # plot coordinates
    plt.figure(figsize=(12, 10))
    plt.scatter(x, y)

    if zoom:
        xmin, xmax, ymin, ymax = zoom
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
    
    if title:
        plt.title(title, size=18)
    plt.xlabel('PC1', size=18)
    plt.ylabel('PC2', size=18)
    plt.axhline(color='red', linestyle=':')
    plt.axvline(color='red', linestyle=':')
    
    # annotate points
    if annotate:
        noun_xy = {} # for noun_dict
        noun_lexs = [f'{reverse_hb(F.voc_lex_utf8.v(counts.target2lex[n]))}' for n in fishertransf.columns]
        for i, noun in enumerate(noun_lexs):
            noun_x, noun_y = x[i], y[i]
            noun_xy[fishertransf.columns[i]] = (noun_x, noun_y)
            if zoom: # to avoid annotating outside of field of view (makes plot small)
                if any([noun_x < xmin, noun_x > xmax, noun_y < ymin, noun_y > ymax]):                
                    continue # skip noun
            plt.annotate(noun, xy=(noun_x, noun_y), size='18')
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    
    plt.show()
    
    if noun_xy_dict:
        return noun_xy

test_components = (pca_nouns[:,0], pca_nouns[:,1])
        


# #### Whole PCA Space

# In[75]:


pca_nouns_xy = plot_PCA(pca_nouns, noun_xy_dict=True, save='plots/PCA_whole.png', components=test_components)


# We can already see some interesting tendencies in the data. קול and דבר are grouped in the same quadrant. In the upper right quadrant we see בן and בת. The lower left quadrant presents a particularly interesting match: יד "hand" and אלהים "God".
# 
# We zoom in closer below to have a better look at the tendencies.

# #### Main Cluster of PCA space

# In[76]:


plot_PCA(pca_nouns, zoom=((-3, 3, -2.5, 1)), save='plots/PCA_main.png')


# ### ~Animate Nouns

# Note that nouns in the lower right quadrant tend to be people, while on the lower left there are primarily things.
# 
# The plot below shows person nouns.

# In[77]:


plot_PCA(pca_nouns, zoom=((-0.1, 5, -2.5, 0.1)), save='plots/PCA_~animates')


# Let's see what nouns to the right of the y axis have most in common. This could corroborate the intuition that the nouns on the right are personal.
# 
# First we isolate the nouns with a x-axis value > 0. Those are shown below, they are obviously personal nouns.

# In[78]:


nouns_xy = pd.DataFrame.from_dict(pca_nouns_xy, orient='index', columns=['x', 'y'])
possibly_animate = pd.DataFrame(nouns_xy[nouns_xy.x > 0])
possibly_animate['gloss'] = [F.gloss.v(counts.target2lex[targ]) for targ in possibly_animate.index]
possibly_animate = possibly_animate.reindex(['gloss', 'x', 'y'], axis=1)


# In[79]:


x_animate = pd.DataFrame(possibly_animate.drop('y', axis=1).sort_values(ascending=False, by='x'))
round(x_animate,2).to_csv('spreadsheets/animate_x.csv')
print(f'total number of ~animate nouns {x_animate.shape[0]}')
x_animate


# #### Why בגד?
# 
# Why has בגד "garment" made it into the set? We compare the top loading scores against the top scores for בגד.

# In[80]:


def cf_PC_Noun(pc_loadings, noun_counts, noun, pc_name='PC1', ascending=False):
    '''
    Compares PC loadings and noun counts.
    Returns a DF containing the top common
    counts sorted on the PC.
    '''
    top_cts = noun_counts[noun][noun_counts[noun]>0] # isolate non-zero counts
    pc_word = pc_loadings.copy() # make copy of PC loadings for modifications
    pc_word.columns = [pc_name] # rename col to PCX
    pc_word[noun] = top_cts[[i for i in top_cts.index if i in pc_word.index]] # add new column for noun
    pc_word = pc_word[pc_word[noun] > 0].sort_values(by='PC1', ascending=ascending) # remove zero counts completely, sort
    return pc_word
    
bgd_pc1 = cf_PC_Noun(pc1_loadings, fishertransf, 'בגד.n1')

bgd_pc1[bgd_pc1.PC1 >= 0.3].round(2).to_csv('spreadsheets/BGD_pc1.csv')
    
bgd_pc1[bgd_pc1.PC1 >= 0.3]


# Show passages for coord relations for paper:

# In[81]:



etcbc2sbl = {
'Genesis': 'Gen', 'Exodus': 'Exod', 'Leviticus': 'Lev', 'Numbers': 'Num',
'Deuteronomy': 'Deut', 'Joshua': 'Josh', 'Judges': 'Judg', '1_Samuel': '1 Sam', '2_Samuel': '2 Sam',
'1_Kings': '1 Kgs', '2_Kings': '2 Kgs', 'Isaiah': 'Isa', 'Jeremiah': 'Jer', 'Ezekiel': 'Ezek',
'Hosea': 'Hos', 'Joel': 'Joel', 'Amos': 'Amos', 'Obadiah': 'Obad', 'Jonah': 'Jonah', 'Micah': 'Mic',
'Nahum': 'Nah', 'Habakkuk': 'Hab', 'Zephaniah': 'Zeph', 'Haggai': 'Hag', 'Zechariah': 'Zech',
'Malachi': 'Mal', 'Psalms': 'Ps', 'Job': 'Job', 'Proverbs': 'Prov', 'Ruth': 'Ruth',
'Song_of_songs': 'Song', 'Ecclesiastes': 'Eccl', 'Lamentations': 'Lam', 'Esther': 'Esth',
'Daniel': 'Dan', 'Ezra': 'Ezra', 'Nehemiah': 'Neh', '1_Chronicles': '1 Chr', '2_Chronicles': '2 Chr'}

def formatPassages(resultslist):
    '''
    Formats biblical passages with SBL style
    for a list of results.
    '''
    book2ch2vs = collections.defaultdict(lambda: collections.defaultdict(set))
    
    for result in resultslist:
        book, chapter, verse = T.sectionFromNode(result[0])
        book = etcbc2sbl[book]                
        book2ch2vs[book][chapter].add(str(verse))
            
    # assemble in to readable passages list
    passages = []
    for book, chapters in book2ch2vs.items():
        ch_verses = []
        for chapter, verses in chapters.items():
            verses = ', '.join(f'{chapter}:{verse}' for verse in sorted(verses))
            ch_verses.append(verses)
        passage = f'{book} {", ".join(ch_verses)}'
        passages.append(passage)
            
    return '; '.join(passages)

def collectPassages(contextslist, targetnoun):
    '''
    Collects and returns neatly 
    formatted passages
    for use in the paper.
    '''
    # map the passages with dicts to avoid repeats
    results = sorted(res for context in contextslist for res in counts.target2basis2result[targetnoun][context])
    return formatPassages(results)
    

bgd_mixContexts = ['']
collectPassages(bgd_pc1.head(4).index[bgd_pc1.head(4).index.str.contains('coord')], 'בגד.n1')


# In[82]:


# B.show(counts.target2basis2result['בגד.n1']['T.coord→ אהרן.n1'], condenseType='phrase', withNodes=True)


# Now we find the context tags that are highest in the set. We pull the fourth quartile (75th percentile) of the context tags to see which ones are most shared accross these nouns.

# In[83]:


animate_context = fishertransf[possibly_animate.index].quantile(0.75, axis=1).sort_values(ascending=False)
pd.DataFrame(animate_context.head(15))


# #### PCA Space: Focus on Bordering ~Animate Nouns

# In[84]:


plot_PCA(pca_nouns, zoom=((-0.5, 0.5, -1.5, -1)), save='plots/PCA_~animate_border')


# In[85]:


nouns_xy[(nouns_xy.x < 0) & (nouns_xy.x > -0.4)].sort_values(ascending=False, by='x')


# Verbs are the greatest distinguishing factor here, with אמר, בוא,נתן, לקח and others serving a big role. מות "die" also plays a role. These are definitely contexts we could expect with animate nouns.

# ### ~Inanimate Nouns
# 
# The nouns to the left of the y axis appear to be mostly inanimate. 

# In[86]:


plot_PCA(pca_nouns, zoom=((-2, 0, -2.5, 0)), title='PCA Space: ~Inanimate Noun Cluster')


# Below we pull the tendencies for the nouns with a PC1 < 0. These nouns appear to be impersonal in nature.

# In[87]:


possibly_inanimate = pd.DataFrame(nouns_xy[(nouns_xy.x < 0) & (nouns_xy.y < 0)])
possibly_inanimate['gloss'] = [F.gloss.v(counts.target2lex[targ]) for targ in possibly_inanimate.index]
possibly_inanimate = possibly_inanimate.reindex(['gloss', 'x', 'y'], axis=1)

x_inanimate = pd.DataFrame(possibly_inanimate.drop('y', axis=1).sort_values(by='x'))
round(x_inanimate,2).head(x_animate.shape[0]).to_csv('spreadsheets/inanimate_x.csv')
print(f'Number of total ~inanimates: {x_inanimate.shape[0]}')
print(f'Top ~inanimates: ')
x_inanimate.head(x_animate.shape[0])


# ### Top Influencing ~inanimate Contexts

# In[88]:


pc1_loadings.tail(25).sort_values(by=1).round(2).to_csv('spreadsheets//PC1_loadings_negative.csv')

pc1_loadings.tail(25).sort_values(by=1)


# #### What about מלאך?
# 
# Why is מלאך rated in this list of mostly "inanimates"?

# In[89]:


pc_mlak = cf_PC_Noun(pc1_loadings, fishertransf, 'מלאך.n1', ascending=True)

pc_mlak[pc_mlak.PC1 <= -0.2].round(2).to_csv('spreadsheets/MLAK_pc1.csv')

pc_mlak.head(10)


# Note that several of the top 4 contexts are related to אלהים. We pull a few examples with אלהים out for use in the paper.

# In[90]:


collectPassages(['T.אחר.n1.Cmpl→ הלך.v1.qal'], 'אלהים.n1')


# In[91]:


collectPassages(['T.אחר.n1.Cmpl→ הלך.v1.qal'], 'מלאך.n1')


# In[92]:


collectPassages(['אחר.n2.atr→ T'], 'מלאך.n1')


# In[93]:


collectPassages(['T.appo→ אלהים.n1'], 'מלאך.n1')


# The next plot shows nouns to the left of the y-origin. Note especially the terms between y(-0.5) and y(0.0.). These are more conceptual nouns. This same trajectory extends up into the far parts of the upper left quadrant through דבר and קול.

# Here is a closer look at the larger cluster near the left side of the y-origin.

# In[94]:


plot_PCA(pca_nouns, zoom=((-0.5, -0.1, -1.5, -1)))


# Moving over one more notch:

# In[95]:


plot_PCA(pca_nouns, zoom=((-1, -0.5, -2, -0.5)))


# ### ~Perception Nouns?
# 
# The first quandrant contains a number of interesting terms that appear to be mostly abstract. These nouns appear to be related in some sense to perceptions:

# In[96]:


plot_PCA(pca_nouns, zoom=((-2, 0.05, -0.05, 1)), save='plots/PCA_~perception_nouns')


# Below are the most common contexts for these nouns.

# In[97]:


perceptions = nouns_xy[(nouns_xy.x < 0) & (nouns_xy.y > 0)]
perception_contexts = fishertransf[perceptions.index].quantile(0.75, axis=1).sort_values(ascending=False).head(15)

pd.DataFrame(perception_contexts)


# Many perceptional related contexts can be seen here, namely when the noun is a direct object to verbs such as שׁמע "hear", ידע "know", ראה "see", מצא "find", and שׁכח "forget".

# ## Experiment in Metaphor Detection
# 
# If the contexts of the animate nouns are queried against the inanimate nouns, is it possible to detect cases of metaphorical extension in the dataset?

# In[98]:


# get top 25 animate verbal contexts with Subj roles:
animate_verbal_contexts = pc1_loadings[pc1_loadings.index.str.contains('v') & pc1_loadings.index.str.contains('Subj')].head(25)
print(f'number of verbal contexts searched: {animate_verbal_contexts.shape[0]}')

metaphors = [] # metaphor data here

for i, ia_noun in enumerate(x_inanimate[1:].head(40).index): # go through top 40
    
    # skip these nouns:
    if ia_noun in {'אלהים.n1', 'מלאך.n1'}: 
        continue
    
    # find attested, common contexts
    contexts = cf_PC_Noun(animate_verbal_contexts, fishertransf, ia_noun)
    
    if contexts.shape[0]: # a match is found
        
        # gather row data with columns of [noun, context, hits, passages, example]
        for context in contexts.index:
            results = counts.target2basis2result[ia_noun][context] # get results from searches
            hits = len(results)
            passages = formatPassages(results)
            example = T.text(results[0][0])
            metaphors.append([ia_noun, context, hits, passages, example, f'({formatPassages([(results[0][0],)])})'])

metaphors = pd.DataFrame(metaphors, columns=['noun', 'context', 'hits', 'passages', 'example', ' '])
metaphors.to_csv('spreadsheets/metaphors.csv')

display(HTML('<h4>Detected Metaphors</h4>'))
metaphors


# ## Cosine Similarity Analysis

# In[99]:


distances_raw = pairwise_distances(fishertransf.T.values, metric='cosine')
distances = pd.DataFrame(distances_raw, columns=fishertransf.columns, index=fishertransf.columns)
similarities = distances.apply(lambda n: 1-n) # convert distance matrix to similarities.


# ### Export Spreadsheet Similarity Data

# In[100]:


for i, term in enumerate(data.sum().sort_values(ascending=False).index):
    these_sims = pd.DataFrame(similarities[term].sort_values(ascending=False)[1:])                    
    # export to simple data
    these_sims.to_csv(f'../easydata/similarities/{i+1}Rank_{term}.csv')


# #### Export Big Table for Paper (top 20 terms)

# In[101]:


sim_rows = []

three_nouns = []
for i, term in enumerate(data.sum().sort_values(ascending=False).head(20).index):
    
    if len(three_nouns) < 3:
        three_nouns.append(term)
    
    elif len(three_nouns) == 3 or i == 20: # 3 per row
        
        # prep header with extra columns
        header = [x for tup in zip(['']*3, three_nouns) for x in tup]
        sim_rows.append(header)
        
        noun_scores = []
        for noun in three_nouns:
            top_sims = similarities[noun].sort_values(ascending=False)[1:].head(5)
            noun_scores.append(list(zip(top_sims.index, top_sims.values)))
        
        score_rows = []
        
        for simdat1, simdat2, simdat3 in zip(*noun_scores):
            score_rows.append([simdat1[0], simdat1[1], simdat2[0], simdat2[1], simdat3[0], simdat3[1]])
    
        sim_rows.extend(score_rows)
        sim_rows.append([''])
        three_nouns = []
        
with open('spreadsheets/top20_sims.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(sim_rows)


# ### Plot Top 5 Similarity for all Terms
# 
# Below we visualize the top 5 similarity scores for all of the nouns.

# In[102]:


plt.figure(figsize=(22, 210))

for i, noun in enumerate(similarities.columns):
    
    sims = similarities[noun].sort_values(ascending=False).head(6)[1:]
    sim_values = sims.values
    lexemes = [reverse_hb(F.voc_lex_utf8.v(counts.target2lex[noun])) for noun in sims.index]
    glosses = [counts.target2gloss[noun] for noun in sims.index]
    lex_glosses = [f'{lex} \'{gloss}\'' for lex, gloss in zip(lexemes, glosses)]
    sims_new = pd.DataFrame(sim_values, index=lex_glosses)
    
    # make plots
    x = np.arange(sims.shape[0])
    plt.subplot(50,4,i+1)
    plt.plot(sims_new.values)
    plt.scatter(x=x, y=sims_new.values, color='red')
    plt.title(f'Top 5 Similar to {reverse_hb(F.voc_lex_utf8.v(counts.target2lex[noun]))}', size=30)
    plt.xticks(x, lexemes, size=26)
    plt.yticks(size=14)
    plt.ylim(0, 1)
    
    
plt.tight_layout()


# ### Distribution Principle Example for Paper: מים and יין

# In[145]:


pos_water = data['מים.n1'][data['מים.n1'] > 0] # get non-zero counts
pos_wine = data['יין.n1'][data['יין.n1'] > 0] 
winewater_common = set(pos_water.index) & set(pos_wine.index) # retrieve data using non-zero indicies in common

winewater = data.loc[winewater_common][['מים.n1', 'יין.n1']]
winewater = winewater.sort_values(by=winewater.sum().sort_values(ascending=False).index[0], ascending=False) # sort on largest sum column

winewater


# Where is "washed WITH wine"?

# In[148]:


B.show(counts.target2basis2result['יין.n1']['T.ב.Cmpl→ כבס.v1.piel'], condenseType='clause')


# In[153]:


T.text(L.u(counts.target2basis2result['יין.n1']['T.ב.Cmpl→ כבס.v1.piel'][0][0], 'verse')[0])


# But מים is much more frequent with the "washing" frame, using a different context:

# In[155]:


data['מים.n1'].sort_values(ascending=False).head(10)


# ### Build Toy Example to Illustrate Similarity Algorithms
# 
# Select three words, two of which strongly related, a third which is only sort of related. I have manually selected לחם "bread", בשׂר "flesh", and זהב "gold", along with two contexts involving object use with אכל and נתן.

# In[103]:


ex_targets = ['לחם.n1', 'בשׂר.n1', 'זהב.n1']
ex_contexts = ['T.Objc→ אכל.v1.qal', 'T.Objc→ נתן.v1.qal']
sim_example = data[ex_targets].loc[ex_contexts]
sim_example.to_csv('spreadsheets/sim_example.csv')
sim_example


# #### Make Overly Simple Distance Matrix 

# In[104]:


sim_dist = collections.defaultdict(lambda: collections.defaultdict(int))

for noun in sim_example:
    for conoun in sim_example:
        sim_dist[noun][conoun] = abs(sim_example[noun] - sim_example[conoun]).sum()
        
sim_dist = pd.DataFrame(sim_dist, columns=sim_example.columns, index=sim_example.columns)

sim_dist.to_csv('spreadsheets/sim_ex_dist.csv')

sim_dist


# #### Example Sorted Distances

# In[105]:


pd.DataFrame(sim_dist['לחם.n1'].sort_values()).to_csv('spreadsheets/sim_ex_lxm.csv')
pd.DataFrame(sim_dist['בשׂר.n1'].sort_values()).to_csv('spreadsheets/sim_ex_bfr.csv')


# #### Compare Attestations

# In[106]:


data['לחם.n1'].sum()


# In[107]:


data['בשׂר.n1'].sum()


# #### Build Cosine Similarity Example for Paper

# In[108]:


plt.figure(figsize=(8, 6))

x = sim_example.values[0]
y = sim_example.values[1]
labels = sim_example.columns

plt.scatter(x, y)
for xn, yn in zip(x, y):
    plt.plot([0, xn], [0, yn], 'ro-', color='blue')

for xn, yn, labeln in zip(x, y, labels):
    plt.annotate(reverse_hb(labeln.split('.')[0])+'.n1', (xn, yn), size=22)

#plt.xlabel(sim_example.index[0], size=22) # commented out because the Hebrew gets messed up
#plt.ylabel(sim_example.index[1], size=22)

plt.xticks(np.arange(90, step=10))
plt.yticks(np.arange(24, step=2))
plt.savefig('plots/sim_example.png', dpi=300,  bbox_inches='tight')
plt.show()


# #### Angles
# 
# Get angles to draw. Will do it manually to save time.

# In[109]:


# src: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


# In[110]:


for target in sim_example.columns:
    for cotarget in sim_example.columns:
        if target == cotarget:
            continue
        else:
            print(f'{target} -> {cotarget}')
            rad = angle(sim_example[target].values, sim_example[cotarget].values)
            print(rad * 180 / math.pi)
            print()


# Cosine Measure applied below.

# In[111]:


simex_dist = pairwise_distances(sim_example.T.values, metric='cosine')
simex_dist = pd.DataFrame(simex_dist, columns=sim_example.columns, index=sim_example.columns)
simex_dist.to_csv('spreadsheets/sim_ex_dist.csv')
simex_dist


# In[112]:


simex_sim = 1 - simex_dist
simex_sim.to_csv('spreadsheets/sim_ex_sim.csv')
simex_sim


# Export an example for the presentation, נחשׁת "copper", the result of which is improved in the semantic network experiment further below.

# In[113]:


save_figs = ['נחשׁת.n1', 'זהב.n1']

for noun in save_figs:
    sims = similarities[noun].sort_values(ascending=False).head(6)[1:]
    sim_values = sims.values
    lexemes = [reverse_hb(F.voc_lex_utf8.v(counts.target2lex[noun])) for noun in sims.index]
    glosses = [counts.target2gloss[noun] for noun in sims.index]
    lex_glosses = [f'{lex} \'{gloss}\'' for lex, gloss in zip(lexemes, glosses)]
    noun_text = F.voc_lex_utf8.v(counts.target2lex[noun])
    
    # make plots
    x = np.arange(sims.shape[0])
    plt.plot(sim_values)
    plt.scatter(x=x, y=sim_values, color='red')
    plt.title(f'Top 5 Similarities for {reverse_hb(noun_text)}', size=30)
    plt.xticks(x, lexemes, size=20)
    plt.ylabel('% Similar', size=20)
    plt.ylim(0, 1)
    plt.savefig(f'plots/similarity_{noun_text}', dpi=300, bbox_inches='tight')
    plt.show()


# Look at the commonalities between bronze and holiness, skin, and work.

# In[114]:


surprising_bronze = fishertransf[['נחשׁת.n1', 'קדשׁ.n1', 'עור.n2', 'עץ.n1']]

surprising_bronze.quantile(0.75, axis=1).sort_values(ascending=False).head(15)


# In[115]:


expected_bronze = fishertransf[['נחשׁת.n1', 'כסף.n1', 'זהב.n1']]

expected_bronze.quantile(0.75, axis=1).sort_values(ascending=False).head(15)


# ## Gephi Network Visualization
# 
# The plots above are helpful for looking at individual words and their top similar terms, but they do not really give a sense of the bigger picture, nor of the connections between terms. For instance, רחב and ארך both share a connection as each others' top most common noun. But this relationship  between the terms, and their mutual relationship to other terms, cannot be clearly seen.
# 
# We can better visualize word relationships by graphing them as nodes connected by edges in a semantic network. The similarity scores can be used to give weight to the edges. A visual plotting studio like [Gephi](www.gephi.org) can then be used to plot the nodes and edges into a visual space.
# 
# There are a couple of challenges to overcome before the nodes and edges can be exported for Gephi. The most important issue is how to decide which edges are exported? It is not ideal to plot 199x199 edges, and at any rate not all noun x noun comparisons have positive similarities (indeed many have negative values). This simplifies, at least, one aspect of the edge selection: we weed out all relationships that have negative similarity scores. Below, the number of positive similarity ratings per noun are shown.

# In[116]:


for lex in similarities.columns:
    positives = similarities[lex][similarities[lex] > 0]
    print(lex, positives.shape[0])


# The best cut-off point for edge values is the point of maximum curvature, known as the "knee" or in this case, an "elbow". `kneed` is an algorithm that automatically detects this point ([here](https://github.com/arvkevi/kneed)). The detector is demonstrated below, where the red lines are the detected elbows. These points will be used as the cut-off point for edge exports.

# In[117]:


plt.figure(figsize=(22, 210))

for i, noun in enumerate(similarities.columns):
    
    sims = similarities[noun][similarities[noun] > 0].sort_values(ascending=False)[1:]
    
    x = np.arange(sims.shape[0])
    y = sims.values
    
    # detect elbow
    elbow = KneeLocator(x, y, curve='convex', direction='decreasing').knee
    
    # make plots
    plt.subplot(50,4,i+1)
    plt.plot(x, y)
    plt.title(f'{reverse_hb(F.voc_lex_utf8.v(counts.target2lex[noun]))}', size=30)
    plt.ylim(0, 0.8)
    plt.axvline(elbow, color='red')
plt.tight_layout()


# ### Export Elbow Edges
# 
# Use `kneed` to detect the ideal edge cut-off point.

# In[118]:


node_maps = {}
nodes = []
edges = []

# make nodes
ID = 0
for noun in similarities:
    label = F.voc_lex_utf8.v(counts.target2lex[noun])
    nodes.append([ID, label])
    node_maps[noun] = ID
    ID += 1

# make edges
for noun in similarities.columns:
    positive_sims = similarities[noun][similarities[noun] > 0].sort_values(ascending=False)[1:]
    
    # detect elbow
    x = np.arange(positive_sims.shape[0])
    y = positive_sims.values
    elbow = KneeLocator(x, y, curve='convex', direction='decreasing').knee   
    edge_sims = positive_sims.head(elbow)
    for conoun in edge_sims.index:
        source = node_maps[noun]
        target = node_maps[conoun]
        weight = similarities[noun][conoun]
        edges.append([source, target, weight])

# export csv's
with open('gephidata/sim_nodes_elbows.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['ID', 'Label'])
    writer.writerows(nodes)

with open('gephidata/sim_edges_elbows.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['Source', 'Target', 'Weight'])
    writer.writerows(edges)


# # TODO: Re-run Gephi with corrected data and import new images
# 
# The results presented below will change slightly since I have found some mistakes in the context parameters. To be fixed soon. -C, 29 Dec 2018

# ## Gephi Results
# 
# Below the results of the gephi graph are displayed. Force-Atlas was used to arrange the nodes. The results show discernible neighborhoods of similar nouns.
# 
# <img src="gephidata/graphs/full_graph.png">

# ## Examining Semantic Networks
# 
# Below we look closer at the neighborhoods found in the semantic network. Specifically, we want to see what features give rise to the similarities that are registered.

# ### Group 1
# 
# <img src="gephidata/graphs/group1.png">

# In[119]:


group1_targets = ['חק.n1', 'שׁבת.n1', 'תורה.n1', 'מצוה.n1', 
                  'חקה.n1', 'עדות.n1', 'משׁפט.n1']

group1_tendencies = fishertransf[group1_targets].quantile(0.75, axis=1).sort_values(ascending=False).head(15)

pd.DataFrame(group1_tendencies)


# ### Group 2
# 
# <img src="gephidata/graphs/group2.png">

# In[120]:


group2_targets = ['מנחה.n1', 'אהל.n1', 'היכל.n1', 'עיר.n1', 
                  'בית.n1', 'חומה.n1', 'במה.n1', 'מזבח.n1']

group2_tendencies = fishertransf[group2_targets].quantile(0.75, axis=1).sort_values(ascending=False).head(15)

pd.DataFrame(group2_tendencies)


# ### Group 3
# 
# <img src="gephidata/graphs/group3.png">

# In[121]:


group3_targets = ['בת.n1', 'אח.n1', 'בן.n1']

group3_tendencies = fishertransf[group3_targets].quantile(0.75, axis=1).sort_values(ascending=False).head(15)

pd.DataFrame(group3_tendencies)


# <hr>
