import math
import numpy
from nltk.translate import IBMModel1
from nltk.translate import AlignedSent

def probability_e_f(e, tr, t, epsilon=1):
    l_e = len(e)
    l_tr = len(tr)
    p_e_tr = 1
    
    for ew in e: # iterate over english words ew in english sentence e
        inner_sum = 0
        for tw in tr: # iterate over foreign words fw in foreign sentence f
            inner_sum += t[(ew, tw)]
        p_e_tr = inner_sum * p_e_tr
    
    p_e_tr = p_e_tr * epsilon / (l_tr**l_e)
    
    return p_e_tr            
# Input: Collection of sentence pairs sentence_pairs, hash of translation probabilities t, epsilon
# Output: Perplexity of model

def perplexity(sentence_pairs, t, epsilon=1, debug_output=False):
    pp = 0
    
    for sp in sentence_pairs:
        prob = probability_e_f(sp[1], sp[0], t)
        if debug_output:
            print('turkish sentence', sp[0], 'english sentence:', sp[1])
            print(prob)
            print()
        pp += math.log(prob, 2) # log base 2
        
    pp = 2.0**(-pp)
    return pp



sentence_pairs = [ 
    [ ['gelecek', 'sene'], ['future', 'year'] ], 
    [ ['geçen', 'sene'], ['last', 'year'] ], 
    [ ['geçen', 'yaz'], ['last', 'summer'] ],
    [ ['bu', 'yaz'], ['this', 'summer'] ] ,   
    [ ['bu', 'sene'], ['this', 'year'] ]
]







print('No. of sentences in translation memory: ', len(sentence_pairs))
print('Content: ', sentence_pairs)





turkish_words = []
english_words = []

for sp in sentence_pairs:
    for ew in sp[1]: 
        english_words.append(ew)
    for tw in sp[0]: 
        turkish_words.append(tw)
        
english_words = sorted(list(set(english_words)), key=lambda s: s.lower()) 
turkish_words = sorted(list(set(turkish_words)), key=lambda s: s.lower())
print('English vocab: ', english_words)
print('Turkish vocab: ', turkish_words)

english_vocab_size = len(english_words)
turkish_vocab_size = len(turkish_words)
print('english_vocab_size: ', english_vocab_size)
print('turkish_vocab_size: ', turkish_vocab_size)



def init_prob(t, init_val, english_words, turkish_words):
    for tw in turkish_words:
        for ew in english_words:
            tup = (ew, tw) # tuple required because dict key cannot be list
            t[tup] = init_val
            
            
num_iterations = 5
perplex = []
debug_output = True
s_total = {}

# Initialize probabilities uniformly
t = {}
init_val = 1.0 / turkish_vocab_size
init_prob(t, init_val, english_words, turkish_words)
if debug_output:
    print('Hash initialized')
    print('No. of foreign/english pairs: ', len(t))
    print('Content: ', t)
    print('**************')
    print()

# Loop while not converged
for iter in range(num_iterations):
    
    # Calculate perplexity
    pp = perplexity(sentence_pairs, t, 1, True)
    print(pp)
    print('**************')
    perplex.append(pp)

    # Initialize
    count = {}
    total = {}

    for tw in turkish_words:
        total[tw] = 0.0
        for ew in english_words:
            count[(ew, tw)] = 0.0

    for sp in sentence_pairs:

        # Compute normalization
        for ew in sp[1]:
            s_total[ew] = 0.0
            for tw in sp[0]:
                s_total[ew] += t[(ew, tw)]

        # Collect counts
        for ew in sp[1]:
            for tw in sp[0]:
                count[(ew, tw)] += t[(ew, tw)] / s_total[ew]
                total[tw] += t[(ew, tw)] / s_total[ew]

    # Estimate probabilities
    for tw in turkish_words:
        for ew in english_words:
            t[(ew, tw)] = count[(ew, tw)] / total[tw]

    if debug_output:
        print("--> *** t[('future','gelecek')]", t[('future','gelecek')])
        print("--> *** t[('year','gelecek')]", t[('year','gelecek')])
    
        print("--> t[('year','sene')]", t[('year','sene')])
        print("--> *** t[('last','sene')]", t[('last','sene')])
        print("--> *** t[('this','sene')]", t[('this','sene')])
        print("--> t[('future','sene')]", t[('future','sene')])
           
        print("--> t[('last','geçen')]", t[('last','geçen')])
        print("--> t[('year','geçen')]", t[('year','geçen')])
        print("--> t[('summer','geçen')]", t[('summer','geçen')])

        print("--> t[('last','yaz')]", t[('last','yaz')])
        print("--> t[('summer','yaz')]", t[('summer','yaz')])
        print("--> t[('this','yaz')]", t[('this','yaz')])
        
        print("--> t[('this','bu')]", t[('last','bu')])
        print("--> t[('summer','bu')]", t[('summer','bu')])
        print("--> t[('year','bu')]", t[('year','bu')])
#IBM MODEL RESULTS
bitext = []
bitext.append(AlignedSent(['gelecek', 'sene'], ['future', 'year']))
bitext.append(AlignedSent(['geçen', 'sene'], ['last', 'year']))
bitext.append(AlignedSent(['geçen', 'yaz'], ['last', 'summer']))
bitext.append(AlignedSent(['bu', 'yaz'], ['this', 'summer']))
bitext.append(AlignedSent(['bu', 'sene'], ['this', 'year']))
   

print("*************IBM MODEL **************")        
ibm1 = IBMModel1(bitext, 5)

print("--> *** [('future','gelecek')]", ibm1.translation_table['gelecek']['future'])
print("--> *** [('year','gelecek')]", ibm1.translation_table['gelecek']['year'])
  
print("--> *** [('sene','year')]", ibm1.translation_table['sene']['year'])
print("--> *** [('sene','last')]", ibm1.translation_table['sene']['last'])
print("--> *** [('sene','this')]", ibm1.translation_table['sene']['this'])
print("--> *** [('sene','future')]", ibm1.translation_table['sene']['future'])

print("-->*** [('geçen','last')]", ibm1.translation_table['geçen']['last'])
print("-->*** [('geçen','year')]", ibm1.translation_table['geçen']['year'])
print("-->*** [('geçen','summer')]", ibm1.translation_table['geçen']['summer'])
  
print("-->*** [('yaz','last')]",ibm1.translation_table['yaz']['last'])
print("-->*** [('yaz','summer')]", ibm1.translation_table['yaz']['summer'])
print("-->*** [('yaz','this')]", ibm1.translation_table['yaz']['this'])
  
print("-->*** [('bu','this')]", ibm1.translation_table['bu']['this'])
print("-->*** [('bu','summer')]",ibm1.translation_table['bu']['summer'])
print("-->*** [('bu','year')]", ibm1.translation_table['bu']['year'])
          
          
          
          