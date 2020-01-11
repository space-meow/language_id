# Imports
import re
import copy
import random
import statistics
import pandas as pd
from collections import Counter
from itertools import combinations
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from scipy.spatial.distance import jensenshannon


# Helpful regexes
num_re = re.compile(r"\d")
punct_re = re.compile(r"[\[\]\(\)!@#\$%\^&–\*_\+\|=\{\};:\"<>,\.\?„“]")
multi_space_re = re.compile(r"\s\s+")
lead_trail_space_re = re.compile(r"^\s|\s$")
lead_trail_hypen_re = re.compile(r"^\-|\-$")

# Global variable to define ngram window size
ngram_window = 3

# All vowels found in the data
vowels = set(["e", "æ", "ę", "ô", "å", "ĭ", "ē", "i", "ï", "ì", "ù", "î", "â", "ī",
              "ɨ", "è", "a", "ê", "ü", "ú", "à", "o", "ö", "ë", "ó", "é", "ā",
              "ò", "á", "ą", "õ", "ä"])

# Functions
def pull_down_tokens(urls):
    '''
    Take a list of URLs and return a tokenized list of tokens found within
    the paragraphs tags in the HTML source of the pages.
    '''
    
    token_freqs = {}
    for url in urls:
        req = Request(url)
        html_page = urlopen(req).read()
        soup = BeautifulSoup(html_page, "html.parser")
        
        paragraphs = soup.select("p")
        for para in paragraphs:
            lines = para.text.split("\n")
            for line in lines:
                tokens = line.split(" ")
                for token in tokens:
                    if len(re.findall(num_re, token)) > 0:
                        continue
                    else:
                        token = re.sub(lead_trail_space_re, "",
                                       re.sub(lead_trail_hypen_re, "",
                                              re.sub(multi_space_re, " ",
                                                     re.sub(punct_re, "", token)))).lower()
                        if token:
                            if token not in token_freqs:
                                token_freqs[token] = 1
                            else:
                                token_freqs[token] += 1
    return token_freqs

def train_test_tokens(tokens, token_max):
    '''
    Using a maximum number of tokens calculated by taking the minimum number
    of tokens found across all token sets, split tokens for a language into
    a roughly 80/20 train/test split.
    '''
    random.seed(25)
    randomized_tokens = random.choices(list(tokens.keys()), k=token_max)
    train = random.choices(randomized_tokens, k=int(len(randomized_tokens)*0.8))
    test = list(set(randomized_tokens) - set(train))
    return random.choices(train, k=int(len(train))), random.choices(test, k=int(len(test)))

def build_sequence(token, ngram_window=3):
    '''
    Given a token and a target ngram window size, split the token into a series
    of ngrams specified by the size.
    '''
    seqs = []
    for i in range(len(token)):
        if i == 0:
            left = "START"
            if ngram_window == 5:
                left_left = "START_START"
        else:
            left = token[i-1]
            if ngram_window == 5:
                if i == 1:
                    left_left = "START"
                else:
                    left_left = token[i-2]
        
        if i == len(token) - 1:
            right = "END"
            if ngram_window == 5:
                right_right = "END_END"
        else:
            right = token[i+1]
            if ngram_window == 5:
                if i == len(token) - 2:
                    right_right = "END"
                else:
                    right_right = token[i+2]
        if ngram_window == 5:
            seqs.append(left_left +  left +  token[i] +  right +  right_right)
        else:
            seqs.append(left +  token[i] +  right)
    
    
    # Comment this block out and comment the return statement below in to
    # return ngram sequences containing all vowels
    new_seqs = []
    for seq in seqs:
        num_vowels = 0
        for c in seq:
            if c in vowels:
                num_vowels += 1
        if float(num_vowels) / float(len(seq.split(":"))) <= 0.5:
            new_seqs.append(seq)
    if len(new_seqs) == 0:
        return seqs
    else:
        return new_seqs
    
    # Comment the below in and the block above out to return ngram sequences
    # containing all vowels
    #return seqs

def multiply_list(list):
    '''
    Multiply all numbers in a list and return the product.
    '''
    i = None
    for num in list:
        if not i:
            i = num
        else:
            i *= num
    return i

def test_lang_pair(train_df, lang0, lang1, lang0_test, lang1_test, sequence_encoder):
    '''
    Train and test a one-against-one SVM model for a given language pair.
    '''
    
    # isolate training data for the two chosen languages
    train_data = train_df.loc[(train_df.lang == lang0) | (train_df.lang == lang1)]
    
    # Train an SVM model capable to returning probabilities (confidence scores)
    svm_model = SVC(C=1, probability=True)
    svm_model.fit(train_data.drop("lang", axis=1), train_data.lang)
    
    # Variables to gather some info
    bad_predictors_set = set()
    bad_index_list = []
    total_rows = 0
    results = {lang0: [], lang1: []}
    
    # Iterate over the first language's tokens
    for token in lang0_test:
        
        # Count vowels and consonants
        num_vowels = 0
        num_cons = 0
        for c in token:
            if c in vowels:
                num_vowels += 1
            else:
                num_cons += 1
        
        # Create test input rows for each ngram sequence
        test_input = {"sequence": [], "num_cons": [], "num_vowels": [], "token_len": []}
        for seq in build_sequence(token, ngram_window):
            test_input["sequence"].append(seq)
            test_input["num_cons"].append(num_cons)
            test_input["num_vowels"].append(num_vowels)
            test_input["token_len"].append(len(token))
        test_input = pd.DataFrame(test_input)
        try:
            test_input["sequence"] = sequence_encoder.transform(test_input["sequence"])
        except ValueError:
            continue
        
        # Isolate and remove low confidence rows
        bad_predictors = []
        proba_predict = svm_model.predict_proba(test_input)
        for i in range(len(proba_predict)):
            total_rows += 1
            
            # Define "low confidence" as between 0.4 and 0.6 exclusive
            if proba_predict[i][0] > 0.4 and proba_predict[i][0] < 0.6:
                bad_predictors.append(i)
                
        bad_predictors_set |= set(test_input.loc[bad_predictors,].sequence)
        bad_index_list += bad_predictors
        test_input = test_input.drop(test_input.index[bad_predictors])
        
        # Run prediction without low confidence rows, take the language with
        # the most representation
        if len(test_input) != 0:
            for lang, count in sorted(Counter(svm_model.predict(test_input)).items(), key=lambda k: k[1], reverse=True):
                results[lang0].append(lang)
                break
        
        # Comment the below in and the above out to get all rows regardless of
        # confidence
        #for lang, count in sorted(Counter(svm_model.predict(test_input)).items(), key=lambda k: k[1], reverse=True):
        #    results[lang0].append(lang)
        #    break
    
    # Repeat the same for the second language
    for token in lang1_test:
        num_vowels = 0
        num_cons = 0
        for c in token:
            if c in vowels:
                num_vowels += 1
            else:
                num_cons += 1
        test_input = {"sequence": [], "num_cons": [], "num_vowels": [], "token_len": []}
        for seq in build_sequence(token, ngram_window):
            test_input["sequence"].append(seq)
            test_input["num_cons"].append(num_cons)
            test_input["num_vowels"].append(num_vowels)
            test_input["token_len"].append(len(token))
        test_input = pd.DataFrame(test_input)
        try:
            test_input["sequence"] = sequence_encoder.transform(test_input["sequence"])
        except ValueError:
            continue
        bad_predictors = []
        proba_predict = svm_model.predict_proba(test_input)
        for i in range(len(proba_predict)):
            total_rows += 1
            if proba_predict[i][0] > 0.4 and proba_predict[i][0] < 0.6:
                bad_predictors.append(i)
        bad_predictors_set |= set(test_input.loc[bad_predictors,].sequence)
        bad_index_list += bad_predictors
        test_input = test_input.drop(test_input.index[bad_predictors])
        if len(test_input) != 0:
            for lang, count in sorted(Counter(svm_model.predict(test_input)).items(), key=lambda k: k[1], reverse=True):
                results[lang1].append(lang)
                break
        
        #for lang, count in sorted(Counter(svm_model.predict(test_input)).items(), key=lambda k: k[1], reverse=True):
        #    results[lang1].append(lang)
        #    break
        
    # Return a dictionary containing the model as well as a set of information
    # to use later for evaluation
    return {"model": svm_model,
            "results": results,
            "bad_predictors": {"set": bad_predictors_set,
                               "list": bad_index_list},
                               "total_rows": total_rows}
    

# Pull down tokesn for the subject languages
german_tokens = pull_down_tokens(["https://de.wikipedia.org/wiki/Deutschland"])
italian_tokens = pull_down_tokens(["https://it.wikipedia.org/wiki/Italia"])
polish_tokens = pull_down_tokens(["https://pl.wikipedia.org/wiki/Polska"])
welsh_tokens = pull_down_tokens(["https://cy.wikipedia.org/wiki/Cymru",
                                 "https://cy.wikipedia.org/wiki/Diwylliant_Cymru",
                                 "https://cy.wikipedia.org/wiki/Cymry",
                                 "https://cy.wikipedia.org/wiki/Ieithoedd_Celtaidd",
                                 "https://cy.wikipedia.org/wiki/Y_Wladfa",
                                 "https://cy.wikipedia.org/wiki/Cymraeg_Canol",
                                 "https://cy.wikipedia.org/wiki/Hen_Gymraeg"])
finnish_tokens = pull_down_tokens(["https://fi.wikipedia.org/wiki/Suomi"])

# Get the maximum number of usable tokens from the minimum number of tokens
# found across the subject languages
token_max = int(min([len(german_tokens), len(italian_tokens), len(polish_tokens), len(welsh_tokens), len(finnish_tokens)]) * 0.5)

# Split tokens into train and test sets
german_train, german_test = train_test_tokens(german_tokens, token_max)
italian_train, italian_test = train_test_tokens(italian_tokens, token_max)
polish_train, polish_test = train_test_tokens(polish_tokens, token_max)
welsh_train, welsh_test = train_test_tokens(welsh_tokens, token_max)
finnish_train, finnish_test = train_test_tokens(finnish_tokens, token_max)

# Calculated JSD
# Gather up all ngram sequences for all languages along with counts
total_ngrams = {"german": 0, "italian": 0, "polish": 0, "welsh": 0, "finnish": 0}
lang_sequences = {}
for lang, tokens in {"german": german_tokens, "italian": italian_tokens, "polish": polish_tokens, "welsh": welsh_tokens, "finnish": finnish_tokens}.items():
    lang_sequences[lang] = []
    for token in tokens:
        lang_sequences[lang] += build_sequence(token, ngram_window)
        
# Isolate the counts of each ngram sequence per language
for lang in total_ngrams.keys():
    total_ngrams[lang] = sum([count for count in Counter(lang_sequences[lang]).values()])
               
# Calculate probabilities for each sequence per language
lang_sequences_prob = {}
for lang, seqs in lang_sequences.items():
    lang_sequences_prob[lang] = {}
    for seq, count in Counter(seqs).items():
        lang_sequences_prob[lang][seq] = float(count) / float(total_ngrams[lang])
        
# Calculate and print pairwise JSD for all languages
for lang_pair in combinations(lang_sequences_prob.keys(), 2):
    l0, l1 = lang_pair
    l0_seqs = (lang_sequences_prob[l0].keys())
    l1_seqs = set(lang_sequences_prob[l1].keys())
    for seq in l1_seqs - l0_seqs:
        lang_sequences_prob[l0][seq] = 0
    for seq in l0_seqs - l1_seqs:
        lang_sequences_prob[l1][seq] = 0
        
    l0_prob_list = [prob for seq, prob in sorted(lang_sequences_prob[l0].items())]
    l1_prob_list = [prob for seq, prob in sorted(lang_sequences_prob[l1].items())]
    
    print("{}\t{}\t{}".format(l0, l1, jensenshannon(l0_prob_list, l1_prob_list)))
    
# Get character counts per language
lang_tok_data = {"german": {}, "italian": {}, "polish": {}, "welsh": {}, "finnish": {}}
for lang, tokens in {"german": german_tokens, "italian": italian_tokens, "polish": polish_tokens, "welsh": welsh_tokens, "finnish": finnish_tokens}.items():
    num_cons = []
    num_vowels = []
    total_lens = []
    for token in tokens:
        vow = 0
        con = 0
        total_lens.append(len(token))
        for c in token:
            if c in vowels:
                vow += 1
            else:
                con += 1
        num_cons.append(con)
        num_vowels.append(vow)
    lang_tok_data[lang] = {"num_cons": statistics.mean(num_cons),
                           "num_vowels": statistics.mean(num_vowels),
                           "total_lens": statistics.mean(total_lens)}

# Create a full training data set containing all languages and encode
# ngram sequences
vocab_freq_map = {"german": [{token: german_tokens[token] for token in german_train}],
                  "italian": [{token: italian_tokens[token] for token in italian_train}],
                  "polish": [{token: polish_tokens[token] for token in polish_train}],
                  "welsh": [{token: welsh_tokens[token] for token in welsh_train}],
                  "finnish": [{token: finnish_tokens[token] for token in finnish_train}]}

train_df = {"lang": [], "sequence": [], "num_cons": [], "num_vowels": [], "token_len": []}
for lang in vocab_freq_map.keys():
    for token in vocab_freq_map[lang][0]:
        num_vowels = 0
        num_cons = 0
        for c in token:
            if c in vowels:
                num_vowels += 1
            else:
                num_cons += 1
        seqs = build_sequence(token, ngram_window)
        for seq in seqs:
            train_df["lang"].append(lang)
            train_df["sequence"].append(seq)
            train_df["num_cons"].append(num_cons)
            train_df["num_vowels"].append(num_vowels)
            train_df["token_len"].append(len(token))
train_df = pd.DataFrame(train_df)
    
sequence_encoder = LabelEncoder()

all_seqs = set()
for token_set in [german_tokens, italian_tokens, polish_tokens, welsh_tokens, finnish_tokens]:
    for token in token_set:
        all_seqs |= set(build_sequence(token, ngram_window))
sequence_encoder = sequence_encoder.fit(list(all_seqs))
train_df["sequence"] = sequence_encoder.transform(train_df["sequence"])

# Stick testing data somewhere convenient
testing_data = {"german": german_test,
                "italian": italian_test,
                "polish": polish_test,
                "welsh": welsh_test,
                "finnish": finnish_test}

# Dictionary to save model results
model_objects = {}

# Run one-against-one model evaluation
for lang_combo in combinations(testing_data.keys(), 2):
    lang0, lang1 = lang_combo
    print("\n\nTesting model for {} vs {}...".format(lang0, lang1))
    result_object = test_lang_pair(train_df, lang0, lang1, testing_data[lang0], testing_data[lang1], sequence_encoder)
    model_objects["_".join(lang_combo)] = copy.deepcopy(result_object)
    for lang, results in result_object["results"].items():
        print("{}\t{}".format(lang, Counter(results)))

# Print out the number of total low confidence predictors as well as the
# total number of predictors if chosen to include them in test_lang_pair()
for lang_pair, data in model_objects.items():
    print("{}\t{}\t{}".format(lang_pair, len(data["bad_predictors"]["list"]), data["total_rows"]))
    
# Test the full set of models against all test sets
correct = 0
total = 0

# Iterate over languages and their test sets
for ref_lang, test_data in testing_data.items():
    
    # Iterate over tokens and gather ngrams in a way similar to test_lang_pair()
    for token in test_data:
        num_vowels = 0
        num_cons = 0
        for c in token:
            if c in vowels:
                num_vowels += 1
            else:
                num_cons += 1
        test_input = {"sequence": [], "num_cons": [], "num_vowels": [], "token_len": []}
        for seq in build_sequence(token, ngram_window):
            test_input["sequence"].append(seq)
            test_input["num_cons"].append(num_cons)
            test_input["num_vowels"].append(num_vowels)
            test_input["token_len"].append(len(token))
        test_input = pd.DataFrame(test_input)
        try:
            test_input["sequence"] = sequence_encoder.transform(test_input["sequence"])
        except ValueError:
            continue
        
        # Gather results for a language's test set
        results = {}
        
        # Iterate over models trained above
        for lang_combo, model_object in model_objects.items():
            model = model_object["model"]
            
            # Prepare to gather high confidence scores, drop low confidence
            # predictors; can be modifed to get all possible predictors rather
            # than dropping low predictors
            high_confs = [[],[]]
            bad_predictors = []
            proba_predict = model.predict_proba(test_input)
            for i in range(len(proba_predict)):
                if proba_predict[i][0] > 0.35 and proba_predict[i][0] < 0.65:
                    bad_predictors.append(i)
                else:
                    high_confs[0].append(proba_predict[i][0])
                    high_confs[1].append(proba_predict[i][1])
            pruned_test_input = test_input.drop(test_input.index[bad_predictors])
            if len(pruned_test_input) != 0:
                for lang, count in sorted(Counter(model.predict(pruned_test_input)).items(), key=lambda k: k[1], reverse=True):
                    
                    # Swap out comments for confidence product or
                    # mean confidence
                    if lang == model.classes_[0]:
                        mean_highest_conf = multiply_list(high_confs[0])
                        #mean_highest_conf = statistics.mean(high_confs[0])
                    elif lang == model.classes_[1]:
                        mean_highest_conf = multiply_list(high_confs[1])
                        #mean_highest_conf = statistics.mean(high_confs[1])
                    else:
                        raise ValueError
                    
                    # "mean_highest_conf" can be swapped out with the line
                    # below it to get the highest multiplied confidence instead
                    results[lang_combo] = {"top_lang": lang,
                                           "mean_highest_conf": mean_highest_conf,
                                           #"mean_highest_conf": max(multiply_list(high_confs[0]), multiply_list(high_confs[1])),
                                           "num_high_confs": len(high_confs[0])}
                    break
        
        # Iterate over the model results and find the language with the highest
        # confidence (score or product) to assign as the token's result
        top_lang = None
        top_conf = -1
        top_num_high_confs = -1
        for lang_combo, result in results.items():
            if result["num_high_confs"] > top_num_high_confs:
                if result["mean_highest_conf"] > top_conf:
                    top_conf = result["mean_highest_conf"]
                    top_lang = result["top_lang"]
                    top_num_high_confs = result["num_high_confs"]
        if ref_lang == top_lang:
            correct += 1
        total += 1

# Print the total correctly predicted tokens and the total tokens
print("{}\t{}".format(correct, total))



#############################################################
################## Multinomial Naive Bayes ##################
#############################################################

# Get all possible ngram sequences
all_seqs = set()
for token_set in [german_tokens, italian_tokens, polish_tokens, welsh_tokens, finnish_tokens]:
    for token in token_set:
        all_seqs |= set(build_sequence(token, ngram_window))

# Fill out training data frame template
training_df = {seq: [] for seq in all_seqs}
training_df["lang"] = []

# Fill the data frame
max_tokens = min(len(german_tokens), len(italian_tokens), len(polish_tokens), len(welsh_tokens), len(finnish_tokens))
for lang, tokens in {"german": german_tokens, "italian": italian_tokens,
                     "finnish": finnish_tokens, "polish": polish_tokens,
                     "welsh": welsh_tokens}.items():
    i = 0
    random.seed(1)
    random_toks = random.choices(list(tokens), k=max_tokens)
    for token in random_toks:
        seqs = Counter(build_sequence(token, ngram_window))
        for seq in all_seqs:
            if seq in seqs:
                training_df[seq].append(seqs[seq])
            else:
                training_df[seq].append(0)
        training_df["lang"].append(lang)
training_df = pd.DataFrame(training_df)

# Split X and y
X = training_df.drop("lang", axis=1)
y = training_df.lang

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train and score basic model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_model.score(X_test, y_test)

# Get predictions on X_test
y_pred = nb_model.predict(X_test)

# Compare against y_test
confusion_matrix(y_test, y_pred, labels=["german", "italian", "polish", "finnish", "welsh"])
print(classification_report(y_test, y_pred, labels=["german", "italian", "polish", "finnish", "welsh"]))


#-----------------------------#
## NB model with Dutch added ##
#-----------------------------#
dutch_tokens = pull_down_tokens(["https://nl.wikipedia.org/wiki/Nederland",
                                 "https://nl.wikipedia.org/wiki/Koninkrijk_der_Nederlanden"])

all_seqs = set()
for token_set in [german_tokens, italian_tokens, polish_tokens, welsh_tokens, finnish_tokens, dutch_tokens]:
    for token in token_set:
        all_seqs |= set(build_sequence(token, ngram_window))


training_df = {seq: [] for seq in all_seqs}
training_df["lang"] = []

for lang, tokens in {"german": german_tokens, "italian": italian_tokens,
                     "finnish": finnish_tokens, "polish": polish_tokens,
                     "welsh": welsh_tokens, "dutch": dutch_tokens}.items():
    i = 0
    random.seed(1)
    random_toks = random.choices(list(tokens), k=max_tokens)
    for token in random_toks:
        seqs = Counter(build_sequence(token, ngram_window))
        for seq in all_seqs:
            if seq in seqs:
                training_df[seq].append(seqs[seq])
            else:
                training_df[seq].append(0)
        training_df["lang"].append(lang)
training_df = pd.DataFrame(training_df)

X = training_df.drop("lang", axis=1)
y = training_df.lang

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_model.score(X_test, y_test)

y_pred = nb_model.predict(X_test)
confusion_matrix(y_test, y_pred, labels=["german", "italian", "polish", "welsh", "finnish", "dutch"])
print(classification_report(y_test, y_pred, labels=["german", "italian", "polish", "welsh", "finnish", "dutch"]))


#-------------------------------#
## NB model with English added ##
#-------------------------------#
english_tokens = pull_down_tokens(["https://en.wikipedia.org/wiki/England",
                                   "https://en.wikipedia.org/wiki/English_language"])

all_seqs = set()
for token_set in [german_tokens, italian_tokens, polish_tokens, welsh_tokens, finnish_tokens, english_tokens]:
    for token in token_set:
        all_seqs |= set(build_sequence(token, ngram_window))

training_df = {seq: [] for seq in all_seqs}
training_df["lang"] = []

for lang, tokens in {"german": german_tokens, "italian": italian_tokens,
                     "finnish": finnish_tokens, "polish": polish_tokens,
                     "welsh": welsh_tokens, "english": english_tokens}.items():
    i = 0
    random.seed(1)
    random_toks = random.choices(list(tokens), k=max_tokens)
    for token in random_toks:
        seqs = Counter(build_sequence(token, ngram_window))
        for seq in all_seqs:
            if seq in seqs:
                training_df[seq].append(seqs[seq])
            else:
                training_df[seq].append(0)
        training_df["lang"].append(lang)
training_df = pd.DataFrame(training_df)

X = training_df.drop("lang", axis=1)
y = training_df.lang

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_model.score(X_test, y_test)

y_pred = nb_model.predict(X_test)
confusion_matrix(y_test, y_pred, labels=["german", "italian", "polish", "welsh", "finnish", "english"])
print(classification_report(y_test, y_pred, labels=["german", "italian", "polish", "welsh", "finnish", "english"]))
