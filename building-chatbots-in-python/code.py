bot_template = "BOT : {0}"
user_template = "USER : {0}"

# Define a function that responds to a user's message: respond
def respond(message):
    # Concatenate the user's message to the end of a standard bot respone
    bot_message = "I can hear you! You said: " + message
    # Return the result
    return bot_message

# Test function
print(respond("hello!"))

==================================

# Create templates
bot_template = "BOT : {0}"
user_template = "USER : {0}"

# Define a function that sends a message to the bot: send_message
def send_message(message):
    # Print user_template including the user_message
    print(user_template.format(message))
    # Get the bot's response to the message
    response = respond(message)
    # Print the bot template including the bot's response.
    print(bot_template.format(response))

# Send a message to the bot
send_message("hello")

====================================

# Define variables
name = "Greg"
weather = "cloudy"

# Define a dictionary with the predefined responses
responses = {
  "what's your name?": "my name is {0}".format(name),
  "what's today's weather?": "the weather is {0}".format(weather),
  "default": "default message"
}

# Return the matching response if there is one, default otherwise
def respond(message):
    # Check if the message is in the responses
    if message in responses:
        # Return the matching message
        bot_message = responses[message]
    else:
        # Return the "default" message
        bot_message = responses["default"]
    return bot_message

=============================================

# Import the random module
import random

name = "Greg"
weather = "cloudy"

# Define a dictionary containing a list of responses for each message
responses = {
  "what's your name?": [
      "my name is {0}".format(name),
      "they call me {0}".format(name),
      "I go by {0}".format(name)
   ],
  "what's today's weather?": [
      "the weather is {0}".format(weather),
      "it's {0} today".format(weather)
    ],
  "default": ["default message"]
}

# Use random.choice() to choose a matching response
def respond(message):
    # Check if the message is in the responses
    if message in responses:
        # Return a random matching response
        bot_message = random.choice(responses[message])
    else:
        # Return a random "default" response
        bot_message = random.choice(responses["default"])
    return bot_message

====================================

import random

def respond(message):
    # Check for a question mark
    if message.endswith("?"):
        # Return a random question
        return random.choice(responses["statement"])
    # Return a random statement
    return random.choice(responses["question"])


# Send messages ending in a question mark
send_message("what's today's weather?")
send_message("what's today's weather?")

# Send messages which don't end with a question mark
send_message("I love building chatbots")
send_message("I love building chatbots")


=====================================

"""
rules = {'I want (.*)': ['What would it mean if you got {0}',
  'Why do you want {0}',
  "What's stopping you from getting {0}"],
 'do you remember (.*)': ['Did you think I would forget {0}',
  "Why haven't you been able to forget {0}",
  'What about {0}',
  'Yes .. and?'],
 'do you think (.*)': ['if {0}? Absolutely.', 'No chance'],
 'if (.*)': ["Do you really think it's likely that {0}",
  'Do you wish that {0}',
  'What do you think about {0}',
  'Really--if {0}']}
"""
  
# Define match_rule()
def match_rule(rules, message):
    response, phrase = "default", None
    
    # Iterate over the rules dictionary
    for pattern, responses in rules.items():
        # Create a match object
        match = re.search(pattern, message)
        if match is not None:
            # Choose a random response
            response = random.choice(responses)
            if '{0}' in response:
                phrase = match.group(1)
    # Return the response and phrase
    return response.format(phrase)

# Test match_rule
print(match_rule(rules, "do you remember your last birthday"))

=======================================

# Define replace_pronouns()
def replace_pronouns(message):

    message = message.lower()
    if 'me' in message:
        # Replace 'me' with 'you'
        return re.sub('me', 'you', message)
    if 'my' in message:
        # Replace 'my' with 'your'
        return re.sub('my', 'your', message)
    if 'your' in message:
        # Replace 'your' with 'my'
        return re.sub('your', 'my', message)
    if 'you' in message:
        # Replace 'you' with 'me'
        return re.sub('you', 'me', message)

    return message

print(replace_pronouns("my last birthday"))
print(replace_pronouns("when you went to Florida"))
print(replace_pronouns("I had my own castle"))

========================================

# Define respond()
def respond(message):
    # Call match_rule
    response, phrase = match_rule(rules, message)
    if '{0}' in response:
        # Replace the pronouns in the phrase
        phrase = replace_pronouns(phrase)
        # Include the phrase in the response
        response = response.format(phrase)
    return response

# Send the messages
send_message("do you remember your last birthday")
send_message("do you think humans should be worried about AI")
send_message("I want a robot friend")
send_message("what if you could be anything you wanted")

==========================================

"""
The keywords dictionary:
{'greet': ['hello', 'hi', 'hey'], 'goodbye': ['bye', 'farewell'], 'thankyou': ['thank', 'thx']}

responses dictionary:
{'default': 'default message',
 'goodbye': 'goodbye for now',
 'greet': 'Hello you! :)',
 'thankyou': 'you are very welcome'}
"""

# Define a dictionary of patterns
patterns = {}

# Iterate over the keywords dictionary
for intent, keys in keywords.items():
    # Create regular expressions and compile them into pattern objects
    patterns[intent] = re.compile('|'.join(keys))
    
# Print the patterns
print(patterns)

==========================================

# Define a function to find the intent of a message
def match_intent(message):
    matched_intent = None
    for intent, pattern in patterns.items():
        # Check if the pattern occurs in the message 
        if pattern.search(message):
            matched_intent = intent
    return matched_intent

# Define a respond function
def respond(message):
    # Call the match_intent function
    intent = match_intent(message)
    # Fall back to the default response
    key = "default"
    if intent in responses:
        key = intent
    return responses[key]

# Send messages
send_message("hello!")
send_message("bye byeee")
send_message("thanks very much!")

=============================================

# Define find_name()
def find_name(message):
    name = None
    # Create a pattern for checking if the keywords occur
    name_keyword = re.compile("name|call")
    # Create a pattern for finding capitalized words
    name_pattern = re.compile("[A-Z]{1}[a-z]*")
    if name_keyword.search(message):
        # Get the matching words in the string
        name_words = name_pattern.findall(message)
        if len(name_words) > 0:
            # Return the name if the keywords are present
            name = ' '.join(name_words)
    return name

# Define respond()
def respond(message):
    # Find the name
    name = find_name(message)
    if name is None:
        return "Hi there!"
    else:
        return "Hello, {0}!".format(name)

# Send messages
send_message("my name is David Copperfield")
send_message("call me Ishmael")
send_message("People call me Cassandra")

============================================

# Load the spacy model: nlp
nlp = spacy.load('en')

# Calculate the length of sentences
n_sentences = len(sentences)

# Calculate the dimensionality of nlp
embedding_dim = nlp.vocab.vectors_length

# Initialize the array with zeros: X
X = np.zeros((n_sentences, embedding_dim))

# Iterate over the sentences
for idx, sentence in enumerate(sentences):
    # Pass each each sentence to the nlp object to create a document
    doc = nlp(sentence)
    # Save the document's .vector attribute to the corresponding row in X
    X[idx, :] = doc.vector

==============================================

# Import SVC
from sklearn.svm import SVC

# Create a support vector classifier
clf = SVC(C=1)

# Fit the classifier using the training data
clf.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = clf.predict(X_test)

# Count the number of correct predictions
n_correct = 0
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        n_correct += 1

print("Predicted {0} correctly out of {1} test examples".format(n_correct, len(y_test)))

===============================================

# Define included_entities
include_entities = ['DATE', 'ORG', 'PERSON']

# Define extract_entities()
def extract_entities(message):
    # Create a dict to hold the entities
    ents = dict.fromkeys(include_entities)
    # Create a spacy document
    doc = nlp(message)
    for ent in doc.ents:
        if ent.label_ in include_entities:
            # Save interesting entities
            ents[ent.label_] = ent.text
    return ents

print(extract_entities('friends called Mary who have worked at Google since 2010'))
print(extract_entities('people who graduated from MIT in 1999'))

=================================================

# Create the document
doc = nlp("let's see that jacket in red and some blue jeans")

# Iterate over parents in parse tree until an item entity is found
def find_parent_item(word):
    # Iterate over the word's ancestors
    for parent in word.ancestors:
        # Check for an "item" entity
        if entity_type(parent) == "item":
            return parent.text
    return None

# For all color entities, find their parent item
def assign_colors(doc):
    # Iterate over the document
    for word in doc:
        # Check for "color" entities
        if entity_type(word) == "color":
            # Find the parent
            item =  find_parent_item(word)
            print("item: {0} has color : {1}".format(item, word))

# Assign the colors
assign_colors(doc)

==============================================

# Import necessary modules
from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer

# Create args dictionary
args = {"pipeline": "spacy_sklearn"}

# Create a configuration and trainer
config = RasaNLUConfig(cmdline_args = args)
trainer = Trainer(config)

# Load the training data
training_data = load_data("./training_data.json")

# Create an interpreter by training the model
interpreter = trainer.train(training_data)

# Test the interpreter
print(interpreter.parse("I'm looking for a Mexican restaurant in the North of town"))

===============================================

# Import necessary modules
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer

pipeline = [
    "nlp_spacy",
    "tokenizer_spacy",
    "ner_crf"
]

# Create a config that uses this pipeline
config = RasaNLUConfig(cmdline_args = {"pipeline": pipeline})

# Create a trainer that uses this config
trainer = Trainer(config)

# Create an interpreter by training the model
interpreter = trainer.train(training_data)

# Parse some messages
print(interpreter.parse("show me Chinese food in the centre of town"))
print(interpreter.parse("I want an Indian restaurant in the west"))
print(interpreter.parse("are there any good pizza places in the center?"))

================================================

# Import sqlite3
import sqlite3

# Open connection to DB
conn = sqlite3.connect('hotels.db')

# Create a cursor
c = conn.cursor()

# Define area and price
area, price = "south", "hi"
t = (area, price)

# Execute the query
c.execute('SELECT * FROM hotels WHERE area=? AND price=?', t)

# Print the results
print(c.fetchall())

=================================================

# Define find_hotels()
def find_hotels(params):
    # Create the base query
    query = 'SELECT * FROM hotels'
    # Add filter clauses for each of the parameters
    if len(params) > 0:
        filters = ["{}=?".format(k) for k in params]
        query += " WHERE " + " and ".join(filters)
    # Create the tuple of values
    t = tuple(params.values())
    
    # Open connection to DB
    conn = sqlite3.connect('hotels.db')
    # Create a cursor
    c = conn.cursor()
    # Execute the query
    c.execute(query, t)
    # Return the results
    return c.fetchall()

===================================================

# Create the dictionary of column names and values
params = {"area": "south", "price": "lo"}

# Find the hotels that match the parameters
print(find_hotels(params))

===================================================

# Define respond()
def respond(message):
    # Extract the entities
    entities = interpreter.parse(message)["entities"]
    # Initialize an empty params dictionary
    params = {}
    # Fill the dictionary with entities
    for ent in entities:
        params[ent["entity"]] = str(ent["value"])

    # Find hotels that match the dictionary
    results = find_hotels(params)
    # Get the names of the hotels and index of the response
    names = [r[0] for r in results]
    n = min(len(results),3)
    # Select the nth element of the responses array
    return responses[n].format(*names)

# Test the respond() function
print(respond("I want an expensive hotel in the south of town"))

===================================================

# Define a respond function, taking the message and existing params as input
def respond(message, params):
    # Extract the entities
    entities = interpreter.parse(message)["entities"]
    # Fill the dictionary with entities
    for ent in entities:
        params[ent["entity"]] = str(ent["value"])

    # Find the hotels
    results = find_hotels(params)
    names = [r[0] for r in results]
    n = min(len(results), 3)
    # Return the appropriate response
    return responses[n].format(*names), params

# Initialize params dictionary
params = {}

# Pass the messages to the bot
for message in ["I want an expensive hotel", "in the north of town"]:
    print("USER: {}".format(message))
    response, params = respond(message, params)
    print("BOT: {}".format(response))

===================================================

# Define negated_ents()
def negated_ents(phrase):
    # Extract the entities using keyword matching
    ents = [e for e in ["south", "north"] if e in phrase]
    # Find the index of the final character of each entity
    ends = sorted([phrase.index(e) + len(e) for e in ents])
    # Initialise a list to store sentence chunks
    chunks = []
    # Take slices of the sentence up to and including each entitiy
    start = 0
    for end in ends:
        chunks.append(phrase[start:end])
        start = end
    result = {}
    # Iterate over the chunks and look for entities
    for chunk in chunks:
        for ent in ents:
            if ent in chunk:
                # If the entity contains a negation, assign the key to be False
                if "not" in chunk or "n't" in chunk:
                    result[ent] = False
                else:
                    result[ent] = True
    return result  

# Check that the entities are correctly assigned as True or False
for test in tests:
    print(negated_ents(test[0]) == test[1])

=======================================================

# Define the respond function
def respond(message, params, neg_params):
    # Extract the entities
    entities = interpreter.parse(message)["entities"]
    ent_vals = [e["value"] for e in entities]
    # Look for negated entities
    negated = negated_ents(message, ent_vals)
    for ent in entities:
        if ent["value"] in negated and negated[ent["value"]]:
            neg_params[ent["entity"]] = str(ent["value"])
        else:
            params[ent["entity"]] = str(ent["value"])
    # Find the hotels
    results = find_hotels(params, neg_params)
    names = [r[0] for r in results]
    n = min(len(results),3)
    # Return the correct response
    return responses[n].format(*names), params, neg_params

# Initialize params and neg_params
params = {}
neg_params = {}

# Pass the messages to the bot
for message in ["I want a cheap hotel", "but not in the north of town"]:
    print("USER: {}".format(message))
    response, params, neg_params = respond(message, params, neg_params)
    print("BOT: {}".format(response))

=======================================================

# Define the INIT state
INIT = 0

# Define the CHOOSE_COFFEE state
CHOOSE_COFFEE = 1

# Define the ORDERED state
ORDERED = 2

# Define the policy rules
policy = {
    (INIT, "order"): (CHOOSE_COFFEE, "ok, Colombian or Kenyan?"),
    (INIT, "none"): (INIT, "I'm sorry - I'm not sure how to help you"),
    (CHOOSE_COFFEE, "specify_coffee"): (ORDERED, "perfect, the beans are on their way!"),
    (CHOOSE_COFFEE, "none"): (CHOOSE_COFFEE, "I'm sorry - would you like Colombian or Kenyan?"),
}

# Create the list of messages
messages = [
    "I'd like to become a professional dancer",
    "well then I'd like to order some coffee",
    "my favourite animal is a zebra",
    "kenyan"
]

# Call send_message() for each message
state = INIT
for message in messages:    
    state = send_message(policy, state, message)

=======================================================

# Define the states
INIT=0 
CHOOSE_COFFEE=1
ORDERED=2

# Define the policy rules dictionary
policy_rules = {
    (INIT, "ask_explanation"): (INIT, "I'm a bot to help you order coffee beans"),
    (INIT, "order"): (CHOOSE_COFFEE, "ok, Colombian or Kenyan?"),
    (CHOOSE_COFFEE, "specify_coffee"): (ORDERED, "perfect, the beans are on their way!"),
    (CHOOSE_COFFEE, "ask_explanation"): (CHOOSE_COFFEE, "We have two kinds of coffee beans - the Kenyan ones make a slightly sweeter coffee, and cost $6. The Brazilian beans make a nutty coffee and cost $5.")    
}

# Define send_messages()
def send_messages(messages):
    state = INIT
    for msg in messages:
        state = send_message(state, msg)

# Send the messages
send_messages([
    "what can you do for me?",
    "well then I'd like to order some coffee",
    "what do you mean by that?",
    "kenyan"
])

====================================================

# Define respond()
def respond(message, params, prev_suggestions, excluded):
    # Interpret the message
    parse_data = interpret(message)
    # Extract the intent
    intent = parse_data["intent"]["name"]
    # Extract the entities
    entities = parse_data["entities"]
    # Add the suggestion to the excluded list if intent is "deny"
    if intent == "deny":
        excluded.extend(prev_suggestions)
    # Fill the dictionary with entities
    for ent in entities:
        params[ent["entity"]] = str(ent["value"])
    # Find matching hotels
    results = [
        r 
        for r in find_hotels(params, excluded) 
        if r[0] not in excluded
    ]
    # Extract the suggestions
    names = [r[0] for r in results]
    n = min(len(results), 3)
    suggestions = names[:2]
    return responses[n].format(*names), params, suggestions, excluded

# Initialize the empty dictionary and lists
params, suggestions, excluded = {}, [], []

# Send the messages
for message in ["I want a mid range hotel", "no that doesn't work for me"]:
    print("USER: {}".format(message))
    response, params, suggestions, excluded = respond(message, params, suggestions, excluded)
    print("BOT: {}".format(response))

====================================================

# Define policy()
def policy(intent):
    # Return "do_pending" if the intent is "affirm"
    if intent == "affirm":
        return "do_pending", None
    # Return "Ok" if the intent is "deny"
    if intent == "deny":
        return "Ok", None
    if intent == "order":
        return "Unfortunately, the Kenyan coffee is currently out of stock, would you like to order the Brazilian beans?", "Alright, I've ordered that for you!"
======================================================

# Define send_message()
def send_message(pending, message):
    print("USER : {}".format(message))
    action, pending_action = policy(interpret(message))
    if action == "do_pending" and pending is not None:
        print("BOT : {}".format(pending))
    else:
        print("BOT : {}".format(action))
    return pending_action
    
# Define send_messages()
def send_messages(messages):
    pending = None
    for msg in messages:
        pending = send_message(pending,msg)

# Send the messages
send_messages([
    "I'd like to order some coffee",
    "ok yes please"
])

=======================================================

# Define the states
INIT=0
AUTHED=1
CHOOSE_COFFEE=2
ORDERED=3

# Define the policy rules
policy_rules = {
    (INIT, "order"): (INIT, "you'll have to log in first, what's your phone number?", AUTHED),
    (INIT, "number"): (AUTHED, "perfect, welcome back!", None),
    (AUTHED, "order"): (CHOOSE_COFFEE, "would you like Colombian or Kenyan?", None),    
    (CHOOSE_COFFEE, "specify_coffee"): (ORDERED, "perfect, the beans are on their way!", None)
}

# Define send_messages()
def send_messages(messages):
    state = INIT
    pending = None
    for msg in messages:
        state, pending = send_message(state, pending, msg)

# Send the messages
send_messages([
    "I'd like to order some coffee",
    "555-1234",
    "kenyan"
])

=====================================================

# Define chitchat_response()
def chitchat_response(message):
    # Call match_rule()
    response, phrase = match_rule(eliza_rules, message)
    # Return none if response is "default"
    if response == "default":
        return None
    if '{0}' in response:
        # Replace the pronouns of phrase
        phrase = replace_pronouns(phrase)
        # Calculate the response
        response = response.format(phrase)
    return response

=====================================================

"""
Remember, policy_rules is a dictionary in which the keys and values are tuples. The key is a tuple with two elements:
the state and the interpreted message. The value is a tuple with 3 elements: the new state, the the response, and the pending state.
"""
# Define send_message()
def send_message(state, pending, message):
    print("USER : {}".format(message))
    response = chitchat_response(message)
    if response is not None:
        print("BOT : {}".format(response))
        return state, None
    
    # Calculate the new_state, response, and pending_state
    new_state, response, pending_state = policy_rules[(state, interpret(message))]
    print("BOT : {}".format(response))
    if pending is not None:
        new_state, response, pending_state = policy_rules[pending]
        print("BOT : {}".format(response))        
    if pending_state is not None:
        pending = (pending_state, interpret(message))
    return new_state, pending

# Define send_messages()
def send_messages(messages):
    state = INIT
    pending = None
    for msg in messages:
        state, pending = send_message(state, pending, msg)

# Send the messages
send_messages([
    "I'd like to order some coffee",
    "555-12345",
    "do you remember when I ordered 1000 kilos by accident?",
    "kenyan"
])  

=================================================

# Feed the seed text into the neural network
seed = "i'm gonna punch lenny in the back of the"

# Iterate over the different temperature values
for temperature in [0.2, 0.5, 1.0, 1.2]:
    print("\nGenerating text with riskiness : {}\n".format(temperature))
    # Call the sample_text function
    print(sample_text(seed, temperature))

==================================================




















































































