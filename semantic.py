import spacy

# Using the medium-sized spaCy model
nlp = spacy.load('en_core_web_md')
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word1.similarity(word1))

# Using the smaller spaCy model
nlp = spacy.load('en_core_web_sm')
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word1.similarity(word1))


''' My observations: 
The similarities between cat and monkey are higher than that of banana and monkey.
This is because cat and monkey both fall under the category of animals, whereas banana is less related to to animals.
Words that are conceptionally or contextually related will have higher correlation scores.
Thus, monkeys and cats being animals vs banana more likely to be categorised as a fruit.
It's great how the system learned to show words in a special way that catches how they're alike in meaning and how they fit into different situations and ideas.

Also, 
The  choice between the simpler and more complex language models depends on the specific requirements of the task at hand. 
If speed is crucial and the analysis is relatively straightforward, the simpler model may suffice. 
Alternatively, the more complex model is better for tasks that require advanced language understanding and precision.
Thus offering a detailed and nuanced analysis.
'''
