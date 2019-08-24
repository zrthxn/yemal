# Automated Detection of Fake News
This project is an attempt to try to build a trustworthy automated fake news classfier that doesn't just use one or two parameters to judge the truth in the given statements. The results and observations that are made will be published in a research paper.

## Methodology
The idea behind this project is to try and find patterns that indicate that a news article published on a website or some other medium, is fake. 

The methodology used is to take an input article and feed it through various mechanisms of modern automated fact-checking like Natural Language Processing, Stance Classification etc. Each stage or will produce a score-like result, grading the article on each specific parameter and build a dosier of the results or scores. <br>
It is worth mentioning that not each level will be a neural network or deep learning program.

After all the stages, a classifier will rate the article as **TRUSTED**, **FAKE** or **UNVERIFIED** using the table of scores from each of the previous stages.

## The Process
The process uptil now is as follows.

1. **Preprocessing:** Create a tokenizer that takes entire articles in text files as input and performs tokenization of the words and symbols.

2. **Natural Language Processing:** Use NLP to detect patterns and adherence to natural language in the article. This stage checks for simple patters in language and gives a score based on how natual the language is. Typically a lie will be told in a very unnatural language. Unless it is a very well hidden lie, which is why NLP by itself can't detect fake news.

3. **Stance Classification:** This is the process of checking the tone of the article and what stance it is taking. This stage gives a score on the basis of how positive or how negative the article is. Usually a fake news article will either take very positive stances, such as glorifying leaders, or take very negative stances, such as hate speech.

4. **Contextual Analysis:** This is the process of checking whether the headline of the article conincides with or is relevant to the claims made within the article. Hyped-up headlines attached to unrelated articles are often charecteristics of clickbait or strawmanning.

5. **Article Classification:** Using the score generated on each stage as features, train a RNN to classify if the article is a fake one or a trusted one. We will use a Recursive Neural Network because the order of input data to this classifier is important, i.e. it is important for the NN to know which score was in which field. This layer might also require the use of LSTM nodes but that has not been conclusively decided upon yet.

## Resources
The following papers, articles and resources were used to build this project
- 

## Datasets
The following datasets were used in this project
- Fake News Challenge https://github.com/FakeNewsChallenge/fnc-1.git