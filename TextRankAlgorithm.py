import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt') # one time execution
nltk.download('stopwords')# one time execution
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
stop_words = stopwords.words('english')
import re
from operator import itemgetter

class SummaryTool(object):

    def split_content_to_paragraphs(self, content):
        return content.split("\n\n")

    def formatSentences(self, content):

        # split the the text in the articles into sentences
        sentences = sent_tokenize(content)

        # flatten the list
        #sentences = [y for x in sentences for y in x]

        # remove some punctuations and special characters
        clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z0-9',.€$£]", " ")

        # make alphabets lowercas"
        #clean_sentences = [s.lower() for s in clean_sentences]

        # function to remove stopwords
        def remove_stopwords(sen):
            sen_new = " ".join([i for i in sen if i not in stop_words])
            return sen_new

        # remove stopwords from the sentences
        #clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

        return clean_sentences

    def sentence_similarity(self, sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1

        return 1 - cosine_distance(vector1, vector2)

    def build_similarity_matrix(self, sentences, stopwords=None):
        # Create an empty similarity matrix
        S = np.zeros((len(sentences), len(sentences)))


        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:
                    continue

                S[idx1][idx2] = self.sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

        # normalize the matrix row-wise
        for idx in range(len(S)):
            if S[idx].sum()==0:
                continue
            S[idx] /= S[idx].sum()

        return S

    def pagerank(self, A, eps=0.0001, d=0.85):
        P = np.ones(len(A)) / len(A)
        while True:
            new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
            delta = abs(new_P - P).sum()
            if delta <= eps:
                return new_P
            P = new_P


    def build_transition_matrix(self, links, index):
        total_links = 0
        A = np.zeros((len(index), len(index)))
        for webpage in links:
            # dangling page
            if not links[webpage]:
                # Assign equal probabilities to transition to all the other pages
                A[index[webpage]] = np.ones(len(index)) / len(index)
            else:
                for dest_webpage in links[webpage]:
                    total_links += 1
                    A[index[webpage]][index[dest_webpage]] = 1.0 / len(links[webpage])

        return A

    def get_summary(self, content):

        # Split the content into paragraphs
        paragraphs = self.split_content_to_paragraphs(content)

        # Add the title
        summary = []
        #summary.append(title.strip())
        summary.append("")

        # Add the best sentence from each paragraph
        for p in paragraphs:
            cleanParagraph = self.formatSentences(p)
            sentence = self.textrank(cleanParagraph, stopwords=stopwords.words('english')).strip()
            if sentence:
                summary.append(sentence)

        return ("\n").join(summary)


    def textrank(self, sentences, top_n=1, stopwords=None):
        """
        sentences = a list of sentences [[w11, w12, ...], [w21, w22, ...], ...]
        top_n = how may sentences the summary should contain
        stopwords = a list of stopwords
        """
        S = self.build_similarity_matrix(sentences, stop_words)
        sentence_ranks = self.pagerank(S)

        # Sort the sentence ranks
        ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
        selected_sentences = sorted(ranked_sentence_indexes[:top_n])
        summary = itemgetter(*selected_sentences)(sentences)
        return summary


def main():


    file = open("textfile.txt","w")

    st = SummaryTool()

    file = open('SampleFile.txt').read()
    #filteredText = st.formatSentences(file)
    #for idx, sentence in enumerate(st.textrank(filteredText, stopwords=stopwords.words('english'))):
    #    print("%s. %s" % ((idx + 1), ' '.join(sentence)))
    #print(st.formatSentences(file))
    print(st.get_summary(file))
    summary = st.get_summary(file)
    f = open('output.txt','w')
    f.write(summary)
    f.close()

if __name__ == '__main__':
    main()
