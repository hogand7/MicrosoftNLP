import spacy
nlp = spacy.load("en_core_web_lg")


#reading the file
class SummaryTool(object):

#splits the text into sentences and puts it into a list.
    def split_into_sentences(self,content):
        sentencelist = list()
        i=0
        for i, sentence in enumerate(content.sents):
            sentencelist.insert(i,sentence)
            #print('sentence')
            #print('sentencelist[i]')
            i= i+1
        return sentencelist

#splits the sentences into tokens and puts it into a list
    def split_into_tokens(self,content):
        tokenlist = list()
        i=0
        for token in content:
            tokenlist.insert(i,token)
            #print('token')
            #print('tokenlist[i]')
            i = i+1
        return tokenlist

#calculate the intersection between 2 sentences
    #def sentences_intersection(self,sent1,sent2):
     #  set1 = self.split_into_tokens(sent1)
      # set2 = self.split_into_tokens(sent2)

       #print(set1)
       #print(set2)

       #if (len(s1) + len(s2)) == 0:
        #   return 0

        # We normalize the result by the average number of words
        #return len(s1.intersection(s2)) / ((len(s1) + len(s2)) / 2)



def main():
        # Create a SummaryTool object
        st = SummaryTool()

        #nlp = spacy.load("en_core_web_lg")
        file = open('SampleFile.txt').read()
        document = nlp(file)
        sentence = st.split_into_tokens(document)
        #sent1 = sentence
    #    intersection = st.sentences_intersection(sentences,sentences)



if __name__ == '__main__':
    main()
