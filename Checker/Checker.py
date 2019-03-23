import spacy
import os
import csv
import sys
import errno
import difflib ##import additional module for compare

nlp = spacy.load("en_core_web_lg")

class summaryChecker(object):

    #Function which reads multiple files
    def read_multiple_files(self,path):
        file_list=list()
        i=0
        for file in [txt for txt in os.listdir(path)if txt.endswith('.txt')]:
            f = file_list.append(file)
        return file_list


    #Splits the summaries into sentences
    def split_content_to_sentences(self, content):
        content = content.replace("\n", ". ")
        return content.split(". ")





def main():
        # Create a SummaryTool object
        st = summaryChecker()

        #Reading multiple files
        path = 'C:\\Users\\Nissimol Aji\\AtomWorkSpace\\NLP\\Src\\TextFiles'
        read_files = st.read_multiple_files(path)
        for file in read_files:
            filepath = os.path.join(path,file)
            f = open(filepath,'r')
            file = f.read()
            #print(file)
            sentence = st.split_content_to_sentences(file)
            print(sentence)
            print("\n")
            f.close()


if __name__ == '__main__':
    main()
