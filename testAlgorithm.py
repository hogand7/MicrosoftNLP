import difflib ##import additional module for compare
#Previous code before function testing call, must get the summary and put it into a list by inserting a sentence at each index

#testing: two list parameters, prints out the difference of the summaries
def testing( summaryA , summaryB):

    # initiate the Differ object
    d = difflib.Differ()

    # calculate the difference between the two texts
    diff = d.compare(summaryA, summaryB)

    # output the result
    print ('\n'.join(diff)) # - : word missing, + : word added, ? - shows this wprd with ^^, added missed whole sentences at end
                            # will print whole summary if theres no mismatch


def main():
    # define main summary
    mainSummary = ["About the IIS", "", "IIS 8.5 has several improvements related", "to performance in large-scale scenarios, such", "as those used by commercial hosting providers and Microsoft's", "own cloud offerings."]

    # define test summary
    testSummary = ["About the IIS", "", "It has several improvements related", "to performance in large-scale scenarios."]

    testing(mainSummary,testSummary)

if __name__ == '__main__':
    main()
