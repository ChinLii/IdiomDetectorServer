dev_set is a development(training set)
test_set is a testing the algorithms.(please use test_set_noclass for testing or you can hide class column)
VNAA and VNAAD txt files include just only some types of words and they were created from postags.py files.
so, you dont need to use postag_1_VNAA.py and postag_2_VNAAD.py files, these were used only get the pos tag results.

we have 3 important files in there;
1-bagofwords_extraction.py(use dev_set.txt for train the algorithm)
2-VNAA_extraction.py(use VNAA.txt for train the algorithm)
3-VNAAD_extraction.py(use VNAAD.txt for train the algorithm)
you need to use these three files and train your selected algorithm on them also test_set for validation. 
and need to share results :)

keep finger crossed :)
