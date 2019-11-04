#
from preprocess import *

text = read_data(False, True, False)
'''
print ('len(text)=\n', len(text))
print ('type(text)=\n', type(text))

for i in range(len(text)):
    print (text[i])
    print (type(text[i]))
    if i == 2:
        break
'''
fact_data = extract_labels(text,'fact')
'''
print ('len(fact_data)=\n', len(fact_data))
print ('type(fact_data)=\n', type(fact_data))

for i in range(len(fact_data)):
    print (fact_data[i])
    print (type(fact_data[i]))
    if i == 2:
        break
'''
token_fact_data = tokenize(fact_data)
print ('len(token_fact_data)=%s \n' % len(token_fact_data))
print ('type(token_fact_data)=%s \n' % type(token_fact_data))

for i in range(len(token_fact_data)):
    print (token_fact_data[i])
    print (type(token_fact_data[i]))
    if i == 2:
        break
