import os

if os.path.isfile('report.html'):
    string = 'PROPER OUTPUT FILE EXISTS'
else:
    string = 'PROPER OUTPUT FILE DNE'

with open('output_test.txt', 'w') as f:
    f.write(string)

