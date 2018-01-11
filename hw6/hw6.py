import os, sys, csv, numpy, pandas

# command: python3 hw6.py image.npy test_case.csv out.csv
class_label = numpy.load('./image_id_mapping.npy')
test_case = pandas.read_csv(sys.argv[2], sep=',', dtype=int).values

with open(sys.argv[3], 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['ID', 'Ans'])
    for i in range(test_case.shape[0]):
        if class_label[ int( test_case[i][1] ) ] == class_label[ int( test_case[i][2] ) ]:
            spamwriter.writerow([ test_case[i][0] , '1' ])
        else:
            spamwriter.writerow([ test_case[i][0] , '0' ])
