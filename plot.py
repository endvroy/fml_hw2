import matplotlib.pyplot as plt

cv_perfs = [(1, 4096, 79.52076677316295), (2, 4096, 80.09584664536742), (3, 4096, 79.48881789137381),
            (4, 4096, 78.97763578274763)]
test_record = [(1, 4096, 77.83524904214559), (2, 4096, 78.05555555555557), (3, 4096, 77.1264367816092),
               (4, 4096, 77.20306513409962)]

cv_perfs = [(1, 8, 73.70607028753993), (2, 8, 73.3226837060703), (3, 8, 73.51437699680511), (4, 8, 73.6741214057508)]
test_record =[(1, 1024, 77.66773162939297), (2, 1024, 75.01597444089457), (3, 1024, 73.9297124600639), (4, 1024, 73.51437699680511)]
# sv_record = [(1, 4096, 136.6), (2, 4096, 135.8), (3, 4096, 140.1), (4, 4096, 145.9)]
# margin_record = [(1, 4096, 79.52076677316295), (2, 4096, 80.09584664536742), (3, 4096, 79.48881789137381),
#                  (4, 4096, 78.97763578274763)]

plt.title('Cross Validation Error')
plt.xlabel('d')
plt.xticks([1, 2, 3, 4])
plt.ylabel('Error')
plt.plot(range(1, 5), [100 - x[2] for x in cv_perfs])
plt.savefig('cv_error_6.png')
plt.close()

plt.title('Test Error')
plt.xlabel('d')
plt.xticks([1, 2, 3, 4])
plt.ylabel('Error')
plt.plot(range(1, 5), [100 - x[2] for x in test_record])
plt.savefig('test_error_6.png')
plt.close()

# plt.title('Average Support Vectors')
# plt.xlabel('d')
# plt.xticks([1, 2, 3, 4])
# plt.ylabel('Count')
# plt.plot(range(1, 5), [x[2] for x in sv_record])
# plt.savefig('sv_cnt.png')
# plt.close()
#
# plt.title('Average Support Vectors on the Margin Hyperplane')
# plt.xlabel('d')
# plt.xticks([1, 2, 3, 4])
# plt.ylabel('Count')
# plt.plot(range(1, 5), [x[2] for x in margin_record])
# plt.savefig('margin_cnt.png')
# plt.close()
