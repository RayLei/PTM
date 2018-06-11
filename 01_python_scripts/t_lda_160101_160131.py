
import lda
#import sqlite3
#import numpy as np
import lda_util as lu


directory='/Users/leihao/Downloads/Data/'
sqlite_file=directory+'nasdaq.db'
start_date, end_date='2016-01-01', '2016-01-31'

# Output is a dataframe contains date, articles
df_art = lu.article_extractor(sqlite_file,start_date, end_date)

# Convert the articles to doc_wds_mat & vocab
doc_wds_mat, voc = lu.article2matrix(df_art.article)

# Save the matrix & vocab
out_dir = '/Users/leihao/Documents/Git/PTM/02_output/'
out_prefix = 't_'
date_range = '160101_160131'
mat_file = out_dir + out_prefix + 'doc_wds' + date_range + '.mat'
voc_file = out_dir + out_prefix + 'vocab' + date_range + '.txt'

# Row 9355 has all zero entries
# Need to remove it.
doc_wds_mat = lu.zero_row_remove(doc_wds_mat)

lu.matrix_dump(doc_wds_mat, mat_file)
lu.vocab_write(voc, voc_file)

# Convert the doc_wds into LDA-C format for PTM
ldac_mat = lda.utils.dtm2ldac(doc_wds_mat)
with open(out_dir+'t_ldac_doc_wds_160101_160131.txt','w+') as ldac_f:
    for item in ldac_mat:
        ldac_f.write(item+'\n')

# Run LDA
lu.lda_out(doc_wds_mat, voc, out_dir, out_prefix, 20, 20, date_range)
