'''
        Author          : MagmaTart
        Last Modified   : 05/06/2018

        edge 레이블링을 토대로 real image 자동 레이블링 작업
'''

import os

# 시작 폴더 번호, 끝 폴더 번호
start_index, end_index = 7, 20

for i in range(start_index, end_index+1):
    if os.path.exists('./' + str(i)):
        ldir = [name for name in os.listdir('./' + str(i)) if 'edge' in name]
        print(ldir)

        for k in range(len(ldir)):
            file_number = ldir[k][:-4].split('-')[1][4:]
            file_label = ldir[k][:-4].split('-')[2]
            os.rename('./' + str(i) + '/' + 'hand-real' + file_number + '.png',
                      './' + str(i) + '/' + 'hand-real' + file_number + '-' + file_label + '.png')
