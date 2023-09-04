import os
def txt8to4():
	# src = "C:\\Users\\xiu\\Desktop\\xiu\\"  #dont place the outcome to the desktop
	# dst = "C:\\Users\\xiu\\Desktop\\Trans\\"
	src = "F:\\UAVTest\\result-UAV150-FANet"
	dst = "F:\\UAVTest\\TestConvert"
	txt_name = os.listdir(src)
	#txt_name.remove('4')
	for txt in txt_name:
		print(txt)
		txt_path = os.path.join(src, txt)
		file_txt = open(txt_path)
		txt_new_path = os.path.join(dst, txt)
		x = open(txt_new_path, 'w').close()  #clear the txt
		loc_4 = []
		for line in file_txt.readlines():
			curline = line.strip().split(' ')
			intline = curline
			loc_4.append(int(float(intline[0])))
			loc_4.append(int(float(intline[1])))
			loc_4.append(int(float(intline[2]) - float(intline[0])))
			loc_4.append(int(float(intline[7]) - float(intline[1])))
			with open(txt_new_path,'a') as fiel_txtnew:
				count = 0
				for k in loc_4:
					count +=1
					fiel_txtnew.write(str(k))
					if count<4:
						fiel_txtnew.write(',')
				fiel_txtnew.write('\n')
			loc_4=[]

		file_txt.close()
if __name__ == "__main__":
    txt8to4()