def calc_rmse():
	fread = open('Pred_ratings_v3_12.txt','r')
	sum_error = 0.00
	count = 0
	for row in fread.readlines():
		rows = row.split('|')
		error = rows[4].split(':')
		sum_error = sum_error + float(error[1])*float(error[1])
		count = count + 1
	print count
	print sum_error/count


calc_rmse()