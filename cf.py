import numpy
import ast
import json
from datetime import datetime

class CFRecommender():
	movies = dict()				# Sr No: movie ID, Movie Name
	movie_users = dict()		# User ID : Sr No: Rating
	ratings = dict()			# User ID: Movie ID: rating
	users = []					# User IDs
	mean_rating = 0.00			# mean rating of Data set
	bu = []						# bu[User ID] = bu of user
	absbu = []					# |bu|
	bi = dict()					# movie ID: bi value
	absbi = dict()				# |bi|
	qi = dict()					# movie ID: [q1,q2,...]
	pu = dict()					# user ID: [p1,p2,...]
	gamma = 0.01				# Step Size for SGD
	lambda4 = 0.2				# Used for regularizing SGD
	''' Read data and store it in appropriate format '''
	def readData(self):

		print 'Reading Movies'
		fm = open("movies.csv","r")
		header = fm.readline()
		for movie in fm.readlines():
			details = movie.split(',')
			if details[2][:1] == '"':
				details[2] = details[2][1:]
			self.movies[int(details[0])] = [int(details[1]),details[2]]
		fm.close()

		print 'Read Ratings'
		fr = open("ratings.csv","r")
		header = fr.readline()
		for row in fr.readlines():
			rating = row.split(",")
			if self.ratings.has_key(int(rating[0])):
				rates = self.ratings[int(rating[0])]
				rates[int(rating[1])] = float(rating[2])
				self.ratings[int(rating[0])] = rates
			else:
				rate = dict()
				rate[int(rating[1])] = float(rating[2])
				self.ratings[int(rating[0])] = rate
			if self.movie_users.has_key(int(rating[1])):
				usrs = self.movie_users[int(rating[1])]
				usrs[int(rating[0])] = float(rating[2])
				self.movie_users[int(rating[1])] = usrs
			else:
				usr = dict()
				usr[int(rating[0])] = float(rating[2])
				self.movie_users[int(rating[1])] = usr
		print len(self.movie_users.viewkeys())
		print self.movie_users[1]
		for key in self.ratings.viewkeys():
			self.users.append(key)
		print self.users

	def calculateBaselineParams(self):
		# Calculate Mean Rating
		means = []
		for user in self.users:
			rate = self.ratings[user]
			means.append(numpy.mean(rate.values()))
		print 'means:', means
		self.mean_rating = numpy.mean(means)
		print 'mean rating:', self.mean_rating

		# Calculating bi with Lambda2 = 7
		for movieid, user_ratings in self.movie_users.viewitems():
			# print movieid, user_ratings
			add = sum(user_ratings.values())
			add = add - (self.mean_rating * len(user_ratings.viewvalues()))
			avg = float(add/len(user_ratings.viewvalues()))
			# print add, avg
			self.bi[movieid] = avg/7
			self.absbi[movieid] = abs(avg)/7
		print 'bi is ', self.bi.values()
		print self.absbi.values()
		print 'mean of bis is : ', numpy.mean(self.bi.values())
		print numpy.mean(self.absbi.values())

		# Calculating bu with Labda3 = 1
		for user in self.users:
			add = 0.00
			item = self.ratings[user]
			for mvid,rtng in item.items():
				# print mvid,rtng
				add = add + rtng - self.mean_rating - self.bi[mvid]
			add = add/len(item.viewitems())
			# print add
			self.bu.append(float(add))
			self.absbu.append(float(abs(add)))
		print 'bu is ',self.bu
		print self.absbu
		print 'mean of bu is :  ', numpy.mean(self.bu)
		print numpy.mean(self.absbu)


	def setdefaults(self):
		# intialize values of q
		q = numpy.ndarray(shape=(10,1),dtype=float)
		q.fill(0.1)
		print q
		print len(q), q.shape
		for srno, movie in self.movies.viewitems():
			self.qi[movie[0]] = q

		# initialize values of p
		p = numpy.ndarray(shape=(1, 10), dtype=float)
		p.fill(0.1)
		print p
		print len(p), p.shape
		for user in self.users:
			self.pu[user] = p
		print self.pu[5]

	def sgd(self):
		print 'This is SGD'
		for val in range(0,25,1):
			print 'Time at start of iteration ', val, ' is: ', str(datetime.now())
			for userid, rtng in sorted(self.ratings.viewitems()):
				movie_ratings = ast.literal_eval(str(rtng))
				cnt = 0
				while cnt <1000:
					cnt += 1
				for mid,r in sorted(movie_ratings.viewitems()):
					# Predict Movie Rating
					# print "predict Rating"
					q_movie = self.qi.get(mid)
					p_user = self.pu.get(userid)
					factor = p_user.dot(q_movie)
					pred_rating = self.mean_rating + self.bu[userid-1] + self.bi.get(mid) + factor[0][0]
					error = r - pred_rating
					# print 'Predicted Rating:',pred_rating, ' | Original Rating: ', r,' | Error: ', error
					fout = open('Pred_ratings_v3_'+str(val)+'.txt','a')
					fout.writelines('User Id:'+str(userid)+' | Movie Id:'+str(mid)+' | Predicted Rating:'+str(pred_rating)+ ' | Original Rating: '+ str(r)+' | Error: '+ str(error)+'\n')
					fout.close()
					if abs(error) < 0.01:
						continue
					else:		# Updating Parameter values
						self.bu[userid-1] = self.bu[userid-1] + self.gamma*(error - self.lambda4 * self.bu[userid-1])
						self.bi[mid] = self.bi.get(mid) + self.gamma*(error - self.lambda4 * self.bi.get(mid))
						qtemp = self.qi.get(mid)
						q = qtemp.transpose() + self.gamma*(error * self.pu.get(userid) - self.lambda4 * qtemp.transpose())
						self.qi[mid] = q.transpose()
						self.pu[userid] = self.pu.get(userid) + self.gamma*(error * self.qi.get(mid) - self.lambda4 * self.pu.get(userid))
		f = open('model.txt','w')
		f.write('Mean Rating$' + str(self.mean_rating) + '\n')
		f.write('bu$'+str(self.bu)+'\n')
		f.write('bi$'+str(json.dumps(self.bi))+'\n')
		f.write('qi$\n')
		for key, value in self.qi.items():
			arr = value
			f.write(str(key)+':'+str(arr.tolist())+'\n')
		f.write('pu$\n')
		for k, v in self.pu.items():
			a = v
			f.write(str(k)+':'+str(a.tolist())+'\n')
		f.close()

	def predict(self, puserid, pmovieid):
		fread = open('model.txt','r')
		flag = "n"
		for line in fread.readlines():
			if '$' in line:
				elements = line.split('$')
				if elements[0] == 'Mean Rating':
					self.mean_rating = float(elements[1])
					print self.mean_rating
				elif elements[0] == 'bu':
					self.bu = ast.literal_eval(elements[1])
					print self.bu
				elif elements[0] == 'bi':
					asdict = ast.literal_eval(elements[1])
					asdict = {int(k):v for k,v in asdict.items()}
					self.bi = asdict
					print self.bi
				elif elements[0] == 'qi':
					flag = "q"
				elif elements[0] == 'pi':
					flag = "p"
			elif flag == "q":
				print "Read qi"
				element = line.split(':')
				print len(element)
				s_index = element[1].index('[',1)
				e_index = element[1].index(']',1)
				tempvar = element[1][s_index:e_index+1]
				tempvar = ast.literal_eval(tempvar)
				arr = numpy.asarray(tempvar)
				print arr.shape
				self.qi[int(element[0])] = arr
			elif flag == "p":
				print "Read pu"
				elements = line.split(':')
				self.pu[int(elements[0])] = elements[1]
			elif flag == "n":
				print "Flag is Not Set"
		# 	Predict Rating
		q_movie = self.qi.get(pmovieid)
		p_user = self.pu.get(puserid)
		factor = p_user.dot(q_movie)
		pred_rating = self.mean_rating + self.bu[puserid - 1] + self.bi.get(pmovieid) + factor[0][0]
		print "Predicted Rating for User: ",puserid," and Movie Id: ", pmovieid," is: ",pred_rating
		return pred_rating


print 'Start time is: ', str(datetime.now())
cf = CFRecommender()
# cf.readData()
# cf.calculateBaselineParams()
# cf.setdefaults()
# cf.sgd()
rating = cf.predict(1,12)
print rating
print 'End time is: ', str(datetime.now())
print 'Program Completed'
