import csv
import re

#names= ['petya', 'wannacry', 'locky', 'cryptolocker', 'dharma', 'teslacrypt', 'alphacrypt', 'tesla #crypt', 'alpha crypt', 'cerber', 'zeus', 'zbot', 'mirai', 'dyn', 'stuxnet', 'flame', 'duqu', 'sony #pictures hack', 'guardians of peace', 'trickbot trojan', 'exploit kit', 'angler exploit kit', 'nuclear #exploit kit', 'red october', 'rocra', 'cyber espionage']

mal_f1 = open("wannacry.csv","a")
wannacry = csv.writer(mal_f1, delimiter=',')


mal_f2 = open("locky.csv","a")
locky = csv.writer(mal_f2, delimiter=',')


mal_f3 =  open("cerber.csv","a")
cerber= csv.writer(mal_f3, delimiter=',')


mal_f4 = open("cryptolocker.csv","a")
crypt= csv.writer(mal_f4, delimiter=',')



mal_f5 = open("al_tesla.csv","a")
alpha = csv.writer(mal_f5, delimiter=',')


mal_f1t = open("red_oct.csv","a")
red= csv.writer(mal_f1t, delimiter=',')

mal_f2t = open("zeus.csv","a")
zeus = csv.writer(mal_f2t, delimiter=',')

mal_f3t =  open("stuxnet.csv","a")
stux= csv.writer(mal_f3t, delimiter=',')

mal_f5t =  open("mirai.csv","a")
mirai= csv.writer(mal_f5t, delimiter=',')

mal_f6t =  open("petya.csv","a")
petya = csv.writer(mal_f6t, delimiter=',')

mal_f7t =  open("duqu.csv","a")
duqu= csv.writer(mal_f7t, delimiter=',')

mal_f8t =  open("trickbot.csv","a")
trickbot= csv.writer(mal_f8t, delimiter=',')

mal_f9t =  open("sony.csv","a")
sony= csv.writer(mal_f9t, delimiter=',')

mal_f6 =  open("aek.csv","a")
aek= csv.writer(mal_f6, delimiter=',')

mal_f7 =  open("nek.csv","a")
nek= csv.writer(mal_f7, delimiter=',')

wannacry_w =['wannacry', 'eternalblue', 'server message block', 'may 2017', 'worldwide cyberattack', 'wannacry ransomware cryptoworm', 'kill switch', 'shadow brokers', 'double pulsar', 'tasksche.exe', 'original rsa public key', 'north korean hackers', 'nsa', 'lazarus', 'marcus hutchins', 'malware tech']

locky_w =['locky', 'invoice requiring payment', 'invoice', 'social engineering', 'enable macros', 'download tor browser', 'hash.locky', '.bmp', '.txt', 'necurs', 'gods of norse', 'egyptian mythology', '.locky', '.osiris', '.zepto', '.oden', '.shit', '.aesir', 'asiris', 'spam campaigns', 'malicious macro', 'dridex']

cerber_w =['cerber', 'cerber ransomware', 'underground russian forum', 'mythological creatures', 'csrstub.exe', 'dinotify.exe', 'rasdial.exe', 'relpost.exe', 'ntkrnlpa.exe', '.cerber', 'square.bmp', '.rc4', 'rsa','crypto ransomware']

crypt_w =['cryptolocker', 'cryptolocker ransomware attack', 'cryptovirus', 'cryptowall', 'torrentlocker','cryptolocker.f', 'rsa public-key cryptography', '2048-bit rsa key pair', 'infected email attachment']

alpha_w = ['teslacrypt', 'alphacrypt', 'tesla crypt', 'alpha crypt', 'aes encryption', 'tesladecoder' , 'tesla decoder', 'shadow volume copies', '.bmp file', 'teslacrypt decryption'] 

red_w = ['red october', 'rocra', 'cyber espionage', 'cyberespionage', 'advanced cyber esponiage network', 'advanced cyber esponiage campaign']

zeus_w = ['zeus', 'zbot', 'russian hacker', 'zeus trojan', 'zeusvm', 'zeus banking trojan' , 'email attachment' , 'zitmo' , 'malicious gmail message', 'man in the browser keystroke logging' , 'credit card details']

stux_w = ['stuxnet' , 'iran nuclear program' , 'programmable logic controllers', 'nuclear program', 'usb flash drive', 'rootkit' , 'equation group', 'f-secure' , 'virus blokada' , 'rootkit.timphider', 'operation olympic games']

mirai_w = ['mirai','ip cameras', 'home routers', 'ddos', 'ovh', 'dyn', 'deutsche telecom', 'talktalk', 'liberia']

petya_w =[ 'petya','notpetya', 'goldeneye', 'ukraine', 'lateral movement', 'eternal romance', 'master boot record', 'wiper', 'm.e. doc', 'expetr', 'mischa', 'vaccine']

duqu_w = ['duqu', 'government malware' , 'israeli campaign' , 'ps + 1' , 'foxconn' , 'cyberespionage', 'sabotage iranian nuclear development program' ,' windows kernel', 'nation-rates']

trickbot_w = ['trickbot' , 'crm system' , 'trickloader', 'paypal' , 'macro embedded docs' , 'rig exploit kit', 'dyre successor' , 'malvertising campaign', "dyreza's successor"]

sony_w = ['film studio sony pictures', "god's apstis", 'fireeye' , 'shamon wiper', 'sony pictures hack', 'guardians of peace', 'north korea']

aek_w = [' angler exploit kit' , 'work behind the scenes' , 'targets a vulnerability', 'malware-as-a-service' , 'ek-as-a-service', 'platform-as-a-service', 'ekaas', 'maas', 'paas']

nek_w = ['nuclear exploit kit', 'maas' , 'malware-as-a-service' , 'nuclear server' , 'nuclear control panel']

with open("sch_blog.csv", "r") as f:
	reader = csv.reader(f, delimiter = ",")

	for row in reader:
		text = row[1].lower()
		for word in wannacry_w:
			if re.search(word, text):
				wannacry.writerow(row)	
				print ' "%s" found in text' % (word)
				break
				
		for word in locky_w:
			if re.search(word, text):
				locky.writerow(row)	
				print ' "%s" found in text' % (word)
				break
				
		for word in cerber_w:
			if re.search(word, text):
				cerber.writerow(row)	
				print ' "%s" found in text' % (word)
				break
				
		for word in crypt_w:
			if re.search(word, text):
				crypt.writerow(row)	
				print ' "%s" found in text' % (word)
				break
				
		for word in alpha_w:
			if re.search(word, text):
				alpha.writerow(row)	
				print ' "%s" found in text' % (word)
				break
				
		for word in red_w:
			if re.search(word, text):
				red.writerow(row)	
				print ' "%s" found in text' % (word)
				break
				
		for word in zeus_w:
			if re.search(word, text):
				zeus.writerow(row)	
				print ' "%s" found in text' % (word)
				break
				
		for word in stux_w:
			if re.search(word, text):
				stux.writerow(row)	
				print ' "%s" found in text' % (word)
				break
				
		for word in mirai_w:
			if re.search(word, text):
				mirai.writerow(row)
				print ' "%s" found in text' % (word)
				break
				
		for word in petya_w:
			if re.search(word, text):
				petya.writerow(row)	
				print ' "%s" found in text' % (word)
				break
				
		for word in duqu_w:
			if re.search(word, text):
				duqu.writerow(row)	
				print ' "%s" found in text' % (word)
				break
				
		for word in trickbot_w:
			if re.search(word, text):
				trickbot.writerow(row)
				print ' "%s" found in text' % (word)
				break
				
		for word in sony_w:
			if re.search(word, text):
				sony.writerow(row)
				print ' "%s" found in text' % (word)
				break
				
		for word in aek_w:
			if re.search(word, text):
				aek.writerow(row)
					
				print ' "%s" found in text' % (word)
				break
				
		for word in nek_w:
			if re.search(word, text):
				nek.writerow(row)
					
				print ' "%s" found in text' % (word)
				break
				
	##avg lengu			
