import csv
import json

def convertJson(filename):
  with open(filename, "rt", encoding="latin-1") as csvfile:
    datareader = csv.reader(csvfile)
    yield next(datareader)
    for row in datareader:
      yield row
    return

def getdata(filename):
  for row in convertJson(filename):
    yield row

newArry = []
count = 0
for row in getdata("1600000Tweets.csv"):
  count = count + 1
  if count > 500:
    break
  newArry.append({"target": row[0], "id": row[1], "date": row[2], "flag": row[3], "user": row[4], "text": row[5]})

with open('output.json', 'w') as outfile:
    json.dump(newArry, outfile)
