import json
import requests
import pandas as pd

# Setting the headers to send and accept json responses
header = {'Content-Type': 'application/json', 'Accept': 'application/json'}

# Read test dataset
test_df = pd.read_csv("data/housing_test.csv")
# For demo purpose, only 6 data points are passed on to the prediction server endpoint.
# Feel free to change this index or use the whole test dataset
test_df = test_df.iloc[40:46]

# Drop the first column
test_df.drop(test_df.columns[0],axis=1,inplace=True)

# Converting Pandas Dataframe to json
data = test_df.to_json()

#print(data)
print("Sending data...")

resp = requests.post("http://0.0.0.0:5000/predict", \
                    data = json.dumps(data),\
                    headers= header)

if (str(resp.status_code)=='200'):
	print("Response received correctly.")
	print()
	
x=resp.json()
j = json.loads(x)
d = dict(j)

for k,v in (d.items()):
	print("{}: {}".format(k,round(v,2)))
	print()
