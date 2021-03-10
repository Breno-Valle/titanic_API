import requests
BASE = 'http://127.0.0.1:5000/'

data = {'Age': 22, 'Pclass': 2, 'Sex': 1, 'Fare': 20.500, 'Parch': 1, 'SibSp': 0}

#use requests for test this API
response = requests.get(BASE + 'API/', data)
json_file = response.json()
print(response.json())
print(json_file['Prediction'])
