from flask import Flask
from flask_restful import Api, reqparse, Resource, fields, marshal_with
import pickle
import pandas as pd

app = Flask(__name__)
api = Api(app)

# import model
path = r'C:\Users\Breno\PycharmProjects\ML_API_base\models\titanic_model'
with open (path, 'rb') as file:
    model = pickle.load(file)

# the parser will look through the parameter that the user send to API
parser = reqparse.RequestParser()
parser.add_argument('Pclass', type=int, help = 'You need to add your probably Pclass', required = True)
parser.add_argument('Age', type=int, help = 'You need to add your Age', required=True)
parser.add_argument('Sex', type=int, help = 'You need to add your Sex', required=True)
parser.add_argument('Fare', type=float, help = 'You need to add the Fare you could probably pay', required=True)
parser.add_argument('SibSp', type=int, help = 'You need to add your SibSp', required=True)
parser.add_argument('Parch', type=int, help = 'You need to add your Parch', required=True)

# class for HTTP methods to serve the API
class TITANIC_API(Resource):

    def get(self):
        #taking the arguments
        args = parser.parse_args()
        user_pclass = args['Pclass']
        user_age = args['Age']
        user_sex = args['Sex']
        user_fare = args['Fare']
        user_sibsp = args['SibSp']
        user_parch = args['Parch']

        #passing the user data to a dataframe (array)
        user = pd.DataFrame({'Pclass': user_pclass, 'Sex': user_sex, 'Age': user_age, 'SibSp': user_sibsp,
                             'Parch': user_parch, 'Fare': user_fare}, index=[0])
        # making the predition if this person would be alive or not
        user_prob = model.predict_proba(user)
        prob = round((user_prob[0][1]) * 100, 2)

        #text changes acording to survival probability
        if prob < 50.0:
            text = "You probably wouldn't survival to Titanic"
        else:
            text = 'You would probably survival to Titanic'

        output = {
                'Prediction': text, 'Probability of survive': prob
            }

        return output


api.add_resource(TITANIC_API, '/API/')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
