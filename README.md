# titanic_API
API project that returns a machine learning prediction about probability of an user survives to the Titanic disaster.

The machine learn model is based on the Titanic dataset from Kaggle. Its a very simple code, using Logistc regression model, that creates a binary file saved as 'titanic_model'.
That file will be loaded on api.py and used by the Flask REST API who returns a jSON file with a text describing if the user survived or not and the probability of this user survives (0-100%)

This project focous on the API, not the machine learn model.

Machine learning variables meaning:

Variable   /    	Definition	   /     Key

#survival	  /    Survival	  /   0 = No, 1 = Yes

#pclass	   /    Ticket class	 / 1 = 1st, 2 = 2nd, 3 = 3rd

#sex	        /      Sex	     /      0- male, 1- female

#Age	        /      Age         /      in years	

#sibsp	    /     number of siblings / spouses aboard the Titanic 

#parch	     /    number of parents / children aboard the Titanic	

#fare	      /   Passenger fare	   most parte of the fare is around 7.000 and 100.000 depends if you are a wealthy person or not.

That API was made to be consumed by a simple website page who takes the user information from a formulary and redirect the user to two diferent pages, acording to chances of survival.
Link to the website code:

https://github.com/Breno-Valle/titanic_website
