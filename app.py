from flask import Flask, request
import pandas as pd
import numpy as np
import datetime
from joblib import load

app = Flask(__name__)


walk_model = load('walk_decision_tree_model.joblib')
ptr_model = load('ptr_decision_tree_model.joblib')
ptm_model = load('ptm_decision_tree_model.joblib')
carr_model = load('carr_decision_tree_model.joblib')
carm_model = load('carm_decision_tree_model.joblib')

def predict_travel_time(from_id, to_id, time, mode):
    """
    Predict the travel time between two locations given the time of day and preferred mode of transportation.
    
    Parameters
    ----------
    from_id : int
        The ID of the starting location.
    to_id : int
        The ID of the destination location.
    time : datetime.datetime
        The time of day for the travel, in the format '%H:%M'.
    mode : str
        The preferred mode of transportation. Possible values are 'walk', 'public_transit', or 'car'.
    
    Returns
    -------
    str
        The predicted travel time in minutes, as a string.
    """
    
    input = np.array([[from_id, to_id]])

    start_r_time1 = datetime.datetime.strptime('08:00', '%H:%M').time()
    end_r_time1 = datetime.datetime.strptime('09:00', '%H:%M').time()
    start_r_time2 = datetime.datetime.strptime('16:00', '%H:%M').time()
    end_r_time2 = datetime.datetime.strptime('18:00', '%H:%M').time()

    if mode == "walk":
        travel_time = walk_model.predict(input)
        travel_time_str = str(int(travel_time))
        return travel_time_str
    elif mode == "public_transit":
        if start_r_time1 <= time.time() <= end_r_time1 or start_r_time2 <= time.time() <= end_r_time2:
            travel_time = ptr_model.predict(input)
            travel_time_str = str(int(travel_time))
        else:
            travel_time = ptm_model.predict(input)
            travel_time_str = str(int(travel_time))
    elif mode == "car":
        if start_r_time1 <= time.time() <= end_r_time1 or start_r_time2 <= time.time() <= end_r_time2:
            travel_time = carr_model.predict(input)
            travel_time_str = str(travel_time)
        else:
            travel_time = carm_model.predict(input)
            travel_time_str = str(travel_time)   
    else:
        return "invalid input"
    
    return travel_time_str        

# Define the API endpoint and request method
@app.route('/travel_time', methods=['GET'])
def travel_time():
    """
    Endpoint for getting travel time prediction between two locations ids. 

    Returns:
    JSON object containing travel time prediction in minutes.

    Example:
    {
        "travel_time": "35 minutes"
    }
    """

    # Parse input data from request JSON
    data = request.get_json()
    origin = data['origin']
    destination = data['destination']
    time_of_day = data['time']
    preferred_mode = data['mode']

    # Converts time of day string to datetime object
    time_of_day_obj = datetime.datetime.strptime(time_of_day, '%H:%M')

    # Makes the travel time prediction using a trained model, and construct result a JSON object and returns it
    result = {'travel_time': predict_travel_time(origin, destination, time_of_day_obj, preferred_mode) + ' minutes'}
    print(result)
    return result

if __name__ == '__main__':
    app.run(debug=True)
