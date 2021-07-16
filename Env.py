    # Import routines

import numpy as np
import math
import random
from sklearn.preprocessing import OneHotEncoder

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(city_A,city_B) for city_A in range(m) for city_B in range(m) if city_A!=city_B or city_A==0]
        self.state_space = [(state_city, state_hour, state_day) for state_city in range(m) for state_hour in range(t) for state_day in range(d)]
        self.state_init = random.sample(list(self.state_space),1)[0]

        # Generate one hot encoded values for cities, hours and days
        input_encode_city = np.asarray(range(0,m)).reshape(m,1)
        input_encode_hour = np.asarray(range(0,t)).reshape(t,1)
        input_encode_day = np.asarray(range(0,d)).reshape(d,1)

        # Define length of an episode
        self.total_cab_time = 0

        _encode_city = OneHotEncoder(sparse=False)
        self.encode_value_city = _encode_city.fit_transform(input_encode_city)
        _encode_hour = OneHotEncoder(sparse=False)
        self.encode_value_hour = _encode_hour.fit_transform(input_encode_hour)
        _encode_day = OneHotEncoder(sparse=False)
        self.encode_value_day = _encode_day.fit_transform(input_encode_day)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        try:
            city_encode = self.encode_value_city[state[0]]
            hour_encode = self.encode_value_hour[state[1]]
            day_encode = self.encode_value_day[state[2]]
            state_encod = np.concatenate((city_encode, hour_encode, day_encode))
            return state_encod
            # print(self.encode_value_city[state[0]-1])
        except IndexError as e:
            print("State in index error {}".format(state))
            return "Please enter a valid City(0-4)/Time(0-23)/Day of the week(0-6) {}".format(e)


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        elif location == 1:
            requests = np.random.poisson(12)
        elif location == 2:
            requests = np.random.poisson(4)
        elif location == 3:
            requests = np.random.poisson(7)
        elif location == 4:
            requests = np.random.poisson(8)
        if requests >15:
            requests =15

        # print("m {}, requests {}".format(m, requests))


        possible_actions_index = random.sample(range(1, ((m-1)*m) +1), requests) + [0] # (0,0) is not considered as customer request
        # print("++++++ Actions space ++++++ {}".format(self.action_space))
        # print("possible_actions_index {}".format(possible_actions_index))
        actions = [self.action_space[i] for i in possible_actions_index]

        
        actions.append((0,0))
        # print("Actions - {}".format(actions))

        return possible_actions_index, actions



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        # print("action - {}".format(action))
        current_point = state[0]
        pickup_point = action[0]
        # print("pickup_point {}".format(pickup_point))
        drop_point = action[1]
        current_hour = state[1]
        current_day = state[2]
        pickup_hour = 0
        pickup_day = current_day
        try:
            time_taken_current_to_pickup = Time_matrix[current_point][pickup_point][current_hour][current_day]
            if (time_taken_current_to_pickup + current_hour) > 23:
                if current_day + 1 > 6:
                    pickup_day = 0
                else:
                    pickup_day += 1
                pickup_hour = int((time_taken_current_to_pickup + current_hour) - 24)
            else:
                pickup_hour = int(time_taken_current_to_pickup + current_hour)
            time_taken_pickup_to_drop = Time_matrix[pickup_point][drop_point][pickup_hour][pickup_day]
            # Calculating reward
            if action == (0,0):
                reward = -C
            else:
                reward = (R * time_taken_pickup_to_drop) - C * (time_taken_pickup_to_drop + time_taken_current_to_pickup)
            return reward

        except IndexError as e:
            print("Error: {}, pickup point {}".format(e, pickup_point))




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        #print("self.total_cab_time ", self.total_cab_time)
        current_point = state[0]
        pickup_point = action[0]
        drop_point = action[1]
        next_hour = current_hour = state[1]
        next_day = current_day = state[2]
        pickup_hour = 0
        pickup_day = 0
        terminal_state = False
        # print("Chosen next Action ", action)
        if action == (0,0):
            if (current_hour + 1) > 23:
                if (current_day + 1) > 6:
                    next_day = 0
                else:
                    next_day += 1
                next_hour = (current_hour + 1) - 24
            else:
                next_hour += 1
            drop_point = current_point
            present_cab_ride_time = 1
        else:
            time_taken_current_to_pickup = Time_matrix[current_point][pickup_point][current_hour][current_day]
            if (time_taken_current_to_pickup + current_hour) > 23:
                if (current_day + 1) > 6:
                    next_day = pickup_day = 0
                else:
                    pickup_day = current_day + 1
                    next_day = pickup_day
                pickup_hour = int((time_taken_current_to_pickup + current_hour) - 24)
            else:
                pickup_hour = int(time_taken_current_to_pickup + current_hour)
                pickup_day = current_day
            time_taken_pickup_to_drop = Time_matrix[pickup_point][drop_point][pickup_hour][pickup_day]
            if (pickup_hour + time_taken_pickup_to_drop) > 23:
                if pickup_day + 1 > 6:
                    next_day = 0
                else:
                    next_day = pickup_day + 1
                next_hour = (pickup_hour + time_taken_pickup_to_drop) - 24
            else:
                next_hour = pickup_hour + time_taken_pickup_to_drop
                next_day = pickup_day
            present_cab_ride_time = (time_taken_current_to_pickup + time_taken_pickup_to_drop)
        # print("present_cab_ride_time ", present_cab_ride_time)
        self.update_total_cab_time(present_cab_ride_time)
        if self.total_cab_time >= 720:
        	terminal_state = True
        # print("total_cab_time {}, terminal_state {}".format(self.total_cab_time, terminal_state))
        next_state = [int(drop_point), int(next_hour), int(next_day)]
        return terminal_state, next_state

    def update_total_cab_time(self, present_cab_time):
        #print("total_cab_time {}, present_cab_time {}".format(self.total_cab_time, present_cab_time))
        self.total_cab_time += present_cab_time


    def reset(self):
        return self.action_space, self.state_space, self.state_init

if __name__ == "__main__":
	x = CabDriver()
	for i in range(20):
		x.update_total_cab_time(i)
