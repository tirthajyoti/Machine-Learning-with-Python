list_to_pickle = [1, 'here', 123, 'walker']
print("List to pickle: ",list_to_pickle)

#Pickling the list
import pickle

list_pickle = pickle.dumps(list_to_pickle)

print("Pickled list: ",list_pickle)

loaded_pickle = pickle.loads(list_pickle)

print("After loading pickle: ",loaded_pickle)
