import pickle

def save(name, val):
    with open('Saved data/' + name + '.pkl', 'wb') as f:
        pickle.dump(val, f)


def load(name):
    with open('Saved data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)