import json
import numpy
from sklearn.model_selection import train_test_split

DATA_PATH = "data\\"

def normalize2d(list):
    min_id = min([min(r) for r in list])
    dif = min_id - 1
    for i in range(len(list)):
        for j in range(len(list[i])):
            list[i][j] -= dif
    return list

def normalize1d(list):
    min_id = min(list)
    dif = min_id - 1
    for i in range(len(list)):
        list[i] -= dif
    return list

def read_data(file_name):
    with open(file_name, 'r') as json_file:
        json_list = list(json_file)
    return json_list

def get_products_dictionary():
    products_list = read_data(DATA_PATH+'products.jsonl')

    dictionary = {}

    for product in products_list:
        result = json.loads(product)
        dictionary[result["product_id"]] = result["category_path"]

    return dictionary

def get_categories_dictionary():
    products_list = read_data(DATA_PATH+'products.jsonl')
    
    dictionary = {}
    marker = 1

    for product in products_list:
        result = json.loads(product)
        dictionary[result["category_path"]] = 0

    for product in products_list:
        result = json.loads(product)
        if dictionary[result["category_path"]] == 0:
            dictionary[result["category_path"]] = marker
            marker += 1

    return dictionary

def get_sessions():
    session_list = read_data(DATA_PATH+'sessions.jsonl')

    product_sessions = []
    category_sessions = []
    product_session = []
    category_session = []
    products_dictionary = get_products_dictionary()
    category_dictionary = get_categories_dictionary()
    last_session_id = -1
    last_event_type = "VIEW_PRODUCT"

    for session_str in session_list:
        result = json.loads(session_str)
        if last_session_id == result["session_id"]:
            product_session.append(result["product_id"])
            category_session.append(category_dictionary.get(products_dictionary.get(result["product_id"])))
        else:
            if last_event_type == "BUY_PRODUCT":
                 product_sessions.append(product_session.copy())
                 category_sessions.append(category_session.copy())
            product_session.clear()
            category_session.clear()
            product_session.append(result["product_id"])
            category_session.append(category_dictionary.get(products_dictionary.get(result["product_id"])))
        last_session_id = result["session_id"]
        last_event_type = result["event_type"]

    if last_event_type == "BUY_PRODUCT":
         product_sessions.append(product_session.copy())
         category_sessions.append(category_session.copy())

    max_products = len(products_dictionary)
    max_categories = len(category_dictionary)
    return product_sessions, category_sessions, max_products, max_categories

def get_max_session_length(sessions):
    max_session_length = 0
    for session in sessions:
        length = len(session)
        if length > max_session_length:
            max_session_length = length
    return max_session_length

def to_numpy_sequences(sequences, max_sequence_length):
    return numpy.array([sequence[0:max_sequence_length]+[0]*(max_sequence_length-len(sequence)) for sequence in sequences])

def get_data(max_session_length=0, test_size=0.33, random_state=None):
    x1_list, x2_list, max_products, max_categories = get_sessions()
    y_list = [] 
    for x1, x2 in zip(x1_list, x2_list):
        y_list.append(x1.pop())
        x2.pop()
    
    init_max_session_length = get_max_session_length(x1_list)
    if max_session_length == 0 or max_session_length > init_max_session_length:
        max_session_length = init_max_session_length
    
    x1_list = normalize2d(x1_list)
    y_list = normalize1d(y_list)

    x1_list = to_numpy_sequences(x1_list, max_session_length)
    x2_list = to_numpy_sequences(x2_list, max_session_length)
    y_list = numpy.array(y_list)

    if test_size > 0:
        x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_list, x2_list, y_list, test_size=test_size, random_state=random_state)
        return (x1_train, x2_train, y_train), (x1_test, x2_test, y_test), max_products, max_categories
    else:
        return (x1_list, x2_list, y_list), ([], [], []), max_products, max_categories