from sklearn.metrics import average_precision_score
class CatCustomAveragePrecisionScore():
    
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        approx = approxes[0]
        approx_list = []
        target_list = []
        for i in range(len(approx)):
            target_list.append(target[i])
            approx_list.append(1+approx[i])
        aps = average_precision_score(target_list, approx_list)
        # approxes - list of list-like objects (one object per approx dimension)
        # target - list-like object
        # weight - list-like object, can be None
        return aps, 0.0