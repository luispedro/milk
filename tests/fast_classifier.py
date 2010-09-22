import numpy as np
class fast_classifier(object):
    def __init__(self):
        pass

    def train(self, features, labels):
        examples = {}
        for f,lab in zip(features, labels):
            if lab not in examples:
                examples[lab] = f
        return fast_model(examples)

class fast_model(object):
    def __init__(self, examples):
        self.examples = examples
    
    def apply(self, features):
        res = []
        for f in features:
            cur = None
            best = +np.inf
            for k,v in self.examples.iteritems(): 
                dist = np.dot(v-f, v-f)
                if dist < best:
                    best = dist
                    cur = k
            res.append(k)
        return res


