from math import exp, log
from operator import itemgetter

'''
AdaBoost implementation with weighted voting as a decision procedure
'''
class weighted_voting_adaboost(object):
    # initializes with already built classifiers and corresponding 
    def __init__(self, in_classifiers, in_coefficients):
        self.classifiers = in_classifiers
        self.coefficients = in_coefficients
    
    # decision by weighted voting
    def apply(self, in_features):
        # a "class number" => "votes value" mapping
        answers = {}
        for classifier, coefficient in zip(self.classifiers, self.coefficients):
            answer = classifier.apply(in_features)
            if answer in answers:
                answers[answer] += coefficient
            else:
                answers[answer] = coefficient
        # dict maximum by value
        result = max(answers.iteritems(), key=itemgetter(1))
        return result[0]
         

class weighted_voting_ada_learner(object):
    def __init__(self, in_composition_size, in_learner):
        self.learner = in_learner
        self.composition_size = in_composition_size
    
    def reset(self, in_features):
        self.classifiers = []
        # linear coefficients for the classifiers in composition
        self.coefficients = []
        self.weights = [1. / float(len(in_features))] * len(in_features)

    def train(self, in_features, in_labels):
        self.reset(in_features)
        
        for iteration in xrange(self.composition_size):
            self.classifiers.append(self.learner.train(in_features, in_labels, weights=self.weights))
            # new classifier initially gets weight 1
            self.coefficients.append(1)
            answers = []
            for obj in in_features:
                answers.append(self.classifiers[-1].apply(obj))
            err = self.compute_weighted_error(in_labels, answers)
            if abs(err) < 1e-6:
	            return weighted_voting_adaboost(self.classifiers, self.coefficients)
            
            alpha = 0.5 * log((1.0 - err) / err)
            # updating the coefficient of the last added classifier
            self.coefficients[-1] = alpha
            
            self.update_weights(in_labels, answers, alpha)
            self.normalize_weights()
        return weighted_voting_adaboost(self.classifiers, self.coefficients)

    def compute_weighted_error(self, in_labels, in_answers):
        error = 0.
        w_sum = sum(self.weights)
        for ind in xrange(len(in_labels)):
            error += (in_answers[ind] != in_labels[ind]) * self.weights[ind] / w_sum
        return error

    def update_weights(self, in_labels, in_answers, in_alpha):
        for ind in xrange(len(in_labels)):
            self.weights[ind] *= exp(in_alpha * (in_answers[ind] != in_labels[ind]))

    def normalize_weights(self):
        w_sum = sum(self.weights)

        for ind in xrange(len(self.weights)):
            self.weights[ind] /= w_sum
