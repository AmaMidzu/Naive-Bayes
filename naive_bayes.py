# -*- mode: Python; coding: utf-8 -*-

import math
from classifier import Classifier

class NaiveBayes(Classifier):
    u"""A naïve Bayes classifier."""
    def __init__(self, model=None):
        super(Classifier, self).__init__()
        self.seen = {}
        self.labels = set()
        self.V = set()
        self.priors_of_classes = {}

    def get_model(self): return self.seen
    def set_model(self, model): self.seen = model
    model = property(get_model, set_model)

    def train(self, instances):
        """Remember the labels associated with the features of instances."""
        # Build vocab and labels sets
        for instance in instances:
            self.labels.add(instance.label)
            for w in instance.features():
                self.V.add(w)
        if '' in self.labels:
            self.labels.remove('')
        # Count total number of docs
        N = len(instances)

        Nclass = {}
        word_counts_per_class = {}
        # Prepare counts of words per class as dict of dicts
        for label in self.labels:
            word_counts_per_class[label] = {}
            self.seen[label] = {}

        # Count docs per class
        for instance in instances:
            Nclass[instance.label] = Nclass.get(instance.label, 0) + 1
        # Compute priors for each class
        for label in self.labels:
            self.priors_of_classes[label] = float(Nclass[label])/N

        # Get counts for each term for each class
        for instance in instances:
            for w in instance.features():
                word_counts_per_class[instance.label][w] = word_counts_per_class[instance.label].get(w, 0) + 1
        # Smoothing
        for label in self.labels:
            for w in self.V:
                word_counts_per_class[label][w] = word_counts_per_class[label].get(w, 0) + 1

        # Compute probs
        for label in self.labels:
            for w in self.V:
                self.seen[label][w] = float(word_counts_per_class[label][w])/sum(word_counts_per_class[label].values())

    def classify(self, instance):
        """Classify an instance using the features seen during training."""
        V_doc = set(instance.features())
        score = {}
        for label in self.labels:
            score[label] = math.log(self.priors_of_classes[label])
            for w in V_doc:
                try:
                    score[label] += math.log(self.seen[label][w])
                except:
                    print "Feature not found"
        return sorted(score, key=score.get, reverse=True)[0]
