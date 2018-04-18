---
layout: post
title: Adversarial Ensemble Learning 
side-project: true
project: true
date: 2017-12-01 13:32:20 +0300
description: Looking at how to defend against adversarial examples by using an ensemble of deep learning models. 
The goal is to train an ensemble of models, each which use a different set of features to perform classification.
# Add post description (optional)
img: chess.jpg # Add image post (optional)
tags: [adversarial, machine-learning, deep-learning]
---


Many machine learning models are vulnerable to attacks known as adversarial examples, where small, 
strategic perturbations in inputs can cause erros in the output of the machine learning model (such as miss-classification).

In this side project, I looked at a method of protecting against adversarial examples by using a <i> ensemble </i> of machine learning models. The idea is to train an ensemble of models which learn to classify or predict based on a different subset of the feature space, so that while the small perturbations may cause errors in some of the machine learning ensemble, because each of the models is using a different set of features the perterbations will only affect a (hopefully) small subset of the ensemble so that overall output is still correct.

One of the challenges here is training a set of models to use different subsets of the feature space. To do this I used a technique introduced in (Ross et al. 2017) for efficiently explaining and regularizing differentiable models using an <i> Annotation Matrix </i>. By training a set of models with different annotation matricies, we can build an ensemble of models, each with a different decision boundary. 

[Github Project Page](https://github.com/saramccarthy/EnsembleLearning)
