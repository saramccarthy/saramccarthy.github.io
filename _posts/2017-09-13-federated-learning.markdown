---
layout: post
title: Private Collaborative Machine Learning
project: true
date: 2017-09-13 13:32:20 +0300
description: A project on differentially private distributed deep learning. 
img: mltree.jpg # Add image post (optional)
tags: [machine-learning, privacy]
---

<script type="text/javascript" async
src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<h4><b>Federated Learning</b></h4>

Recently Google introduced <i>Federated Learning</i> as a method for decentralized learning <span class="citation" data-cites="McMahan">(McMahan et al. 2016)</span>. Motivated by the wealth of information now stored on mobile devices, <i>Federated Learning allows</i> allows phones or other local devices to collaboratively learn a shared machine learning model <i>without their data ever leaving their device.</i><br><br>
The way that it works is through communication of model parameters from <i>clients</i> (ie. mobile devices) and a <i>central curator</i> (ie. Google). The central curator maintains a shared, central model. Each client downloads the current model and uses its own data to compute an update of the model parameters. The parameters of this updated model are then communicated to the central curator. The curator then updates the shared model by averaging the new model parameters communicated by each of the clients to improve the central shared model. <br><br>

While Federated Learning provides many advantages, such better latency, less power consumption and higher quality models, one significant benefit to such a decentralized model is that client data does not need to be shared, allowing them to keep potentially sensitive data <i> private</i>. <br><br>

However, although this algorithm does not require sharing private data, it does not provide any rigorous privacy measure, particularly against an adversary attempting to uncover a client's data.  Given knowledge of the parameter updates from each client, an intelligent adversary may be able to reverse engineer or gain information about the data used to generate that update. To attempt to address this issue, paper <span class="citation" data-cites="Bonawitz">(Bonawitz et al. 2016)</span>  suggests a secure aggregation policy to protect the privacy of each user's model gradient. This still doesn't provide a measurable level of privacy against arbitrary adversaries though. Recent work in differential privacy has focused on the "local model", where users privatize their own data before sending it to a central curator. This provides extra security by avoiding the need for a central curator to maintain and privatize everyone’s data. Since the Federated Learning model already has users providing their own deep learning model updates on-device, the local model of differential privacy is an excellent privacy model to use in order to bridge Federated Learning with differential privacy.

<h4><b>Differential Privacy</b></h4>
Differential privacy is a formal privacy definition which roughly defines an algorithm as “private” if it is not sensitive to any individual data point (or any small group of data points). 
<br><br>

If a machine learning model is trained in a differentially private manner on private data, we can guarantee that no individual’s private data will be at any significantly increased risk than if they had chosen not to provide their data in the first place. The challenge for designing differentially private algorithms is the inherent tradeoff between privacy and the algorithm’s resulting utility. Specifically, the more privacy the algorithm designer requires, the less useful the algorithm’s results will be. For instance, an algorithm could achieve perfect privacy by simply ignoring the private data input and outputting random values…but this clearly isn’t a useful algorithm. On the other hand, an algorithm that outputs the exact “correct” value every time is susceptible to leaking private data, whether directly or indirectly. Therefore, in practice, we seek differentially private algorithms that are both private and useful.<br><br>

For the case of deep learning, Google researchers published a method in late 2016 for training deep models in a differentially private manner and showed that their method still attainted respectable results on the MNIST and CIFAR-10 data sets <span class="citation" data-cites="Papernot">(Papernot et al. 2016)</span>. They achieved this by modifying the stochastic gradient descent (SGD) algorithm in a couple ways so that it achieved differential privacy; we’ll refer to this method as DP-SGD. They achieved DP-SGD with two primary modifications to SGD. The first change was to introduce gradient clipping to bound the amount that any single example could influence the training process. This has been done before, but not for privacy purposes. The second change was introducing carefully calibrated Gaussian noise to the computed gradient. This has the effect of masking the effect of any individual training data point on the computed gradient, while still allowing the aggregate effect of all training data points to shine through and compute a useful gradient.
<br><br>

While the researchers have published their DP-SGD code publicly, there are some setbacks to naively applying it to training deep networks. The main challenge that we found is that there is no support for training convolutional layers, making many deep learning tasks quite difficult to do from scratch. <span class="citation" data-cites="Papernot">(Papernot et al. 2016)</span> worked around this problem in one way to train their MNIST network, and another way to train their CIFAR-10 network. For MNIST, the researchers used a simple 2 layer fully connected neural network, but added a preprocessing step of using the training data to do PCA. Since traditional PCA provides no privacy guarantees, the researchers had to (and did) use a differentially private PCA algorithm in order to satisfy an overall guarantee of differential privacy. Note briefly that this is not directly applicable in the Federated Learning framework, since it would require each client to compute its own PCA without the aid of a centralized party. For CIFAR-10, the researchers used a network with 2 initial convolutional layers followed by 2 fully connected layers. Since they couldn’t train the convolutional layers, they set those layers as non-trainable in Tensorflow and then loaded in a fully pretrained network from CIFAR-100. On CIFAR-10 with this setup (where only the final 2 fully connected layers are being trained), they were able to achieve decent results. The other challenge was simply the training computation: both the speed of computing each gradient descent step as well as the overall working memory required increased by nearly an order of magnitude when compared to Tensorflow’s non-differentially private SGD implementation.

<h4><b>Optimized Averaging</b></h4>
In <span class="citation" data-cites="McMahan">(McMahan et al. 2016)</span>, the introduced federated learning algorithm uses a naive average merging procedure to update the central model parameters. In particular, the weights of the central model are set to be the average of the weights received from the clients.  Intuitively, in a federated learning scheme, different clients that are learning from different datasets will not have the same contribution when updating the central model. Thus, one would expect enhanced results with smarter merging procedures. In a recent ongoing research, <span class="citation" data-cites="Izbicki">(M. Izbicki 2017)</span> proposed a method called {\it optimal weighted average} (OWA), which computes the optimal average weights for merging client parameters. <br><br>

Consider the following multiple layer feedforward neural network optimization problem:
<div>
<p>
\[(1) \;
\text{minimize}_{\boldsymbol{W}^{c}} \, \, \ell(\mathcal{F}_h(\boldsymbol{W}^{c}),\boldsymbol{Y}^{c}) \quad \mbox{with} \quad \begin{cases} \mathcal{F}_1(\boldsymbol{W}^{c}) \triangleq \sigma_1({\boldsymbol{W}_1^{c}}\boldsymbol{X}^{c})\\
\mathcal{F}_k(\boldsymbol{W}^{c}) \triangleq \sigma_k \big({\boldsymbol{W}_{k}^{c}}\mathcal{F}_{k-1}(\boldsymbol{W}^{c})\big), \, k \in [2,h]\end{cases} i\]
\[(1)
\displaystyle{{\text{minimize}}_{\boldsymbol{W}^{c}}} \, \, \ell(\mathcal{F}_h(\boldsymbol{W}^{c}),\boldsymbol{Y}^{c}) \quad \mbox{with} \quad \begin{cases} \mathcal{F}_1(\boldsymbol{W}^{c}) \triangleq \sigma_1({\boldsymbol{W}_1^{c}}\boldsymbol{X}^{c})\\
\mathcal{F}_k(\boldsymbol{W}^{c}) \triangleq \sigma_k \big({\boldsymbol{W}_{k}^{c}}\mathcal{F}_{k-1}(\boldsymbol{W}^{c})\big), \, k \in [2,h]\end{cases}\]

where \( \sigma_k (\cdot),\, k = 1,\ldots h \), 
are the activation functions for different layers, 
\(\boldsymbol{W}=\big( {\boldsymbol{W}}_i \big)_{i=1}^h\), 
\({\boldsymbol{W}}_i \in \mathbb{R}^{d_i \times d_{i-1}}\)
are the weight matrices, 
\(\boldsymbol{X}^{c} \in \mathbb{R}^{d_0 \times n}\)
is the input training data for client \(c\), \(\mathcal{F}_h()\) gives the output predictions of the model, \(\boldsymbol{Y}^{c} \in \mathbb{R}^{d_h \times n}\) is the target training data for client \(c\) with loss function \(\ell()\). Each client \(c\) solves problem (1) and return the parameters \(\big(\boldsymbol{W}^{c}\big)_{i=1}^h\) to the central model. In <span class="citation" data-cites="McMahan">(McMahan et al. 2016)</span>, the naive averaging merge procedure 
\(\boldsymbol{W}_i = \sum_{c \in {\cal C}} \frac{\boldsymbol{W}_i^c}{|\mathcal C|}\)
where \({\cal C}\) is the set of clients, was used to update the central parameters. An improved averaging procedure was introduced by <span class="citation" data-cites="Izbicki">(M. Izbicki 2017)</span> which involves solving the following optimization problem:

\[\label{OWA} (2)
\displaystyle{ {\mbox{minimize}}_{\boldsymbol{v}}} \, \, \ell(\mathcal{F}_h(\boldsymbol{v}),\boldsymbol{Y}) \quad \mbox{with} \quad \begin{cases} \mathcal{F}_1(\boldsymbol{v}_1) \triangleq \sigma_1({\boldsymbol{v}_1\boldsymbol{W}_1}\boldsymbol{X})\\
\mathcal{F}_k(\boldsymbol{v}_k) \triangleq \sigma_k \big(\boldsymbol{v}_k{\boldsymbol{W}_{k}}\mathcal{F}_{k-1}(\boldsymbol{W})\big), \mbox{  for } k \in [2,h],\end{cases}
\]

where \(\boldsymbol{v} = \big(\boldsymbol{v}_i)_{i=1}^h\), \(\boldsymbol{v}_i\) are the optimal averaging weights for layer \(i)\), \(\boldsymbol{W_i} \) is a third order tensor constructed by stacking \(\boldsymbol{W}_i^c \) for all \() \in {\cal C}\), and \(\boldsymbol{X}\) and \(\boldsymbol{Y}\) are subsets of the input and target training data for clients respectively.<br><br>

It follows from (2), that OWA merging procedure uses the same model learned by the clients, but replaces, at each layer, the weight matrices \(\boldsymbol{W}_i^c\) by a weighted average \(\boldsymbol{v}_i\boldsymbol{W}_i\), and optimizes over \(\boldsymbol{v}_i\). This model is trained by \textit{central curator}, and \(\boldsymbol{v}_i^{*}\boldsymbol{W}_i\), with \(\boldsymbol{v}_i^{*}\) being the optimal client weight for layer \(i\), will be used to update the shared model.<br><br>
</p>
</div>
As mentioned above the reduction of privacy risk is one of the main advantages of <i>Federated Learning</i>. This is achieved by waving the need to send user data to the <i>central curator</i>. One drawback of the OWA model is that it requires sending data form the clients in order to train the client weights. However, since the number of weights that need to be optimized is fairly small (Number of clients x Number of layers), we only require a small subset of the data for training. Thus implementing the OWA model imposes a trade-off between privacy and the algorithm's resulting accuracy. In particular, with a larger data subset sent to train the OWA model, we can achieve a better testing accuracy, but the algorithm will become less private. For instance, if no data is sent to the OWA model, the model will not lose any privacy, but might not achieve a high accuracy. On the other hand, by sending all client datasets to train the OWA model, we sacrifice the privacy benefit gained by the Federated Learning algorithm to get a higher accuracy. In practice, we suggest sending a small subset of data, to gain accuracy without losing too much privacy. <br>

An experiment that trains the default tensorflow convolutional neural network (CNN) on MNIST data was performed in <span class="citation" data-cites="Izbicki">(M. Izbicki 2017)</span>. The MNIST was partioned into IID subsets each with 429 to 430 data points, and an 8-layer CNN was trained using Adam Optimizer and dropout. The results show that the accuracy increases by around 1% when the number of clients is 10. In this project, we combined this recent merging procedure with federated learning scheme and showed up to 15% improvement in test accuracy with our model.<br><br>

<h4><b>Our Model & Experimental Results</b></h4>
While existing techniques like DP-SGD provide rigorous guaruntees of privacy, they often come at a cost of loss in a model's utility (usually measured by the accuracy of their outputs). We thus improve on the federated averaging algorithm proposed in <span class="citation" data-cites="McMahan">(McMahan et al. 2016)</span> by not only providing privacy guaruntees by using DP-SGD in place of standard SGD, but by also improving the central model update by optimizing over the aggregation and averaging of the client parameters. This allows us to mitigate the tradeoff between loss in quality and increased privacy more effectively than previous work. <br><br>

Our model, which we refer to as <i> Optimized Differentially Private Federated Learning </i> (ODPFL) is built in the following way. 
<ul>
<li> Each client trains their own copy of a central model. All clients share the same model architecture.
<li> After a certain number of training rounds, clients send their updated model parameters to the central curator.
<li> We assume that there exists a small public dataset that is available to the central curator, which they then use to solve the optimization problem (2) and solve for the optmal weights for each of the client parameters. 
<li> The central model is then updated by taking a weighted average of the parameters, according to the optimized weights.
<li> Clients can then download this central model and the process repeats.
</ul>
In order to test the ODPFL model we had each client train a model using the achitecture from the <a href="https://www.tensorflow.org/tutorials/deep_cnn">TensorFlow Cifar10 tutorial</a> and using the Cifar10 dataset. We chose this particular model architecture as it was the same architecture used in the original Federated Learning paper, which allowed us to compare the results of our baseline model (without differential privacy or optmized weights) to their model to ensure that we were training all the clients properly.

This model architecture is 5-layer network, consisting of 2 convolutional layers, two
fully connected layers and then a linear transformation layer to produce logits. 

<h5> Implementing Federated Learning </h5>
Since no code was provided by the original paper, we first implemented our own version of the Federated Learning algorithm using this model. This required (1) partitioning the data set for each of the clients, (2) building and training each of the client models, (3) retreiving the learning model parameters for each of the clients, (4)computing the updated weights, (5) storing the weights in a central model file (6) and finally having the clients read in new parameters from that model file upon their next training session. <br><br>

For the data partitioning we considered two potential scenarios: IID distribution of data among cliends and non-IID distribution. We believe the later scheme is more practical and realistic as in practice, different clients may have a widely varying data, and this setting captures these different distributions of different clients. For testing the performance of our federated learning implementation used 10 clients and partitioned the data into 20 data sets: 10 iid-distributed sets and 10 non-iid distributed sets each containing 10% of the full cifar10 dataset. For the iid partitioning for each client we sampled 10% of the dataset with replacement, and for the non-idd partitioning each client sampled 10% of the dataset without replacement. We ran the federated learning algorithm for 130 and 200 communication rounds for both the iid and non-iidrespectively ie. number of rounds of client training and central model updating) and found that we were able to acheive ~80% test accuracy on the full cifar10 test set for the idd partitioning and ~70% test accuracy for the non-idd partitioning as shown in Figure 1 and Figure 2. These results are comparable to the original federated learning paper, who report a 85% but only tested the iid data partitioning, and report training accuary for each of the clients test sets (potentially only a subset of the full test set, since partitioning was done iid), and after 2000 communication rounds. <br><br>

<h5> Optimizing Client Weights </h5>

To improve the accuracy of the models trained with our Federated Learning, we then built a separate model to optimize over the weights assigned to the parameters of each of the clients. Because the optimization problem (2) is complex and non-linear, we use a neural network to solve for these weights. The architecture of this model is similar in structure to those of the clients, but for each trainable tensor \(\boldsymbol{W}_i^c\) in the client model we replace these variables by a weighted average \(\boldsymbol{v}_i\boldsymbol{W}_i\) where the \(\boldsymbol{v}\) are the new trainable variables. As a result, this model has much few trainable parameters than the original client models which means that they (1) train very quickly and (2) do no require a lot of data to train. <br><br>

In contrast to <span class="citation" data-cites="Izbicki">(M. Izbicki 2017)</span>, who imposed no constraints on the set of allowed weights, we found that the best results were acheived when we added normalization constraints to weights on the client parameters, so that \(\sum_{c\in \mathcal C} \boldsymbol{v^c_i} = 1 \,\, \forall i \in \{1,h\} \). Practically, these constraints were imposed by training a set of auxillary variables \(x_i^c\) so that \(\boldsymbol{v_i^c} = \frac{x_i^c}{\sum_c x_i^c} \).

To test the Federated Learning model with optimized weights we modified the partitioning of the data slightly so that we reserved 5% of the data to be used to train the optimization model. The remaining portion of the cifar dataset was partitioned in an iid and non-iid fashion as before. In the iid setting the optimization of the weights did not provide a large benefit. We see a slight improvement in accuracy and in speed of convergence, but the standard averaging is able to match the optimized averaging quite closely. We believe that this is due to the nature of the iid distributed data; Since each client model is trained on similar distributions of data, at the end of each training session, all of the models are likely to have learned similar parameters, and thus an equal weight averaging of the parameters is likely to be close to optimal. However, we see significant improvements in the setting of non-iid distributed data. In this case we see we acheive much heigher training accuracy, reaching up to 80% accuracy with optimized averaging compared to 65% without the optimization, which is approximately a 23% improvement in accuracy. Not only that, but we also see that we get much smoother convergence with the optimized weights, while the standard federated learning accuracy is still very noisy even after 200 communication rounds. <br><br>
<figure>
<img src="iid.png" alt="iid-graphs" width="100%">
<figcaption>Figure 1: Test Accuracy and Loss of optimized weighted averaging (owa) 
compared to standard equal weighted averaging (average) 
for iid-data partitioning.</figcaption>
</figure>
<figure>
<img src="niid.png" alt="niid-graphs" width="100%">
<figcaption>Figure 2: Test Accuracy and Loss of optimized weighted averaging (owa) 
compared to standard equal weighted averaging (average) 
for non-iid-data partitioning.</figcaption>
</figure>
These results are significant, as they not only demonstrate the benefit of optimized averaging on networks much more complex than in <span class="citation" data-cites="Izbicki">(M. Izbicki 2017)</span> but we also show significantly larger improvements with this method (23% improvement here versus 2% improvement in their setting). </p>
<h5><b> Differentially Private Learning </b></h5>
As mentioned previously it is currently computationally infeasible to provide differential privacy guaruntees for training convolutional layers. What this means is that in order to use DP-SGD for our Cifar10 network architecture we cannot have trainable convolutional layers. To get around this and still provide privacy guaruntees for our model we tried three different techniques for training the network privately:
<ul>
<li> Using the same model architecture, but removing the conv layers completely</li>
<li> Using non-trainable (static) randomly initialised conv layers </li>
<li> Using non-trainable (static) pretrained conv layers, trained on the Cifar100 dataset </li>
</ul>
We found with the first two techniques we were not able to obtain good results, with accuracies in the range of only 10-20%. For the third method we trained a model with the same architecture as our own on the Cifar100 data set, up to an accuracy of 40%. However when we tried to apply this model to out federated learning framework we were only able to acheive around 30% test accuracy. We believe that it is due to the significantly greater amound of training that is required to train models with DP-SGD. In original DP-SGD paper required sevearal hundred epochs of training with batch sizes of around 2000 to 4000 for the Cifar10 dataset to acheive good results. As such it still remains a challenge to properly integrate Differentially Private learning into this distributed learning framework.

<h4><b>Future Work</b></h4>
This course project is now the basis for a much larger scope project. Particularly, since we have seen that it is possible to improve utility with the OWA method in non-iid settings, and since we know that it is possible to learn models differentially privately (albeit, with poor utility), we are interested in combining these two methods. First though, we must overcome this utility issue in the straightforward differentially private learning setting. As mentioned before, there are several challenges here. One main computational issue is that the current implementation of DPSGD only supports CPU computation, resulting in nearly untrainably-slow models. Moreover, it doesn't appear to have been engineered in a memory-performant manner either. Fixing these issues is likely primarily a software engineering challenge rather than a fundamental research problem, but it would need to be done in order to move forward with the research goals.
<br><br>
The other main issue with DPSGD currently is that there is no support for training convolutional layers, which (in the case of the publication) meant pre-training them non-privately on a different data set. But, even when we did pre-training for the Cifar10 model using the Cifar100 dataset, the results were still quite poor relative to what you get in the non-private learning setting. Thus, we did not find this solution satisfactory for this reason as well as the fact that we'd prefer to keep the problem as contained as possible (and not require an auxiliary data set). Therefore, figuring out how to train convolutional layers efficiently with differential privacy is another major challenge in making this concept useful.
<br><br>
Once these two challenges are addressed and a model can be trained differentially privately in this traditional setting, only then can we move on to combining differential privacy and federated learning. Here, we hope that the resulting loss in utility will be minimal when  moving simultaneously away from the centralized model to the federated learning model as well as from the non-private model to the differentially private model. We hope that any loss in utility will be made up for by the introduction of the OWA algorithm into the federated learning framework.
<br><br>
Finally, if this is accomplished, we further hope to move away from differential privacy's traditional homogeneous privacy guarantee for all users to a heterogeneous per-user privacy guarantee (selectable by each user). This would be useful in scenarios where users want some control over the data, allowing them to be in control of the trade-off between their own privacy and the model's resulting utility.   


<h3><b>References</b></h3>
<div id="ref-Abadi">
<p>Abadi, Martín, Andy Chu, Ian Goodfellow, H Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. 2016. “Deep Learning with Differential Privacy.” In <em>Proceedings of the 2016 Acm Sigsac Conference on Computer and Communications Security</em>, 308–18. ACM.</p>
</div>
<div id="ref-Bonawitz">
<p>Bonawitz, Keith, Vladimir Ivanov, Ben Kreuter, Antonio Marcedone, H Brendan McMahan, Sarvar Patel, Daniel Ramage, Aaron Segal, and Karn Seth. 2016. “Practical Secure Aggregation for Federated Learning on User-Held Data.” <em>arXiv Preprint arXiv:1611.04482</em>.</p>
</div>
<div id="ref-Konecny">
<p>Konečny, Jakub, H Brendan McMahan, Felix X Yu, Peter Richtárik, Ananda Theertha Suresh, and Dave Bacon. 2016. “Federated Learning: Strategies for Improving Communication Efficiency.” <em>arXiv Preprint arXiv:1610.05492</em>.</p>
</div>
<div id="ref-Izbicki">
<p>M. Izbicki, C. R. Shelton. 2017. “Merging Neural Networks.” In <em>SOCAL Machine Learning Symposium, University of Southern California (Not yet Published</em>.</p>
</div>
<div id="ref-McMahan">
<p>McMahan, H Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and others. 2016. “Communication-Efficient Learning of Deep Networks from Decentralized Data.” <em>arXiv Preprint arXiv:1602.05629</em>.</p>
</div>
<div id="ref-McMahanNew">
<p>McMahan, H Brendan, Daniel Ramage, Kunal Talwar, and Li Zhang. 2017. “Learning Differentially Private Language Models Without Losing Accuracy.” <em>arXiv Preprint arXiv:1710.06963</em>.</p>
</div>
<div id="ref-Alexandroff">
<p>Papernot, Nicolas, Martín Abadi, Úlfar Erlingsson, Ian Goodfellow, and Kunal Talwar. 2016. “Semi-Supervised Knowledge Transfer for Deep Learning from Private Training Data.” <em>arXiv Preprint arXiv:1610.05755</em>.</p>
</div>

