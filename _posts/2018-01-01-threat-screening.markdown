---
layout: post
title: Threat Screening Games and National Security
project: true
date: 2018-01-01 13:32:20 +0300
description: Game theory for preventing attack in secure areas and protecting critical infrastructure. # Add post description (optional)
img: airplane.jpg # Add image post (optional)
tags: [game-theory, threat-screening-game, airport-security]
---

The goal of this work is to develop efficient and effective game theoretic methods for preventing attacks in a secure area. While there are many different ways one may defend these areas, an effective way is to screen for threats (people, objects) before entry. This is a standard practice throughout the world, e.g., screening resources are used to secure border crossings, sports stadiums, government buildings, etc. In all of these scenarios, the screener is faced with the challenge of having a very large volume of items to be screened, and a very limited number of screening resources available to screen them with, making it critical for them to carefully plan the use of these limited resources in order to maximize the effectiveness of any defence strategy.

Threat Screening Games (TSG) are a game theoretic model, developed in collaboration with the Transportation Security Administration (TSA), which address these specific challenges by allowing the defender to remain unpredictable to a strategic adversary through randomized screening, allowing for more effective use of limited screening resources, leading to improved security. 

![screening]({{site.baseurl}}/assets/img/security.png)
<br><br>

However there are several challenges that are present in solving these types of games, namely:<br>

<h4> Dynamic Risk Management </h4>  
We want to comprehensively assess risk on an individual basis and design a dynamic screening strategy for different risk levels, traffic volumes, and resource availability.

<h4> Real World Uncertainty </h4>    
There is inherent uncertainty in the arrival times of the screenees. Addressing this challenge can be difficult as it requires reasoning about all the possible realizations of the uncertainty and coming up with an optimal plan for each of those scenarios. When dealing with a large number of screenees, this result in millions of possible scenarios, making the planning problem extremely difficult.

To address this shortcoming, I introduced a new model <i>Robust Threat Screening Games (RTSG)</i> (Mc Carthy 2017), which expresses the required uncertainty in screenee arrival times. In RTSG, we model the problem faced by a screener as a robust multistage optimization problem. We present a tractable solution approach with three key novelties that contribute to its efficiency: (i) compact linear decision rules; (ii) robust reformulation; and (iii) constraint randomization. We present extensive empirical results that show that our approach outperforms the original TSG methods that ignore uncertainty, and the exact solution methods that account for uncertainty.

<h4> Team Formation </h4>  
We have many different screening resources with differing efficacies, capacities and costs that may be combined to work in teams. The problem of Simultaneously Optimizing over Resource Team composition and Tactical deployment (SORT) (Mc Carthy 2016) is an important problem in many security game models, as solutions can be suboptimal if the defender does not strategically reason about how to form teams of resources. 

This class of problems, combines strategic planning which looks at the challenge of optimizing over <i>the configurations of resources</i> in each pure strategy, and tactical planning, which optimizes the <i>deployment</i> of these resources. In contrast with standard security games, where the defender has a fixed set of resources to be deployed, the SORT problem generalizes the security game problem by allowing the defender to additionally choose what <i>teams</i> of resources to form. In (Mc Carthy 17) we introduce the TSG-SORT model by extend the TSG model to allow the defender to perform this strategic planning over the teams of resources.

<h4> Operationalizable Plans </h4>
Optimal solutions to TSGs typically involve randomizing over a large number of pure strategies, each corresponding to a different security protocol. Thus, while they are by far preferable to deterministic strategies from an efficiency perspective, they are difficult to operationalize, requiring the security personnel to be familiar with numerous protocols in order to execute them.

To address this challenge, I developed a mixed-integer optimization model for computing strategies that are <i>easy to operationalize</i> and that bridge the gap between the suboptimal deterministic solution and the optimal yet impracticable mixed strategy. These strategies only randomize over an optimally chosen subset of pure strategies whose cardinality is selected by the defender, enabling them to conveniently tune the trade-off between ease of operationalization and efficiency using a single design parameter. The optimization formulation does not scale to realistic size instances and we propose a novel solution approach for computing &epsilon;-optimal equilibria as well as a heuristic for computing operationalizable strategies to TSG-SORT. We perform extensive numerical evaluation that showcases the solution quality and scalability of our approach and illustrate that the Price of Usability is typically not high.


<h3><br>Relevant Publications<br></h3>

<ul>
<li><strong>Sara Mc Carthy</strong>, Phebe Vayanos, Milind Tambe. <a href="https://doi.org/10.24963/ijcai.2017/527" target="https://doi.org/10.24963/ijcai.2017/527">Staying Ahead of the Game: Adaptive Robust Optimization for Dynamic Allocation of Threat Screening Resources</a>. In <em> Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI) 2017</em> 
</li> 
<li><strong>Sara Mc Carthy</strong>, Corine Lann, Kai Wang, Phebe Vayanos, Milind Tambe, Arunesh Sinha. <a href="" target="">The Price of Usability: Designing Operationalizable Strategies for Security Games</a>. In <em> submission </em> 
</li> 
</ul> 
