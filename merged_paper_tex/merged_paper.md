# 摘要

摘要：
随着无线通信与雷达探测技术的飞速发展，现代工程对天线的综合性能提出了更高要求。传统基于数值全波电磁仿真的天线设计方法面临计算成本高、寻优周期长以及易陷入局部最优等瓶颈。为突破上述局限，探讨人工智能（AI）在天线设计领域的应用潜力与演进趋势具有重要意义。本文对AI辅助天线设计的相关研究进行了全面综述。首先，系统梳理了人工神经网络（ANN）、深度神经网络（DNN）以及图神经网络（GNN）在天线性能预测、复杂结构表征与阵列综合中的应用现状；其次，探讨了元启发式算法与深度学习代理模型协同优化技术的发展历程。研究发现，AI技术显著提升了复杂天线设计的非线性建模能力与多目标优化效率，但在高质量数据依赖、跨结构泛化能力、物理可解释性及工程可制造性验证方面仍存在明显局限。同时，本综述指出了当前领域在标准化评价体系、物理机理深度融合及多物理场协同等方面的研究空白。基于上述分析，本文提出未来AI天线设计应彻底告别碎片化的“一任务一模型”专用范式，加速向着以大规模电磁数据预训练为核心的“天线基础模型”演进，以实现跨结构与跨频段的知识迁移；同时亟需推动图深度学习向“全域电磁拓扑”延伸，并依托大语言模型智能体与主动学习机制构建打通“需求-生成-仿真-反馈”的智能闭环设计平台。本综述为全面实现天线设计的通用化、全域化与全流程自动化落地提供了理论参考与创新思路。

关键词：
AI辅助设计、深度学习代理模型、图神经网络、天线基础模型、多目标优化

# 引言

随着无线通信、雷达探测、物联网以及 5G/6G 通信技术的高速发展，现代工程应用对天线综合性能提出了愈发严苛的要求，多频段、超宽带、高增益、低旁瓣、小型化等指标已成为天线设计中的主流需求 [1]。现阶段，HFSS、CST 等基于数值分析方法的电磁仿真软件被广泛应用于天线设计与优化过程，是开展电磁分析不可或缺的重要工具。然而，此类仿真软件在实际应用中仍存在一定局限：一方面，全波电磁求解计算量庞大，对硬件资源要求较高；另一方面，在涉及多结构参数批量扫描与优化的场景中，计算耗时较长，设计效率较低 [2]。同时，传统天线设计流程高度依赖工程师的电磁理论基础与工程实践经验，参数调试过程繁琐且重复性强。综上所述，传统天线设计优化模式普遍存在流程复杂、迭代效率低等问题，并且容易出现优化结果陷入局部最优、综合性能难以达到设计预期等情况 [2]。

为缓解传统天线设计中计算成本高、寻优迭代周期长等问题，近年来，随着人工智能技术的快速发展，机器学习与深度学习算法逐步赋能电磁领域，并已在电磁散射、雷达信号处理、电磁兼容等研究方向取得了丰富成果 [3]。由于天线结构参数与电磁性能指标之间通常具有多变量、多极值、强非线性的耦合特征，深度学习凭借其优异的非线性拟合能力与特征表征能力，在天线优化设计中展现出良好的应用潜力 [3,4]。该类方法能够通过挖掘天线“结构—性能”之间的内在映射关系构建代理模型，不仅可以实现对未知天线结构电磁性能的快速预测，还可依据既定性能指标反向推演最优结构参数，从而完成逆向寻优设计 [2-4]。

早在 1998 年，Mishra R. K. 等人便率先将人工神经网络应用于方形贴片天线的性能预测研究 [5]。相关研究表明，该模型能够在保证一定计算精度的基础上显著缩短仿真耗时，初步验证了人工智能技术应用于天线设计领域的可行性。发展至今，多层感知机、卷积神经网络、径向基函数网络以及克里金插值等方法已被广泛用于构建电磁代理模型 [2-4]。现有研究充分表明，将机器学习算法引入天线设计流程，能够有效缓解传统仿真方法计算成本高、资源消耗大的问题，并显著提升设计与优化效率 [2,3]。

在天线阵列分析与综合方向，传统人工神经网络与深度神经网络已形成较为成熟的应用体系 [3,6]。此类网络能够较好地拟合阵列结构参数、激励参数与辐射方向图之间复杂的非线性映射关系，从而显著提升阵列综合的计算效率 [3,6]。然而，ANN、DNN 等传统深度模型通常只能处理固定维度的欧式空间张量数据，其特征提取方式存在一定局限。这一特性使得传统网络更适用于规则阵列或单一拓扑结构阵列；而对于非均匀稀疏阵列、复杂共形阵列等具有多样化拓扑结构的阵列形式，模型的建模能力相对不足，难以实现高精度的全局拟合分析 [6,7]。

近年来，图神经网络凭借其对非欧式空间图结构数据的强大建模能力，在多个学术与工程领域得到广泛应用 [7]。目前，GNN 已成熟应用于社交推荐、电商分析、交通预测、生物制药等多个方向。例如，在交通流量预测任务中，图神经网络可将时间、地理位置、气象条件等多模态异构信息抽象为图节点与拓扑边，从而实现高精度的时空流量推演，为复杂关联数据的建模分析提供了新的思路 [7,8]。

由此可见，图神经网络的引入反映了 AI 辅助天线设计方法从固定参数建模向结构化表示学习发展的趋势。传统 ANN、DNN 通常依赖固定长度的参数输入，适合处理结构形式相对确定的天线模型；而面对非均匀阵列、稀疏阵列、共形阵列等复杂拓扑结构时，阵元之间的空间关系和互耦作用难以通过简单参数向量充分表达。相比之下，GNN 能够将阵元、馈电关系和电磁耦合关系统一建模为图结构，从而增强模型对不同阵列规模和拓扑形式的适应能力。更重要的是，这一思路表明，未来天线智能设计不应局限于针对单一结构或单一任务训练专用模型，而应进一步探索能够融合几何结构、电磁响应、材料属性、方向图和设计约束的通用表征方法。在此基础上，结合大规模预训练、物理约束学习和多任务迁移机制，构建面向天线设计的基础模型，将可能成为提升 AI 天线设计泛化能力和自动化水平的重要方向。

参考文献

[1] Balanis, C. A. Antenna Theory: Analysis and Design. 4th ed. Wiley, 2016.

[2] Koziel, S., & Ogurtsov, S. Antenna Design by Simulation-Driven Optimization. Springer, 2014.

[3] Zhang, Q. J., Gupta, K. C., & Devabhaktuni, V. K. “Artificial neural networks for RF and microwave design—From theory to practice.” IEEE Transactions on Microwave Theory and Techniques, 2003.

[4] LeCun, Y., Bengio, Y., & Hinton, G. “Deep learning.” Nature, 2015.

[5] Mishra, R. K., & Patnaik, A. “Neural network-based CAD model for the design of square-patch antennas.” IEEE Transactions on Antennas and Propagation, 1998, 46(12): 1890-1891.

[6] Haupt, R. L. Antenna Arrays: A Computational Approach. Wiley, 2010.

[7] Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., & Philip, S. Y. “A comprehensive survey on graph neural networks.” IEEE Transactions on Neural Networks and Learning Systems, 2021.

[8] Yu, B., Yin, H., & Zhu, Z. “Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting.” International Joint Conference on Artificial Intelligence, 2018.

# 文献方法综述

# 2 文献方法综述

## 2.1 基于深度学习的天线设计研究现状

传统的天线设计和优化多依赖于工程师的理论基础和工程经验，通常需要使用电磁仿真软件通过等步长扫参方式进行多次调试，优化结果难以保证全局最优，并且设计过程复杂、耗时较长。随着人工智能技术的迅速发展，深度学习方法逐渐被引入天线领域。该类方法通过学习天线结构参数与性能指标之间的映射关系，在一定程度上弥补了传统电磁仿真软件计算成本高、设计周期长等不足，为天线设计与优化提供了新的技术路径。

目前，人工神经网络（Artificial Neural Network, ANN）和深度神经网络（Deep Neural Network, DNN）是在天线领域应用较多且较为成熟的网络模型，主要研究方向包括天线结构设计、性能预测以及天线阵列综合等。在天线结构设计和优化方面，相关研究多采用 ANN 进行训练和预测。例如，Mishra R K 等人较早将人工神经网络应用于方形贴片天线设计，验证了人工智能方法用于天线设计优化的可行性[1]。随后，为提高模型精度和训练效率，许多研究在基础 ANN 模型上进行改进，构建了更适用于天线设计任务的网络模型。

在具体应用方面，近年来的研究更加关注低样本代理建模、复杂结构快速生成和多目标自动设计。例如，Mahouti 等人针对透射阵天线提出低成本代理建模方法，利用自动化深度学习构建单元模型，在较少训练样本条件下完成 8-14 GHz、22-28 GHz 和 28-36 GHz 三个频段透射阵设计，并通过实验进行了验证[2]。Wei 等人提出 CNN-LSTM 结合的天线结构自动建模方法，仅以论文或图像中的天线结构图作为输入，便可自动生成对应的建模代码，从而加快不同天线物理结构样本的数据获取过程[3]。另有研究面向任意形状多边形贴片天线，采用 CNN 代理模型结合多岛差分进化算法进行多目标逆向设计，在数分钟内完成对反射系数、输入阻抗和辐射方向图等多个目标的综合优化[4]。

除单一天线结构设计外，深度学习方法也被应用于天线阵列的设计和综合。Ayestaran R G 等人使用神经网络进行非均匀天线阵列合成，将天线阵列的辐射模式与各阵元激励电压相关联，在不对阵列几何形状作过多假设的情况下实现阵列综合，并可在考虑耦合效应时处理非均匀阵列问题[5]。Mishra Subhash 等人利用 RBF 神经网络对共线短偶极子和平行短偶极子的均匀线性阵列进行方向性估计，该方法能够适应天线阵列的非线性特性，计算过程相对简洁，具有较好的鲁棒性[6]。

随着网络结构的不断发展，深度学习代理模型不再局限于简单的 S 参数预测，而是逐渐与多目标优化、结构生成和数据自动获取结合。相关研究表明，深度网络可作为快速预测器嵌入进化算法中，用于处理具有大量设计变量和多个性能指标的现代天线综合问题；同时，CNN、LSTM 等网络也可服务于训练样本自动生成，缓解传统手工建模效率较低的问题。这说明深度学习在天线设计中的作用正在从“性能预测工具”扩展为“建模、预测和优化协同工具”。

总体来看，深度学习方法在天线设计与优化中具有较强的非线性建模能力，能够在一定程度上替代重复的电磁仿真计算，提高天线结构设计、性能预测和阵列综合效率。然而，传统 ANN 和 DNN 主要适用于规则欧式数据或固定拓扑结构，对于非均匀阵列、复杂拓扑阵列等非欧式结构数据的表达能力仍存在一定局限，这也推动了图深度学习方法在天线设计领域中的进一步应用。

## 2.2 基于图深度学习的天线设计研究现状

为更好地学习非欧式空间数据的特征，利用深度学习技术实现图数据的端到端建模已成为重要研究方向。图神经网络（Graph Neural Network, GNN）能够处理由节点和边构成的图结构数据，适合描述复杂对象之间的连接关系和相互作用。随着图卷积、图注意力等方法的发展，越来越多基于图的神经网络被应用于社交网络、交通预测、生物制药以及电路和电磁建模等领域。

在天线和阵列问题中，阵元、用户、馈电点或可移动天线位置之间往往存在天然的图关系，因此可用图神经网络描述节点之间的连接、距离和耦合关系。近年来，GNN 已从早期的电路和阵列性能预测逐渐扩展到稀疏阵列测向、流体天线系统和新型可重构天线系统的联合优化中。

例如，Yang 等人针对随机稀疏线阵的波达方向估计问题，提出基于 GraphSAGE 的残差图卷积网络，将稀疏阵列接收信号构造为图结构，通过邻居节点聚合与更新补偿缺失阵元信息。该方法在低信噪比、低快拍数和大稀疏度条件下仍表现出较好的鲁棒性，在特定条件下相较 MUSIC 算法取得了明显性能提升[7]。

此外，GNN 也开始用于新型可重构天线系统的联合设计。He 等人面向流体天线系统提出两阶段 GNN 框架，分别推断天线位置和波束形成向量，并通过无监督损失函数联合训练，以实现多用户系统中的和速率与能效优化[8]。Xie 等人则针对 Pinching 天线系统，将下行通信系统建模为二部图，提出基于图注意力网络的 BGAT 模型，用于联合优化天线放置和功率分配，并强调了模型在最优性、可扩展性和计算效率方面的优势[9]。

综上，图神经网络为天线阵列建模提供了一种新的结构化表达方式。与传统 ANN 和 DNN 相比，GNN 能够将阵元、阵元间距、耦合关系和拓扑连接纳入统一模型中，突破了规则阵列或固定拓扑结构的限制。因此，将图深度学习应用于天线阵列分析、综合和优化具有较好的理论基础和发展潜力。

## 2.3 基于元启发式算法的天线设计研究现状

深度学习模型和机器学习模型在天线设计中多用作代理模型，其作用是替代传统全波仿真方法，对天线性能或阵列方向图进行快速预测；而元启发式算法则更多用于天线或阵列参数的反向寻优。元启发式算法通常通过模拟自然界或群体行为中的搜索机制进行优化，适合求解高度非线性、多变量和多目标问题，因此在天线设计优化中得到广泛应用。

遗传算法（Genetic Algorithm, GA）和粒子群优化算法（Particle Swarm Optimization, PSO）是较早应用于天线设计的典型元启发式算法，并且发展较为成熟。近年来，研究重点逐渐从单独使用传统优化算法转向“代理模型 + 智能优化算法”的协同框架。例如，Liu 等人提出的多边形贴片天线多目标逆向设计方法采用 CNN 作为快速代理预测模型，并结合多岛差分进化算法完成多目标搜索，使任意形状贴片天线能够在较短时间内完成综合优化[4]。

随着天线结构和优化目标日益复杂，更多新型元启发式算法被引入天线代理建模和参数优化中。Huang 等人研究了元启发式智能算法优化神经网络权值和偏置的天线建模方法，引入海鸥优化算法、融合反向学习的改进蝴蝶算法和人工兔优化算法等，对神经网络进行优化。实验结果显示，融合反向学习的改进蝴蝶算法优化后的神经网络在预测精度和运行效率方面表现更优，说明新型群智能算法可用于提升天线代理模型的建模精度[10]。

此外，为克服高保真电磁仿真计算成本高的问题，Koziel 等人在 2023 年研究了面向天线优化的变保真元启发式优化流程，通过在不同分辨率电磁模型之间切换来降低自然启发式优化过程中的计算开销[11]。2025 年的相关研究进一步将 1D GP-HetCNN 代理模型与 WOA、COA、GWO、PSO 和贝叶斯优化等方法结合，用于超宽带天线参数优化；在统一代理模型评估预算下，不同智能算法通过调用训练好的代理模型搜索更优几何参数，仅需少量 HFSS 验证即可完成优化流程[12]。这类研究表明，智能优化算法正在与轻量化代理模型、多保真模型和深度学习模型更紧密地结合。

总体而言，元启发式算法已在天线结构设计、阵列综合、带宽展宽、旁瓣抑制和多目标优化等任务中得到广泛应用。其优势在于不依赖梯度信息，适合处理复杂非线性问题；不足在于计算过程中可能需要大量性能评估，且部分算法存在局部最优和收敛速度不稳定等问题。因此，如何将元启发式算法与机器学习代理模型、深度学习预测模型相结合，以进一步提高优化效率和设计精度，是当前 AI 天线优化设计中的重要研究方向。

## 本章参考文献

[1] Mishra R K, Patnaik A. Neural network-based CAD model for the design of square-patch antennas[J]. IEEE Transactions on Antennas and Propagation, 1998, 46(12): 1890-1891.

[2] Belen M A, Caliskan A, Koziel S, Pietrenko-Dabrowska A, Mahouti P. Optimal design of transmitarray antennas via low-cost surrogate modelling[J]. Scientific Reports, 2023, 13: 15044.

[3] Wei Z, Zhou Z, Wang P, Ren J, Yin Y, Pedersen G F, Shen M. Fast and automatic 3D modeling of antenna structure using CNN-LSTM network for efficient data generation[EB/OL]. arXiv:2306.15530, 2023.

[4] Singh P, Panda S S, Hegde R S. Deep-learning empowered multi-objective antenna design: a polygon patch antenna case study[C]. 2024 National Conference on Communications (NCC), 2024: 1-6. DOI: 10.1109/NCC60321.2024.10486033.

[5] Ayestaran R G, Las-Heras F, Martinez J A. Non uniform-antenna array synthesis using neural networks[J]. Journal of Electromagnetic Waves and Applications, 2007, 21(8): 1001-1011.

[6] Mishra S, Yadav R N, Singh R P. Neural estimations of first null beamwidth for broadside and end-fire uniform linear antenna arrays using RBF neural networks[C]//Advances in Intelligent Systems and Computing. Springer, 2015, 343: 445-451. DOI: 10.1007/978-81-322-2268-2_46.

[7] Yang Y, Zhang M, Peng S, Ye M, Zhang Y. Direction-of-arrival estimation for a random sparse linear array based on a graph neural network[J]. Sensors, 2024, 24(1): 91.

[8] He C, Lu Y, Chen W, Ai B, Wong K K, Niyato D. Graph neural network enabled fluid antenna systems: a two-stage approach[EB/OL]. arXiv:2502.03922, 2025.

[9] Xie X, Lu Y, Ding Z. Graph neural network enabled pinching antennas[EB/OL]. arXiv:2502.05447, 2025.

[10] Huang J, Nan J, Gao M, Wang Y. Antenna modeling based on meta-heuristic intelligent algorithms and neural networks[J]. Applied Soft Computing, 2024, 159: 111623.

[11] Pietrenko-Dabrowska A, Koziel S, Leifsson L. Expedited metaheuristic-based antenna optimization using EM model resolution management[C]. International Conference on Computational Science, 2023.

[12] Yang X, Nan J, Wang M. Antenna modeling and optimization based on 1D GP-HetCNN surrogates and intelligent methods[J]. Discover Applied Sciences, 2026, 8(2): 1-22. DOI: 10.1007/s42452-025-07962-7.

# 讨论与分析

# 3 讨论与分析

## 3.1 现有技术优势

### 3.1.1 设计效率提升

AI 辅助天线设计方法最直接的优势体现在设计效率的提升。传统天线设计通常依赖全波电磁仿真软件对结构参数进行反复扫描和迭代优化，单次仿真往往需要较高的计算资源和较长的计算时间。已有综述指出，机器学习与深度学习方法可通过建立天线结构参数与电磁响应之间的代理映射，提高天线设计、优化和选型效率[1-3]。正如第二章文献所述，将深度学习作为代理模型，能够在训练完成后快速预测天线性能，从而显著减少高保真全波仿真的调用次数。

这种效率提升改变了仿真在设计流程中的作用。例如，利用低成本代理建模技术，可以在极少训练样本下快速完成多个频段的透射阵天线设计，Belen 等仅使用 270 个训练样本和 100 个保留样本构建透射阵单元代理模型，并完成 8-14 GHz、22-28 GHz 和 28-36 GHz 多频段设计与实验验证[4]。对于复杂的优化任务，代理模型还可以与多岛差分进化算法、粒子群优化（PSO）或变保真元启发式流程结合，形成“代理模型 + 智能优化算法”的快速搜索框架[5-7]。在这种协同框架下，大量候选结构的初步筛选和优化迭代由代理模型完成，仅需少量全波仿真（如 HFSS）进行验证，从而极大缩短了设计周期。

### 3.1.2 非线性建模与多目标优化能力增强

天线结构参数与电磁性能之间通常呈现强非线性关系。传统经验公式难以准确描述贴片尺寸、馈电位置和阵元间距微小变化所带来的影响。而人工神经网络（ANN）、径向基函数（RBF）网络和卷积神经网络（CNN）展现出了强大的非线性逼近能力[2,8]。研究表明，RBF 神经网络能够有效适应天线阵列的非线性特性，实现稳健的方向性估计[9]。

在多目标优化方面，AI 方法也表现出极强的适应性。现代天线设计需要同时兼顾反射系数、输入阻抗、辐射方向图和带宽等多个指标。通过深度学习代理模型与元启发式算法（如海鸥优化算法、改进蝴蝶算法等）的深度融合，系统能够在几分钟内完成对多边形贴片天线等多目标逆向设计[5]。此外，新型群智能优化算法也能反过来优化神经网络的权值和偏置，进一步提升非线性代理模型的预测精度[6]。

### 3.1.3 复杂结构表征能力提升

随着天线形态向非均匀阵列、流体天线和复杂拓扑结构发展，传统基于规则欧式数据的参数化表达逐渐失效。AI 方法的重要优势在于突破了这一限制，提供了更多维的结构表征方式。对于任意形状的多边形天线或 3D 结构，CNN 与 LSTM 结合的网络能够直接以图像作为输入进行特征提取并自动生成建模代码[10]。

更重要的是，图神经网络（GNN）的引入为复杂天线拓扑提供了天然的表达工具。在天线阵列中，GNN 能够将阵元、阵元间距以及耦合关系作为图的“节点”和“边”纳入统一模型中。例如，在随机稀疏线阵测向或流体天线系统联合优化中，GNN 通过邻居节点聚合有效补偿了缺失阵元信息，在不对阵列几何形状作过多假设的前提下，显著提升了复杂结构的表征能力和阵列综合效果[11-13]。

## 3.2 现有技术不足

### 3.2.1 数据样本依赖较强

尽管深度学习代理模型能够显著加快预测过程，但其本质是数据驱动的，性能高度依赖训练样本的数量和质量。在天线设计中，获取高质量数据需要依赖耗时的全波电磁仿真。这意味着 AI 方法在一定程度上是将计算成本从“优化迭代阶段”转移到了“数据集构建阶段”[1,4]。

为了缓解传统手工建模和扫参获取数据效率低下的问题，现有研究甚至需要专门开发基于 CNN-LSTM 的结构自动建模框架来加速数据生成过程[10]。这侧面反映了高昂的数据获取成本依然是制约深度学习在天线设计中广泛应用的核心瓶颈。对于超表面或大型阵列等高维设计空间，若缺乏有效的主动学习、自适应采样或变保真建模策略，庞大的样本需求将使其失去效率优势[7,14]。

### 3.2.2 泛化能力与物理可解释性不足

现有多数 AI 天线设计模型（如基于 ANN 或单一 CNN 的预测器）主要捕捉训练数据中的统计规律，缺乏明确的物理机理支撑。当待预测的天线拓扑、工作频段或材料边界条件超出训练集分布时，模型的泛化能力会急剧下降。微波 CAD 领域的 ANN 综述也指出，数据集质量、训练样本覆盖范围和模型外推能力仍是神经网络代理模型落地时必须面对的问题[2,8]。

同时，深度学习模型往往是一个“黑盒”，其输出结果虽然在数值上能够拟合 S 参数或方向图，但无法保证这些结果严格满足麦克斯韦方程、能量守恒或互易性定理等基础电磁规律。若模型仅追求拟合精度而脱离物理约束，极易在设计空间的边缘区域产生非物理的预测结果，导致逆向设计失效。物理信息神经网络（PINN）的核心思想正是将偏微分方程等物理规律嵌入损失函数或网络训练过程，为解决此类问题提供了重要启发[15-16]。

### 3.2.3 工程验证与可制造性考虑不足

AI 算法在复杂天线参数反向寻优和阵列综合（如非均匀阵列合成）中可能生成理想化的数值解，但这些解并不一定具备良好的可制造性。许多研究仍停留在软件仿真和算法对标阶段，未将加工公差、介质损耗、装配误差和馈电网络复杂度作为约束条件纳入模型[1,3]。

虽然部分研究通过实验验证了代理模型的有效性[4]，但总体而言，面向 AI 生成结构的暗室测试和长期稳定性评估依然缺乏。若优化结果缺少实物验证环节的闭环校正，全波仿真结果与实际工程性能之间的偏差将限制该类方法的落地应用。

## 3.3 研究空白

### 3.3.1 标准化数据集与评价体系不足

目前的 AI 天线设计研究大多依赖研究者自行构建的数据集，不同文献在天线类型、频段（如 8-14 GHz 或 28-36 GHz）、仿真软件、网格精度和评价指标上各不相同[4,17]。这种“各自为战”的局面导致不同深度学习模型或改进后的元启发式算法（如 WOA、COA、GWO）难以进行横向公平比较，也阻碍了优秀算法的工程复用。构建开源的、涵盖多频段多拓扑的标准天线数据集与统一评价指标，是当前领域的迫切需求。

### 3.3.2 物理机理与数据驱动的深度融合不充分

虽然部分研究引入了多保真度模型来平衡计算精度与开销[7,14]，但大多数模型依然停留在纯数据拟合层面。天线设计本质受制于复杂的电磁场分布和边界条件。如何将电磁学先验知识（如等效电路模型、解析公式或电磁偏微分方程）嵌入到神经网络的损失函数或网络结构中，构建“物理信息神经网络（PINN）”，从而提升小样本条件下的预测精度和跨频段泛化能力，是亟待突破的研究空白[15-16]。

### 3.3.3 系统级与多物理场协同优化研究不足

现有的深度学习和图神经网络研究，大多针对单一天线的 S 参数预测或特定阵列的方向图综合（如波达方向估计或流体天线波束形成）[11-13]。然而，在实际工程中，天线系统不仅需要满足电磁性能，还需要在复杂的通信系统中进行功率分配联合优化（如二部图建模联合优化），同时承受热、力、风载等环境影响。已有多物理场与机器学习辅助优化研究表明，结构、电磁和环境约束的联合建模正成为工程化设计的重要方向[14,18]。从单一电磁性能优化，向包含电磁-热-力多物理场耦合以及系统级通信指标联合优化的演进，是当前研究尚未全面覆盖的盲区。

## 3.4 本章总结

本章基于文献综述分析了 AI 辅助天线设计的发展现状。总体而言，CNN、LSTM、GNN 等深度学习模型与改进的元启发式算法协同工作，已成功将传统天线设计从单一的“仿真试错”转化为“高效建模、精准预测与多目标优化联合”的智能框架，显著提升了针对非均匀阵列、多边形贴片等复杂结构的表征与设计能力。然而，该领域仍面临数据样本获取成本高、模型缺乏物理可解释性及泛化能力弱等痛点。未来的研究应聚焦于减少数据依赖的轻量化代理模型、融入麦克斯韦物理规律的机制约束网络，以及建立涵盖制造约束和多物理场的全流程智能验证体系，以推动 AI 天线技术从理论仿真向工程落地迈进。

## 本章参考文献

[1] Sarker N, Podder P, Mondal M R H, et al. Applications of machine learning and deep learning in antenna design, optimization, and selection: a review[J]. IEEE Access, 2023, 11: 103890-103915.

[2] El Misilmani H M, Naous T, Al Khatib S K. A review on the design and optimization of antennas using machine learning algorithms and techniques[J]. International Journal of RF and Microwave Computer-Aided Engineering, 2020, 30(10): e22356.

[3] Khan M M, Hossain S, Mozumdar P, et al. A review on machine learning and deep learning for various antenna design applications[J]. Heliyon, 2022, 8(4): e09317.

[4] Belen M A, Caliskan A, Koziel S, Pietrenko-Dabrowska A, Mahouti P. Optimal design of transmitarray antennas via low-cost surrogate modelling[J]. Scientific Reports, 2023, 13: 15044.

[5] Singh P, Panda S S, Hegde R S. Deep-learning empowered multi-objective antenna design: a polygon patch antenna case study[C]//2024 National Conference on Communications (NCC). IEEE, 2024: 1-6.

[6] Huang J, Nan J, Gao M, Wang Y. Antenna modeling based on meta-heuristic intelligent algorithms and neural networks[J]. Applied Soft Computing, 2024, 159: 111623.

[7] Koziel S, Pietrenko-Dabrowska A, Leifsson L. Expedited metaheuristic-based antenna optimization using EM model resolution management[C]//International Conference on Computational Science. Springer, 2023.

[8] Feng F, Na W, Jin J, Zhang J, Zhang W, Zhang Q J. Artificial neural networks for microwave computer-aided design: the state of the art[J]. IEEE Transactions on Microwave Theory and Techniques, 2022, 70(11): 4597-4619.

[9] Mishra S, Yadav R N, Singh R P. Neural estimations of first null beamwidth for broadside and end-fire uniform linear antenna arrays using RBF neural networks[C]//Advances in Intelligent Systems and Computing. Springer, 2015, 343: 445-451.

[10] Wei Z, Zhou Z, Wang P, et al. Fast and automatic 3D modeling of antenna structure using CNN-LSTM network for efficient data generation[EB/OL]. arXiv:2306.15530, 2023.

[11] Yang Y, Zhang M, Peng S, Ye M, Zhang Y. Direction-of-arrival estimation for a random sparse linear array based on a graph neural network[J]. Sensors, 2024, 24(1): 91.

[12] He C, Lu Y, Chen W, Ai B, Wong K K, Niyato D. Graph neural network enabled fluid antenna systems: a two-stage approach[EB/OL]. arXiv:2502.03922, 2025.

[13] Xie X, Lu Y, Ding Z. Graph neural network enabled pinching antennas[EB/OL]. arXiv:2502.05447, 2025.

[14] Koziel S, Ogurtsov S. Antenna design by simulation-driven optimization[M]. Cham: Springer International Publishing, 2014.

[15] Raissi M, Perdikaris P, Karniadakis G E. Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations[J]. Journal of Computational Physics, 2019, 378: 686-707.

[16] Karniadakis G E, Kevrekidis I G, Lu L, et al. Physics-informed machine learning[J]. Nature Reviews Physics, 2021, 3: 422-440.

[17] Yang X, Nan J, Wang M. Antenna modeling and optimization based on 1D GP-HetCNN surrogates and intelligent methods[J]. Discover Applied Sciences, 2026, 8(2): 1-22.

[18] Han B, Wu Q, Yu C, et al. Low-wind-load broadband dual-polarized antenna and array designs using sequential multiphysics machine-learning-assisted optimization[J]. IEEE Transactions on Antennas and Propagation, 2025, 73(1): 135-148.

# 改进与未来展望

# 4 改进与未来展望

正如第二章所述，深度学习代理模型、图神经网络以及生成式模型为天线设计提供了新的方法路径，也显著提升了性能预测、结构优化和复杂阵列建模的效率。然而，结合第三章分析可以看出，现有方法仍存在数据依赖较强、跨结构泛化能力不足、物理机理融合不充分以及工程闭环验证不足等问题。为进一步推动 AI 辅助天线设计从任务专用模型走向可信、可迁移和可落地的智能设计体系，未来研究可从以下几个方向展开。

## 4.1 从任务专用代理模型向天线基础模型演进

现有高保真代理模型，如传统神经网络、支持向量机等，多针对特定天线结构、特定频段或特定性能指标建立映射关系，本质上仍属于“一任务一模型”的专用建模范式。这类方法在训练数据覆盖范围内通常能够取得较好的预测效果，但当结构类型、频段范围、材料参数或边界条件发生变化时，模型往往需要重新采样和训练，跨任务迁移能力有限。

未来的一个重要方向是构建面向电磁与天线领域的基础模型。该类模型可在大规模、多类型、多模态的电磁数据上进行预训练，学习天线几何结构、材料参数、馈电方式、电磁场分布、S 参数和远场方向图之间的共性表征规律。在此基础上，面对新的天线结构或新的设计任务时，可通过少量样本微调实现快速适配，从而缓解传统代理模型泛化能力不足的问题。与单一代理模型相比，天线基础模型更强调跨结构、跨频段和跨任务的知识迁移能力，是提升 AI 天线设计通用性的重要方向。

> **参考文献：** 《基于深度学习与代理模型的天线优化设计研究进展》_汤昊臻

## 4.2 从图结构建模向全域电磁应用延伸

图深度学习在天线领域仍具备广阔优化空间与研究前景，后续可扩充阵列样本类型，融入圆阵、圆环阵等多元拓扑结构，构建通用性更强的图神经网络模型；还可重构方向图图结构数据，调整数据输入输出形式，依托图神经网络独立完成天线阵列综合任务。同时可立足电磁领域非欧式数据处理特点，搭建适配各类研究场景的图数据表征方式，将相关技术延伸至超材料结构、超材料天线设计等更多研究方向，进一步拓宽图深度学习在天线与电磁领域的应用范围。

> **参考文献：** 《基于图深度学习的天线阵列优化设计》_苏倩

## 4.3 从孤立算法组合向智能闭环设计平台发展

当前 AI 天线优化流程通常由多个相对独立的环节组成，包括结构建模、样本生成、代理模型训练、智能优化、全波仿真验证和结果修正等。这些环节之间仍需要较多人工干预，设计效率和流程自动化水平受到限制。未来，AI 辅助天线设计需要从单一算法研究走向面向工程应用的智能闭环设计平台。

在该平台中，设计者可输入工作频段、增益、带宽、极化方式、尺寸限制和应用场景等需求，系统自动完成结构生成、代理模型预测、优化算法调用、HFSS/CST 仿真验证和结果反馈。大语言模型智能体可用于调度不同工具和算法，例如自动生成仿真脚本、控制电磁仿真软件、分析仿真结果并选择新的采样点。结合主动学习机制，平台可优先仿真最具价值的候选结构，并将仿真或实测结果反馈给代理模型持续更新。由此形成“需求输入—结构生成—代理预测—全波验证—模型更新—工程反馈”的闭环流程。该方向有望提升 AI 天线设计的自动化、可验证性和工程落地能力。

> **参考文献：** 《基于智能优化算法和代理模型的天线优化设计》_范文敏。（注：在文初步实现了微带天线的自动优化设计，将不同的算法、模型集成为软件平台以应对各种类型的天线设计是可继续研究的方向）

# 总结

本文系统梳理了人工智能辅助天线设计的技术演进轨迹，重点剖析了从人工神经网络、图深度学习到元启发式算法的协同优化方案。本综述的主要贡献在于：深刻揭示了当前天线AI设计高度依赖高质量数据、缺乏物理机理约束以及跨任务泛化能力弱的核心瓶颈，并由此论证了天线设计范式向“基础模型（Foundation Models）”跃升的必然性。

未来的 AI 天线设计将彻底告别孤立、碎片化的“一任务一模型”专用范式，加速向着以大规模电磁数据预训练为核心的“天线基础模型”演进以实现跨结构与跨频段的知识迁移，同时推动图深度学习向圆阵、超材料等“全域电磁拓扑”延伸，并依托大语言模型智能体与主动学习机制构建打通“需求-生成-仿真-反馈”的“智能闭环设计平台”，从而全面实现天线设计的通用化、全域化与全流程自动化落地。

# 附录：天线优化设计相关论文内容提取

# 天线优化设计相关论文内容提取

生成日期：2026-05-18

说明：前三篇为“绪论”章节及对应“参考文献”提取；《基于深度学习与代理模型的天线优化设计研究进展_汤昊臻.pdf》为正文及参考文献全文提取。文本由本地 PDF 自动提取生成，个别公式、图表、上下标和分栏顺序可能需要对照原 PDF 校核。

## 基于机器学习的天线性能分析与优化_彭康宁：绪论

> 来源：文献/基于机器学习的天线性能分析与优化_彭康宁.pdf
> 提取范围：第一章 绪论 + 参考文献

1.1 研究背景

  天线作为一种变换器，能够实现传输线上传播的导行波与自由空间中传播的电磁波之间
的相互变换，是无线电设备中用来发射或接收电磁波的部件。无线电通信、广播、电视、雷
达、导航、电子对抗、遥感、射电天文等工程系统，凡是利用电磁波来传递信息的，都依靠
天线来进行工作[1]。天线的发展大致可分为三个历史阶段，第一阶段为 19 世纪末至 20 世纪
30 年代初的线天线时期，典型天线为八木天线；第二阶段为 20 世纪 30 年代初至 50 年代末
的面天线时期，这一时期广泛采用抛物面天线和其他形式的反射面天线，并且还发明了波导
缝隙天线、介质棒天线、螺旋天线等。此外，天线的基本理论也得到了丰富和完善；第三阶
段为 20 世纪 50 年代至今的大发展时期，这一时期对天线提出许多新的要求，如：高增益、
精密跟踪、快速扫描、宽频带、低旁瓣等，并出现了许多新型天线。由于天线结构日益复杂，
对其进行分析的方法也从理论分析法如传输线模型法、腔体模型法、几何光学法、口径场法
等，发展为电磁场数值计算方法如矩量法(MOM)[2] 、时域有限差分法(FDTD)[3] 、有限元法
(FEM)[4]和几何绕射理论(GTD)[5]等，这些数值计算方法成为分析各种复杂天线问题的有力工
具，并已形成商用软件，主要有 HFSS、CST、FEKO、XFDTD 等。

  1.1.1 天线的发展方向

  随着无线通信技术的不断发展，市场对无线通信设备的要求越来越高，其也被赋予越来
越多的功能，而天线作为无线电系统中不可缺少的设备，主要的发展方向为：多频带化、智
能化、小型化、集成化及高性能化(宽频带、高增益、低旁瓣、低交叉极化等)、可重构[6-7]。
随着天线的结构愈加复杂，设计自由度增加，以及天线的多功能化，基于 HFSS、CST 等仿
真软件实现对天线结构的仿真通常花费几分钟至十几分钟，对于天线性能的优化以及获得最
优设计参数这一过程则需要在优化参数范围内重复多次迭代，这一优化过程可能要花费数天
甚至数十天，不仅耗费大量时间，且占据大量的计算资源[8]。因此，对于较为复杂的天线结
构或大型阵列，需要采取一定的方法在保证精确度的情况下，加速其仿真和优化过程，减少
计算资源的消耗。

  1.1.2 人工智能在电磁场中的应用

  人工智能从概念提出到蓬勃发展，机器智能水平飞速提升主要得益于三个因素：大数据、
机器学习(深度学习)算法、GPU 硬件平台。并且人工智能的概念已经逐步渗入到各行各业，
并将其与该领域的技术相融合，转化为现实生产力，满足实际需求。而机器学习、深度学习
算法现已被逐渐应用到电磁场领域中，主要为天线[9-11]、微波电路的加速优化设计[12-13]，超表
面逆设计[14]，数值计算加速[15]，故障检测[16]等。机器学习或深度学习算法通过采集的训练样
本数据来训练代理模型，并将该训练好的模型代替复杂繁琐的理论计算模型，或者耗时的电
磁仿真软件，以便后续能快速而精确地预测性能，实现优化设计。这也为天线性能分析和结
构参数的优化提供了研究思路。

1.2 研究现状

  机器学习方法现已被广泛地应用于单个天线性能分析和优化设计，阵列天线单元的性能
分析和设计，超表面天线阵列的逆向设计，以及天线阵列故障单元的检测等。其研究思路大
致为：首先，基于 CST 或 HFSS 等软件对初步设计的天线结构进行性能仿真，获得一定数量
的样本数据集；其次，设计适用于该问题的机器学习或深度学习算法，将数据样本分为训练
样本和验证样本，并用训练样本数据训练该机器学习算法模型，当该算法模型在验证样本数
据集上的预测误差小于设定的阈值时，其可以代替电磁仿真软件对新的天线尺寸参数进行性
能分析，在保证一定预测精度的情况下，大大缩减了仿真时间；最后，结合优化算法如遗传
算法(GA)[17]，粒子群算法(PSO)[18]等，在设定的尺寸参数范围内实现对天线某一性能或多个
性能的优化，大大缩减了仿真优化时间。

  1.2.1 单个天线或阵列单元的性能分析

  文献[19]中将 10 种不同的基本机器学习模型相结合，并基于该混合机器学习模型对天线在
某个频点处的多个性能参数进行分析或设计。首先，通过特征选择选出对天线性能影响最大
的尺寸参数，以减少训练样本数据量；其次，混合多种基础机器学习模型，并建立天线尺寸
参数与性能参数之间的非线性映射关系，从而快速而准确地分析天线性能或设计天线；混合
机器模型与 10 个基本机器学习模型相比，具有最小的预测误差，表明了该混合机器模型的可
行性和有效性；文献[20]为了克服微带贴片天线工作带宽窄的缺点，通过在辐射贴片上加载缝
隙来实现双频谐振，增加阻抗带宽，并且基于 MLP 和 RBF 来预测天线的两个谐振频点，从

而计算该天线的工作带宽，实验结果表明，RBF 神经网络具有更高的预测精度和更短的预测
时间；此外，除了采用 ANN 对单个天线的某一性能参数进行仿真，文献[21]设计了一种新的
机器学习模型，实现了对天线的 S 参数，增益，以及辐射方向图的计算。该机器学习模型将
ANN，SVM 算法与向量拟合(Vector Fitting)方法相结合，分别训练三个预测不同性能参数的
模型，预测结果表明，该机器学习模型能代替仿真软件实现快速而精确的性能分析；文献[22]
则基于 SVM 算法对反射天线阵列单元的反射系数进行预测，阵列单元的尺寸参数，介电基
质参数，以及入射角度，工作频率作为 SVM 模型的输入，而天线在两种极化状态下的反射
系数作为输出。预测结果和仿真结果的对比表明该算法可以作为一种代理模型，并用于整个
阵列的辐射方向图的快速计算。

  1.2.2 单个天线或阵列单元的优化设计

  在单一天线或天线阵列的设计过程中，通常需要对其某个性能参数进行优化或多目标优
化。由于基于仿真软件的仿真优化耗费较多的时间和计算资源，因此，训练机器学习模型来
作为代理模型对天线及其阵列的性能进行预测，并结合优化算法对其进行优化设计。文献[21]
中基于机器学习模型对法布里-珀罗谐振腔天线的 S 参数，增益以及辐射方向图进行预测，并
结合 NSGA-II 算法实现了对天线的优化，满足设计指标；文献[23]中基于 SVM 模型建立反射
阵列单元的尺寸参数、入射角度与其对应的共极化和交叉反射极化反射系数之间的非线性映
射关系，用以代替耗时繁琐的 MOM-LP 仿真模型，在保证预测精度的情况下，实现对其反射
系数矩阵的快速预测，从而便于后续对天线阵列单元的性能进行优化，以满足设计要求。

  1.2.3 超表面天线阵列及其单元的逆向设计

  随着超表面概念和技术的日益成熟，其在天线中的应用也愈加广泛。并且超表面单元或
阵列的设计自由度较大，因此，结合机器学习尤其是深度学习算法能够快速实现超表面单元
或阵列的设计。文献[24]提出了一种将卷积自编码器(CAN)和逆向设计神经网络(IDN)相结合的
深度学习模型，建立超表面吸收体在工作频带内的反射系数曲线与尺寸参数之间的非线性映
射关系。其中，CAN 用于对采样频点较多，维度较高的反射系数曲线进行降维，IDN 用于建
立 CAN 得到的降维数据样本与尺寸参数之间的映射关系。因此，该深度学习模型可以基于反
射系数的要求，直接预测得到结构参数，大大缩短了优化时间，节省了计算资源；文献[25]中
基于 GAN 对于给定的传输系数光谱，逆向设计对应的超表面结构。该超表面结构由介质基
底和不同形状的金属贴片结构组成，将金属贴片结构编码化，用 0-1 矩阵表示。该基于 GAN

的深度学习模型主要由生成器，判别器，以及模拟器组成，生成器用于根据给定的传递函数
得到对应的超表面结构矩阵，判别器和模拟器共同作用于生成器，控制生成器产生合理的，
符合传输系数频谱的结构，预测结果表明，该深度学习模型能基于任意设定的频谱，快速逆
向设计超表面；对于可编码超表面阵列天线的设计，文献[26]采取了 CNN 来建立阵列天线的
波束指向与可编码超表面对应的编码矩阵之间的映射关系，超表面单元集成了有源元件，以
其通断状态表示 1-0，基于辐射方向图表示天线阵列的波束指向。该方案为天线中的实时复波
束的形成提供一种可能的研究思路。

  1.2.4 天线阵列故障单元的检测

  天线阵列由许多辐射单元组成，当存在故障单元时，会对天线阵列的辐射方向图等造成
破坏，使天线系统无法正常工作。因此，很有必要对故障单元进行检测，便于采取相应措施
进行馈电调整，以恢复方向图。文献[27]中基于 ANN 对直线阵列天线的故障元件进行监测，
建立存在故障元件的阵列天线的辐射方向图与故障单元的数量和位置的映射关系，实验结果
证明，ANN 可以代替校准系统实现对故障元件的精准预测，避免校准系统损坏或者由此造成
的维护成本增加等问题；文献[28]则基于 CNN 实现了微带贴片天线阵列故障元件的数量和位
置的预测，将辐射方向图作为 CNN 的输入，所有天线单元的馈电情况作为 CNN 的输出，随
着训练样本的增多，CNN 的预测准确率逐渐提高，并且高于 SVM 的预测精度。

1.3 本文研究工作

  随着无线通信技术的不断发展，其在广播、电视、雷达、导航、电子对抗、射电天文等
工程系统中发挥越来越重要的作用。天线作为无线通信技术中的重要组成部分，由于通信行
业高速率大容量低时延的要求，其也被赋予越来越多的功能，并且朝着多频带化、小型化、
高性能化和可重构的方向发展，因此，对于天线或天线阵列的设计变得愈加复杂。而基于 CST
和 HFSS 等电磁仿真软件的性能分析和优化设计通常需要耗费较长时间，
                                  占用较多计算资源，
随着人工智能学科的不断发展，机器学习和深度学习理论的不断丰富和完善，GPU 计算性能
的不断增强，其在电磁场与电磁波领域的应用也逐渐广泛，尤其是微波器件的加速设计。本
文主要研究机器学习在天线性能分析和优化设计中的应用，工作内容主要包括以下三个方面：
  （1）设计新的机器学习模型和天线结构，并将该机器学习模型用于分析设计天线的性能，
主要为工作频带内的 S 参数曲线和增益曲线。首先，设计了一个 CPW 馈电的ψ形缝隙加载
的单极子天线，该天线具有阻抗带宽较宽的特点，并且通过ψ形缝隙在阻抗带宽内形成阻带

特性；其次，将 GRNN 模型与 PCA 算法相结合，用于预测不同缝隙尺寸下天线结构的 S 参
数曲线；最后，将所设计的 GRNN-PCA 与 BPNN-PCA，RBFNN-PCA，GRNN 和 RBFNN 算
法的预测性能进行比较，验证了该模型较高的预测精度和较为简单的结构特征。此外，还将
集成学习的思想用于机器学习模型的设计中，并用于天线性能分析，进一步验证了集成学习
模型相比于单一机器学习模型的优势。
  （2）基于 CGAN 求解二维泊松方程。设置 9 个不同的应用场景，并基于 FDM 求解所有
场景下的泊松方程，得到一定的训练样本用以训练设计的 CGAN 模型，其对测试样本的预测
结果表明，其能够代替 FDM 算法实现对电磁方程的加速计算。并且与基于 CNN 的计算结果
对比，其具有更高的预测精度，更短的预测时间，并且需要的训练样本数更少，适用的计算
场景更多。
  （3）设计新的机器学习模型，将 VAE 与 BPNN 相结合，作为代理模型实现对天线性能
的分析，并将 VAE-BPNN 模型与 GA 算法相结合，对其 S 参数性能进行优化。采用其他文献
中的天线结构以验证该算法的有效性，并将其与其他算法对比，进一步证明了该算法的高效
性与准确性。将 VAE-BPNN 与 GA 算法中的 NSGA-II 算法相结合，可以实现多目标性能的优
化，并对参考文献中的天线性能进行多目标优化来验证该算法的有效性。

### 参考文献

[1] Balanis C A . Antenna theory : analysis and design[J]. IEEE Antennas & Propagation Society Newsletter, 2003,
24(6):28-29.
[2] Sheng X , Wei S . Method of Moments[M]. Wiley-IEEE Press, 2012.
[3] Sullivan, Dennis M . Electromagnetic Simulation Using the FDTD Method[M]. IEEE Press, 2013.
[4] Reddy J N J N . An Introduction To The Finite Element Method[M]. John Wiley & Sons, Ltd, 2013.
[5] Pathak P H , Carluccio G , Albani M . The Uniform Geometrical Theory of Diffraction and Some of Its
Applications[J]. IEEE Antennas & Propagation Magazine, 2013, 55(4):41-69.
[6] 曹逸. 机器学习辅助优化与天线设计技术研究[D]. 东南大学.
[7] 杨振超. 新型手机天线的研究及设计[D]. 电子科技大学, 2011.
[8] Qi, Wu, Yi, et al. Machine-Learning-Assisted Optimization and Its Application to Antenna Designs:
Opportunities and Challenges[J]. 中国通信：英文版, 2020, 17(4):13.
[9] Wang Z , Fang S , Wang Q , et al. An ANN-Based Synthesis Model for the Single-Feed
Circularly-Polarized Square Microstrip Antenna With Truncated Corners[J]. IEEE Transactions on Antennas &
Propagation, 2012, 60(12):5989-5992.
[10] Liu B , Akbarian H A , Vandenbosch G , et al. An efficient method for antenna optimization based on
evolutionary computation and machine learning techniques[J]. IEEE Transactions on Antennas & Propagation,
2014, 62(1):7-18.
[11] Robustillo P , Zapata J , Encinar J A , et al. ANN Characterization of Multi-Layer Reflectarray Elements
for Contoured-Beam Space Antennas in the Ku-Band[J]. IEEE Transactions on Antennas & Propagation, 2012,
60(7):3205-3214.
[12] Feng, Zhang, Chao, et al. Parametric Modeling of EM Behavior of Microwave Components Using Combined
Neural Networks and Pole-Residue-Based Transfer Functions[J]. IEEE Transactions on Microwave Theory &
Techniques, 2016.
[13] Jin J , Zhang C , Feng F , et al. Deep Neural Network Technique for High-Dimensional Microwave
Modeling and Applications to Parameter Extraction of Microwave Filters[J]. IEEE Transactions on Microwave
Theory and Techniques, 2019, 67(10):4140-4155.
[14] Qiu T , Shi X , J Wang, et al. Deep Learning: A Rapid and Efficient Route to Automatic Metasurface
Design[J]. Advanced Science, 2019.
[15] He, Ming, Yao, et al. Machine-Learning-Based PML for the FDTD Method[J]. IEEE Antennas & Wireless
Propagation Letters, 2018.
[16] Yeo B K , Lu Y . Fast detection and location of failed array elements using the fast SVM algorithm. IEEE,
2010:1-4.
[17] Moghaddam E S . Design of a Printed Quadrifilar-Helical Antenna on a Dielectric Cylinder by Means of a
Genetic Algorithm [Antenna Applications Corner][J]. IEEE Antennas & Propagation Magazine, 2011,
53(4):262-268.
[18] Li Y L , Shao W , You L , et al. An Improved PSO Algorithm and Its Application to UWB Antenna
Design[J]. IEEE Antennas & Wireless Propagation Letters, 2013, 12:1236-1239.
[19] M. Xue, D. Shi, Y. He and C. Li, "A Novel Intelligent Antenna Synthesis System Using Hybrid Machine
Learning Algorithms," 2019 International Symposium on Electromagnetic Compatibility - EMC EUROPE, 2019,
pp. 902-907, doi: 10.1109/EMCEurope.2019.8871996.
[20] Thakare V V , Singhal P K . BANDWIDTH ANALYSIS BY INTRODUCING SLOTS IN MICROSTRIP
ANTENNA DESIGN USING ANN[J]. Progress in Electromagnetics Research M, 2009, 9:107-122.

[21] Xiao L Y , Shao W , Jin F L , et al. Multi-Parameter Modeling with ANN for Antenna Design[J]. IEEE
Transactions on Antennas and Propagation, 2018:1-1.
[22] Prado D R , Lopez-Fernandez J A , Barquero G , et al. Fast and Accurate Modeling of Dual-Polarized
Reflectarray Unit Cells Using Support Vector Machines[J]. IEEE Transactions on Antennas and Propagation, 2018,
PP(99):1-1.
[23] Prado, Daniel R , Lopez-Fernandez, et al. Support Vector Regression to Accelerate Design and Crosspolar
Optimization of Shaped-Beam Reflectarray Antennas for Space Applications[J]. IEEE Transactions on Antennas
and Propagation, 2019.
[24] Ma J , Huang Y , Pu M , et al. Inverse Design of Broadband Metasurface Absorber Based on
Convolutional Autoencoder Network and Inverse Design Network[J]. Journal of Physics D Applied Physics, 2020,
53(46).
[25] Zhaocheng, Liu, Dayu, et al. Generative Model for the Inverse Design of Metasurfaces.[J]. Nano Letters,
2018.
[26] Shan T , Li M . Coding Programmable Metasurfaces Based on Deep Learning Techniques[C]// 2019 IEEE
International Symposium on Antennas and Propagation and USNC-URSI Radio Science Meeting. IEEE, 2019.
[27] Patnaik A , Choudhury B , Pradhan P , et al. An ANN Application for Fault Finding in Antenna Arrays[J].
IEEE Transactions on Antennas & Propagation, 2007, 55:775-777.
[28] Q. Chen, H. Ma and E. -P. Li, "Failure Diagnosis of Microstrip Antenna Array Based on Convolutional
Neural Network," 2019 IEEE Asia-Pacific Microwave Conference (APMC), 2019, pp. 90-92, doi:
10.1109/APMC46564.2019.9038656.
[29] 陈先昌. 基于卷积神经网络的深度学习算法与应用研究[D]. 浙江工商大学, 2014.
[30] Hou Y C , Su D L , Ma J P . Analysis and design of Ultra Wide Band Planar monopole antenna[C]//
International Symposium on Antennas. IEEE, 2008.
[31] Liang J , Lu G , Chiau C C , et al. CPW-fed circular disc monopole antenna for UWB applications[C]//
IEEE International Workshop on IEEE Antenna Technology: Small Antennas & Novel Metamaterials. IEEE,
2005.
[32] Oktafiani F , Wahyu Y , Wijayanto Y N . Broadband octagonal patch antenna for cognitive radio
applications[C]// 2015 IEEE Asia Pacific Conference on Wireless and Mobile (APWiMob). IEEE, 2016.
[33] Ruchandani K , Kumar M . A novel CPW fed octagonal aperture antenna with back patch for UWB
applications[C]// International Conference on Nascent Technologies in the Engineering Field. IEEE, 2015:1-5.
[34] Ibrahim A A , Abdalla M A . Compact size UWB antenna with multi-band notched charactristics for
wireless applications[C]// 2016 IEEE International Symposium on Antennas and Propagation & USNC/URSI
National Radio Science Meeting. IEEE, 2016.
[35] Tarbouch M , Amri A E , Terchoune H . Compact CPW-Fed microstrip octagonal patch antenna with H
slot for WLAN and WIMAX applications[C]// International Conference on Wireless Technologies. IEEE, 2017.
[36] Weng W C , Hung C L . An H-Fractal Antenna for Multiband Applications[J]. IEEE Antennas & Wireless
Propagation Letters, 2015, 13:1705-1708.
[37] Tarbouch M , Amri A E , Terchoune H . Compact CPW-Fed microstrip octagonal patch antenna with H
slot for WLAN and WIMAX applications[C]// International Conference on Wireless Technologies. IEEE, 2017.
[38] Singh B K . Design of rectangular microstrip patch antenna based on Artificial Neural Network
algorithm[C]// 2015 2nd International Conference on Signal Processing and Integrated Networks (SPIN). IEEE,
2015.
[39] Rolph S . Fractal Geometry: Mathematical Foundations and Applications[J]. Mathematical Gazette, 1990,
74(469):288-317.
[40] Yang S , Browne A . Neural network ensembles: combining multiple models for enhanced performance
using a multistage approach[J]. Expert Systems, 2010, 21(5):279-288.

[41] Hansen L K , Salamon P . Neural network ensembles[J]. IEEE Transactions on Pattern Analysis & Machine
Intelligence, 2002, 12(10):993-1001.
[42] Wold S , Esbensen K , Geladi P . Principal component analysis[J]. Chemometrics & Intelligent Laboratory
Systems, 1987, 2( 1–3):37-5.
[43] Zenko B , Todorovski L , Dzeroski S . A Comparison of Stacking with Meta Decision Trees to Bagging,
Boosting, and Stacking with other Methods. IEEE, 2001.
[44] Gu J, Wang Z, Kuen J, et al. Recent advances in convolutional neural networks. 2015, arXiv preprint
arXiv:1512.07108.
[45] Ohnishi R , Wu D , Yamaguchi T , et al. Numerical Accuracy of Finite-Difference Methods[C]// 2018
International Symposium on Antennas and Propagation (ISAP). IEEE, 2019.
[46] Cui R , Wei M . Research and application of Successive Over-Relaxation Iterative Algorithm[C]//
International Conference on Electronic & Mechanical Engineering & Information Technology. IEEE, 2011.
[47] Shan T , Tang W , Dang X , et al. Study on a Fast Solver for Poisson's Equation Based On Deep Learning
Technique[J]. IEEE Transactions on Antennas and Propagation, 2020, PP(99):1-1.
[48] Mirza M, Osindero S. Conditional Generative Adversarial Nets. 2014, arXiv: 1411.1784. [Online]. Avaliable:
https://arxiv.org/abs/1411.1784
[49] Kingma DP, Ba J. Adam: A Method for Stochastic Optimization. 2014, arXiv: 1412.6980. [Online]. Avaliable:
https://arxiv.org/abs/1412.6980
[50] Bengio, Yoshua, Courville, et al. Representation Learning: A Review and New Perspectives[J]. IEEE
Transactions on Pattern Analysis & Machine Intelligence, 2013, 35(8):1798-1828.
[51] Kingma DP, Welling M. Auto-encoding variational bayes. arXiv.org:1312.6114v10, 2014.
[52] Kuo Y L , Wong K L . Printed double-T monopole antenna for 2.4/5.2 GHz dual-band WLAN operations[J].
IEEE Xplore, 2003.
[53] Sharma Y , Zhang H H , Xin H . Machine Learning Techniques for Optimizing Design of Double
T-Shaped Monopole Antenna[J]. IEEE Transactions on Antennas and Propagation, 2020, 68(7):5658-5663.
[54] Madsen K, Nielsen HB, Tingleff O. Methods for non-linear least squares problems. Society for industrial &
applied mathematics 2004.
[55] Naseri P , Hum S V . A Generative Machine Learning-Based Approach for Inverse Design of Multilayer
Metasurfaces[J]. IEEE Transactions on Antennas and Propagation, 2021, PP(99):1-1.
[56] Kiarashinejad Y , Abdollahramezani S , Adibi A . Deep learning approach based on dimensionality
reduction for designing electromagnetic nanostructures[J]. 计算材料学(英文), 2020(1):12.
[57] Sharma Y , Wu J , Hao X , et al. Sparse linear regression for optimizing design parameters of double
T-shaped monopole antennas[C]// 2017 IEEE International Symposium on Antennas and Propagation &
USNC/URSI National Radio Science Meeting. IEEE, 2017.
[58] Xiao L Y , Shao W , Jin F L , et al. Multi-Parameter Modeling with ANN for Antenna Design[J]. IEEE
Transactions on Antennas and Propagation, 2018:1-1.
[59] Ching-Shih. Deb, K. Pratap, A. Agarwal, S. and Meyarivan, T. (2002), " A fast and elitist multiobjective
genetic algorithm: NSGA-II", IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
[60] Li G , Yang S , Feng Q , et al. Dual-polarized Differential-fed Phased Array Antenna with Sidelobe
Suppression Based on NSGA-II for 5G Millimeter Wave Application[C]// 2020 International Conference on
Microwave and Millimeter Wave Technology (ICMMT). 2020.


## 基于图深度学习的天线阵列优化设计_苏倩：绪论

> 来源：文献/基于图深度学习的天线阵列优化设计_苏倩.pdf
> 提取范围：第 1 章 绪论 + 参考文献

绪论

1.1 研究背景与意义

   随着现代通信系统对天线的需求日益增多，各种基于数值分析方法的电磁仿
真软件（如 HFSS、CST 等）在天线的设计与优化上发挥着重要的作用。但是电
磁仿真软件在使用时通常会占用极大的计算资源，优化过程耗时较长，并且往往
需要工程师掌握夯实的电磁理论基础并根据丰富的工程经验进行多次仿真调试，
这使得天线的设计和优化过程繁琐复杂、效率低下，并存在优化结果不理想等问
题。近年来，随着人工智能技术和深度学习的迅速发展，基于深度学习的方法逐
渐被应用到了电磁领域，如电磁散射、雷达、电磁兼容等[1-5]。而在天线的设计和
优化过程中，天线结构特征和电磁特征性能指标之间存在高度非线性关系，且存
在多极值、多变量等问题。机器学习、深度学习等人工智能算法因其具有强大的
泛化能力，在天线的设计优化过程中有着巨大的潜力，能够通过对已有天线结构
特征和电磁特征性能指标之间关系的学习，建立机器学习模型，从而对未知天线
结构进行性能预测或对需求性能指标进行反向结构寻优。
   为实现天线的智能设计优化，提高天线优化效率，将机器学习等优秀算法应
用于天线设计中，更好地解决传统天线设计的耗时耗资问题意义深远。早在 1998
年，Mishra R K 等人[6]就将人工神经网络（Artificial Neural Network, ANN）应用
在了方形贴片天线的设计过程中，提出了一种基于神经网络的方形贴片天线模型，
该模型在计算时间和精度方面都具有较好的鲁棒性，初步实现了人工智能在天线
领域的应用。目前已经有很多代理模型被引入天线设计等电磁领域[7-11]，如人工
神经网络、克里金插值（Kriging Interpolation）、径向基函数（Radial Basis Function,
RBF）等。基于人工智能的新型天线设计方法与使用电磁仿真软件的传统天线设
计方法相比，无论是在天线的设计效果还是优化效率方面都有着巨大的优势。
   在天线阵列分析与综合领域，人工神经网络和深度神经网络等传统的神经网
络算法因其在提取欧式空间数据特征方面的强大能力取得了巨大的成功，能够对
阵列结构和激励等参数和辐射方向图之间的关系进行学习和训练，从而实现天线
阵列的分析或综合。但传统的神经网络也正是因为只适合处理欧式空间数据，所
以只能应用于单一拓扑结构的天线阵列，具有一定的局限性。

   近几年，图神经网络（Graph Neural Network, GNN）因其能够处理从非欧式
空间中生成的图结构数据而受到了广泛的关注，被应用在了电子商务、社交网络、
生物制药、交通预测等诸多领域[12-15]。例如在电子商务中，将用户和商品及其之
间的关系建模为图结构数据，基于图的学习系统可以利用用户和商品之间的交互
关系给出更高准确度的建议；在交通预测中，将地点、时间、交通流量、天气等
多模态信息建模为图结构数据，基于图神经网络的模型可以利用这些多模态信息
更精确的预测未来的交通流量。将图深度学习技术应用于天线阵列设计是一个较
新的课题研究方向，但与上述图神经网络在各领域的应用理念相似，某些相对复
杂的天线或天线阵列模型也可看作是一种图结构数据，数据中包含天线结构、阵
元间的距离和耦合关系等参量。将多种不同拓扑结构的天线阵列建模为不规则又
具有统一格式的图结构数据后，图神经网络可以利用该图结构数据更灵活有效地
学习阵列结构和电磁性能之间的关系，从而突破阵列拓扑结构的限制，实现对多
种不同拓扑结构的天线阵列的分析和综合。将图深度学习应用于天线阵列领域具
有巨大的发展前景。

1.2 国内外研究现状

1.2.1 基于深度学习的天线设计研究现状

   传统的天线设计和优化多依赖于工程师的理论基础和工程经验，使用电磁仿
真软件通过等步长的扫参方式进行多次调试，结果很难达到最优，并且过程复杂
且耗时长。随着人工智能技术的迅速发展，深度学习又迎来了新的发展高潮，将
深度学习技术引入天线领域，通过学习天线结构参数和性能指标参数之间的关系，
在一定程度上弥补电磁仿真软件的弊端，加快天线的设计优化过程，成为近年来
科研人员的关注重点。这也为天线的设计和优化带来了一个新的设计思路。目前
人工神经网络（Artificial Neural Network, ANN）和深度神经网络（Deep Neural
Network, DNN）是在天线领域使用最多且最成熟的网络模型，天线结构设计或天
线阵列综合为目前研究最多的两大问题。
   在天线结构设计和优化方面，多使用 ANN 进行训练和预测[16-18]。例如，1998
年，Mishra RK 等人[6]将人工神经网络用于方形贴片天线设计，实现了 ANN 在
天线设计领域的应用，验证了基于人工智能实现天线设计优化的可行性。随后，
为了提高模型的准确率和训练的速度，很多学者在基础的 ANN 上进行了改进和
优化，构建了更符合需求的网络模型。2005 年，Delgado H J 等人[19]提出一种结
合有限差分时域（Finite Difference Time Domain, FDTD）新型神经网络算法用于

合成印刷偶极子天线。2006 年，Mohamed M D A 等人[20]将径向基函数（RBF）
引入神经网络模型中，成功预测贴片天线的辐射贴片和馈电微带线的最佳几何尺
寸。2013 年，Akdagli A 等人[17]使用多层感知器（Multilayer Perceptron, MLP）优
化人工神经网络，并成功应用于 E 形紧凑型微带天线的谐振频率计算。2018 年，
Kapetanakis T N[21]利用人工神经网络解决了求解薄圆形环形天线在自由空间辐
射的逆问题。同年，Xiao Liye 等人[22]提出了一个引入了支持向量机的人工神经
网络模型，该模型能够通过每个分支对天线的回波损耗、增益、辐射方向图多个
性能指标同时进行预测，在法布里珀罗天线上的实验结果验证了模型的有效性和
模型效率。众多文献都验证了人工神经网络在天线结构尺寸设计中的可行性，并
与传统的天线结构设计方法相比，有着巨大的优势。
   在天线阵列的设计和综合方面，2007 年，Ayestaran R G 等人[23]使用神经网
络进行非均匀天线阵列合成，模型将天线阵列的辐射模式施加到每个阵元与激励
电压相关联，无需对天线的几何形状进行任何假设，并且即使考虑耦合效应也可
以解决非均匀天线阵列的问题。2015 年，Mishra Subhash 等人[24]利用径向基函数
神经网络（RBF-NN）对共线短偶极子和平行短偶极子的均匀线性阵列进行了方
向性估计，该方法很容易适应天线阵的非线性特性，不需要复杂的数学过程，因
此速度更快，文章也验证了该方法的鲁棒性。2020 年，Yuan L[8]提出一种新的多
分支人工神经网络建模技术并将其应用于解决天线阵列方向性估计的逆问题。在
天线阵列的逆问题的建模过程中，模型的输入是电磁参数，阵列的几何结构参数
和物理参数被设置为输出，利用逆问题的单调性对数据进行训练，通过人工神经
网络的并行分支同时对阵列的多个电磁参数进行预测和求解。2021 年，Zhou Z[25]
将深度神经网络用于解决 1×8 贴片天线阵列方向图综合及其逆问题求解。该文
章以电磁仿真结果作为数据集，训练 DNN 来学习产生辐射模式的激励信号的映
射及其逆问题，提出的 DNN 架构为电磁解决方案提供了一种高效、可靠的补充，
可推广到更大规模的天线阵或用于辅助复杂天线阵的反向设计。2022 年，Can Cui
等人[26]提出了一个新颖的基于编码器-解码器结构的人工神经网络框架，用于可
重构的非均匀间隔线阵综合。在该框架中，不同的编解码器分支用于合成阵列所
需的辐射方向图。而在每个编解码器的分支中，编码器充当阵列合成器，解码器
充当阵列分析器，通过对阵元振幅和位置相关损失函数最小值的寻优，不仅可以
成功实现辐射方向图多种赋形，还可以获得阵元间距和阵元振幅。
   上述文献证明了深度学习在天线设计与优化上的优势和潜力，也为图深度学
习在该领域的应用提供了一定的基础。为突破 ANN 和 DNN 只能应用在单一结
构天线阵列的局限性，将能够处理非欧式空间数据的图神经网络应用于天线阵列
的分析和综合十分必要。

1.2.2 基于图深度学习的天线设计研究现状

   为更好地学习非欧式空间数据的特征，利用深度学习技术实现对图数据的端
到端学习成为了一个热门且重要的研究课题。2005 年，Marco Gori 等人[27]首次
提出图神经网络（Graph Neural Network, GNN）的概念。在随后的几年发展中，
受卷积神经网络发展的影响，人们开始将卷积应用到图神经网络之中，并通过不
断地改进和拓展，越来越多的基于图的神经网络被提出，并应用在了含有图数据
的众多领域。但将图深度学习应用于天线领域当前还是一个较新的课题。
   2019 年 Zhang G 等人[28]第一次将 GNN 应用于电磁领域，提出了一个用于
分布式电路设计的 GNN 模型——Circuit-GNN。如图 1-1 所示，该模型将谐振器
作为节点，谐振器之间的距离（耦合关系）作为边，从而将电路建模成一种图数
据；将谐振器的位置、开环方向和谐振器之间的距离作为模型的输入，电路的电
磁特性作为模型的输出，训练后的模型能够实现分布式电路的性能预测和谐振器
分布设计，此方法极大地加速了高频电路的设计工作。

                    图 1-1 Gircuit-GNN 模型架构

     (a) 准周期阵列结构             (b) 用于准周期阵列的 DGCNN 模型架构
               图 1-2 基于 DGCNN 的准周期阵列分析方法

   近两年，许多学者开始尝试将 GNN 应用于天线设计与优化中，并取得了一
定的进展和成果。2022 年，Wei X.等人[29]提出了一种用于分析准周期阵列的基
于动态图卷积神经网络（Dynamic Graph Convolutional Neural Network, DGCNNs）
的方法来分析由不同几何参数单元组成的准周期阵列。图中的每个节点表示准周
期阵列中的一个单元，每个节点特征由相应单元的几何参数决定，如图 1-2(a)所
示。文章提出的 DGCNN 模型如图 1-2(b)所示，能够动态更新图的边特征以获得

单元之间的互耦关系。训练成功后的 DGCNN 模型够快速预测具有随机分布和可
变尺寸单元的超材料准周期阵列的回波损耗。
   同年，Tekbiyik 等人[30]提出了一个用于天线阵列雷达波达方向（Direction of
arrival, DoA）估计的基于图注意力网络（GAT）的深度学习框架，如图 1-3 所示。
该网络由四个全局注意力层（Global Attention Layers, GALs）和四个全连接层
（Dense Layers）组成，分别用于图结构数据的特征提取和对二维角度值的回归
分析，最终实现对单像素接收机压缩信道上的数据测量和进行 DoA 估算，并在
相对较低信噪比的情况下综合检索出高保真的 DoA 信息。

                图 1-3 用于 DoA 的 GAT 模型框架

   在上述文献中，图神经网络被成功应用在了电磁领域中，这为本课题使用图
神经网络解决天线阵列设计问题提供了理论依据和可行性支撑。

1.2.3 基于元启发式算法的天线设计研究现状

   深度学习模型和框架在天线设计领域多用来作为代理模型，其作用是代替传
统全波仿真方法对天线性能或阵列方向图进行正向预测；而元启发式算法多用于
进行天线或阵列参数的反向寻优。元启发式算法是通过模拟自然界的规则而演化
出的一种新兴的计算技术，是智能优化算法的一个重要分支，因其在处理非线性、
复杂性等问题时表现出来的先进性，已成为越来越多的学者关注的焦点。
   元启发式算法适用于求解高度非线性多参数多目标问题，因此为寻找天线的
最优解提供了可能。其中遗传算法（Genetic Algorithm, GA）和粒子群算法（Particle
Swarm Optimization, PSO）提出较早且发展成熟，在电磁领域和天线的设计优化
中应用最为广泛[31-35]。例如，2000 年，杨帆等[36]将遗传算法成功地应用到微带天
线的优化中，优化出了带宽由初始 5%展宽到 16.6%的宽带天线、以及双频比为
1:131 且具有同向的线极化的双频工作天线；2006 年，焦永昌[37]等将粒子群优化
算法运用于天线阵的方向图综合上，表明粒子群算法在天线阵列综合中具有广泛
的应用前景；2007 年，丁敏等[38]将二维遗传算法和时域有限差分混合模型应用
于有限尺寸超宽带平面天线的自动设计，成功地设计了一种小尺寸的超宽带天线。
同年，Aaron J. Kerkhoff 等人[39]将遗传算法优化方法应用于平面单极子天线的设
计中，优化了天线的性能，改进了方向图的对称性，拓宽了陷波的有效带宽。2019

年 ， Fan Xiaohong 等 [40] 提 出 了 一 种 自 更 新 适 应 度 估 计 的 粒 子 群 优 化 算 法
（SFEPSO），以提高微带天线的设计效率，实现了对 E 型双频微带天线和
WLAN/WiMAX 多频段天线的优化设计。
     除此之外，更多的元启发式算法被提出，例如灰狼优化算法（Grey Wolf Op-
timizer, GWO）[41]、鲸鱼优化算法（Whale Optimization Algorithm, WOA）[42]、果
蝇优化算法（Fruit Fly Optimization Algorithm, FOA）[43]、人工蜂群算法（Artificial
bee colony, ABC）[44]等也逐渐被应用于天线的设计与优化中。2014 年，Sotirios
K.等人[45]利用人工蜂群算法设计并优化了一种用于超高频无源射频识别标签的
平面螺旋天线。2017 年，Yuan Pengliang 等人[46]将鲸鱼优化算法引入到稀疏阵列
天线综合中，优化后的稀疏阵列天线具有更低的旁瓣电平。通过在具有不同约束
条件的稀疏阵列上进行测试，实验结果表明鲸鱼优化算法优于改进粒子群优化算
法和改进遗传算法等优化算法。2018 年，Amirashkan Darvish 等人[47]提出了一种
改进的果蝇优化算法用于天线阵综合，利用提出的算法合成了具有超宽带特性的
U 型槽微带阵列天线，优化后的天线结构的阻抗带宽比原结构带宽提高了 181.6%。
同年，张鑫等[48]提出了基于自适应变微分的人工蜂群算法，并通过稀疏非均匀天
                              2020 年，
线阵列和八木天线的设计验证了该算法有助于解决天线阵列设计问题。
Li Xun 等人[49]将灰狼算法应用于线性阵列合成以及双频 E 形贴片天线和宽频带
磁电偶极子天线的优化设计，并与遗传算法、粒子群算法和差分进化等著名的元
启发算法进行了比较，验证了该算法在天线设计领域的优势和潜力。
   通过以上算例可以看出，元启发式算法已在天线的设计优化过程中得到了广
泛应用。但元启发式算法存在容易陷入局部最优的问题，如何寻找更适用于天线
优化的元启发式算法，如何改进算法而进一步提高天线的优化效率，是本课题的
研究内容之一。

1.3 本文工作安排

   本文主要对图深度学习在天线阵列分析和综合上的应用进行了研究，其章节
安排如下：
   第一章为绪论。主要介绍了图深度学习在天线设计领域上的研究背景与意义，
并分析了基于深度学习的天线设计方法、基于图深度学习的天线设计方法和基于
元启发式算法的天线设计方法的研究现状。最后介绍了本文的工作安排和整体结
构。

  第二章为基础理论。首先介绍了天线的相关理论和基本参数，以及天线阵列
的相关知识，重点介绍了阵列方向图的相关公式。其次对图深度学习的基础理论
进行了介绍，包括图的定义以及常见的图神经网络。
  第三章为基于图深度学习的天线阵列分析方法。首先详细描述了本文提出的
用于阵列方向图预测的图神经网络的模型架构，架构主要分为三个部分：天线阵
列的图表示、基于图同构网络的特征提取、基于残差增强和复数全连接网络的方
向图预测。接着介绍了用于模型训练和预测的天线阵列算例和数据集样本构成。
其次介绍了模型训练的参数设置和模型优劣的评估指标。最后进行了实验结果的
定量分析和定性分析，并通过对比实验和消融实验验证了所提模型架构的先进性
和有效性。实验结果表明，所提模型能够同时有效预测多种不同拓扑结构的阵列
方向图，且准确性优于其他神经网络模型，并且计算效率比传统全波仿真算法提
高了 6 个数量级。
  第四章为基于图深度学习和黏菌算法的天线阵列综合方法。在本章中，首先
提出了用于阵列综合的图深度学习模型架构，将黏菌算法引入其中，实现图神经
网络的反向寻优任务。其次介绍了黏菌算法的基本原理和精英反向学习、惯性权
重两个改进策略，并在标准测试函数上对改进算法进行了性能评估。最后完成了
多种满足需求的天线阵综合，并验证了所提算法相较其他算法的优越性。
  第五章为总结与展望。本章对全文的工作进行了总结，并对课题的后续研究
和未来发展提出了展望。

### 参考文献

[1] Li L L, Wang L G, Teixeira F L, et al. DeepNIS: Deep neural network for nonlinear
    electromagnetic inverse scattering [J]. IEEE Transactions on Antennas and
    Propagation, 2019, 67(3): 1819-1825.
[2] Yao H M, Jiang L J. Machine learning based neural network solving methods for
    the FDTD method [C]. Proceedings of the 2018 IEEE Antennas and Propagation
    Society International Symposium on Antennas and Propagation & Usnc/Ursi
    National Radio Science Meeting. Boston: IEEE, 2018: 2321-2322.
[3] Yang Y, Li Y, Li K, et al. DECCO: Deer-learning enabled coverage and capacity
    optimization for massive MIMI systems [J]. IEEE Access, 2018, 6: 23361-23371.
[4] Shi D, Fang W, Zhang F F, et al. A novel method for intelligent EMC management
    using a "knowledge base" [J]. IEEE Transactions on Electromagnetic
    Compatibility, 2018, 60(6): 1621-1626.
[5] Chen S Z, Wang H P, Xu F, et al. Target classification using the deep
    convolutional networks for SAR images [J]. IEEE Transactions on Geoscience
    and Remote Sensing, 2016, 54(8): 4806-4817.
[6] Mishra R K, Patnaik A. Neural network-based CAD model for the design of
    square-patch antennas [J]. IEEE Transactions on Antennas and Propagation, 1998,
    46(12): 1890-1891.
[7] Yuan L, Yang X-S, Wang C, et al. Multibranch artificial neural network modeling
    for inverse estimation of antenna array directivity [J]. IEEE Transactions on
    Antennas and Propagation, 2020, 68(6): 4417-4427.
[8] Sharma Y, Zhang H H, Xin H. Machine learning techniques for optimizing design
    of double T-shaped monopole antenna [J]. IEEE Transactions on Antennas and
    Propagation, 2020, 68(7): 5658-5663.
[9] Cui L, Zhang Y, Zhang R, et al. A modified efficient KNN method for antenna
    optimization and design [J]. IEEE Transactions on Antennas and Propagation,
    2020, 68(10): 6858-6866.
[10] Aguni L, El Yassini A, Chabaa S, et al. Design of a symmetric CPW-fed patch
     antenna for WLAN/WIMAX applications using ANN [J]. Wireless Personal
     Communications, 2020, 115(1): 439-456.
[11] Kumar P, Tripathi P, Raj S, et al. ANN based novel inverted L shape slotted patch
     antenna for wide band applications [C]. Proceedings of the 2019 International

     Conference on Computing, Power and Communication Technologies (GUCON).
     New Delhi: IEEE, 2019: 390-393.
[12] Liu W, Zhang Y, Wang J, et al. Item relationship graph neural networks for e-
     commerce [J]. IEEE Transactions on Neural Networks and Learning Systems,
     2022, 33(9): 4785-4799.
[13] Kan S, Cen Y, Li Y, et al. Local semantic correlation modeling over graph neural
     networks for deep feature embedding and image retrieval [J]. IEEE Transactions
     on Image Processing, 2022, 31: 2988-3003.
[14] Jiang S, Balaprakash P. Graph neural network architecture search for molecular
     property prediction [C]. Proceedings of the 2020 IEEE International Conference
     on Big Data. Atlanta: IEEE, 2020: 1346-1353.
[15] Cui Z, Henrickson K, Ke R, et al. Traffic graph convolutional recurrent neural
     network: A deep learning framework for network-scale traffic learning and
     forecasting [J]. IEEE Transactions on Intelligent Transportation Systems, 2020,
     21(11): 4884-4894.
[16] Kaur M, Sivia J S. Giuseppe peano and cantor set fractals based miniaturized
     hybrid fractal antenna for biomedical applications using artificial neural network
     and firefly algorithm [J]. International Journal of Rf and Microwave Computer-
     Aided Engineering, 2020, 30(1): 1-11.
[17] Akdagli A, Toktas A, Kayabasi A, et al. An application of artificial neural network
     to compute the resonant frequency of E-shaped compact microstrip antennas [J].
     Journal of Electrical Engineering-Elektrotechnicky Casopis, 2013, 64(5): 317-322.
[18] Guney K, Sarikaya N. A hybrid method based on combining artificial neural
     network and fuzzy inference system for simultaneous computation of resonant
     frequencies of rectangular, circular, and triangular microstrip antennas [J]. IEEE
     Transactions on Antennas and Propagation, 2007, 55(3): 659-668.
[19] Delgado H J, Thursby M H. A novel neural network combined with FDTD for the
     synthesis of a printed dipole antenna [J]. IEEE Transactions on Antennas and
     Propagation, 2005, 53(7): 2231-2236.
[20] Mohamed M D A, Soliman E A, El-Gamal M A. Optimization and
     characterization of electromagnetically coupled patch antennas using RBF neural
     networks [J]. Journal of Electromagnetic Waves and Applications, 2006, 20(8):
     1101-1114.
[21] Kapetanakis T N, Vardiambasis I O, Ioannidou M P, et al. Neural network
     modeling for the solution of the inverse loop antenna radiation problem [J]. IEEE
     Transactions on Antennas and Propagation, 2018, 66(11): 6283-6290.

[22] Xiao L-Y, Shao W, Jin F-L, et al. Multiparameter modeling with ANN for antenna
     design [J]. IEEE Transactions on Antennas and Propagation, 2018, 66(7): 3718-
     3723.
[23] Ayestaran R G, Las-Heras F, Martinez J A. Non uniform-antenna array synthesis
     using neural networks [J]. Journal of Electromagnetic Waves and Applications,
     2007, 21(8): 1001-1011.
[24] Mishra S, Yadav R N, Singh R P. Directivity estimations for short dipole antenna
     arrays using radial basis function neural networks [J]. IEEE Antennas and
     Wireless Propagation Letters, 2015, 14: 1219-1222.
[25] Zhou Z, Wei Z, Zhang Y, et al. Training of deep neural networks in
     electromagnetic problems: a case study of antenna array pattern synthesis [C].
     Proceedings of the 2021 IEEE MTT-S International Wireless Symposium (IWS).
     Nanjing: IEEE, 2021: 1-3.
[26] Cui C, Li W T, Ye X T, et al. Synthesis of mask-constrained pattern-reconfigurable
     nonuniformly spaced linear arrays using artificial neural networks [J]. IEEE
     Transactions on Antennas and Propagation, 2022.
[27] Gori M, Monfardini G, Scarselli F. A new model for learning in graph domains
     [C]. Proceedings of the 2005 IEEE International Joint Conference on Neural
     Networks. Montreal: IEEE, 2005: 729-734.
[28] G Z, H H, D K. Circuit-GNN: Graph neural networks for distributed circuit design
     [C]. Proceedings of the 36th International Conference on Machine Learnig.
     California: PMLR, 2019: 7364-7373.
[29] Xiang W, Zhang Z, Yang X, et al. Fast prediction of quasi-periodic array using
     dynamical graph convolutional neural networks [J]. IEEE Antennas and Wireless
     Propagation Letters, 2022: 1-5.
[30] Tekbıyık K, Yurduseven O, Kurt G K. Graph attention network-based single-pixel
     compressive direction of arrival estimation [J]. IEEE Communications Letters,
     2022, 26(3): 562-566.
[31] Dhaliwal B S, Pattnaik S S. Development of PSO-ANN ensemble hybrid
     algorithm and its application in compact crown circular fractal patch antenna
     design [J]. Wireless Personal Communications, 2017, 96(1): 135-152.
[32] Pratap P, Bhatia R S, Kumar B. Design and simulation of equilateral triangular
     microstrip antenna using particle swarm optimization (PSO) and advanced particle
     swarm optimization (APSO) [J]. Sadhana-Academy Proceedings in Engineering
     Sciences, 2016, 41(7): 721-725.
[33] 宋立众, 段舒雅, 林涛. 一种结合遗传算法和 HFSS 的天线仿真与优化方法
     [J]. 微波学报, 2015, 31(03): 1-6.

[34] Li Y-L, Shao W, You L, et al. An improved PSO algorithm and its application to
     UWB antenna design [J]. IEEE Antennas and Wireless Propagation Letters, 2013,
     12: 1236-1239.
[35] 孙思扬, 吕英华, 张金玲, et al. 基于遗传算法的超宽带微带天线优化设计
     [J]. 电波科学学报, 2011, 26(01): 62-66.
[36] 杨帆, 张雪霞. 遗传算法在微带天线优化中的应用 [J]. 电子学报, 2000, (9):
     91-95.
[37] 焦永昌, 杨科, 陈胜兵, et al. 粒子群优化算法用于阵列天线方向图综合设计
     [J]. 电波科学学报, 2006, (01): 16-20.
[38] Ding M, Jin R, Geng J. Optimal design of ultra wideband antennas using a mixed
     model of 2-D genetic algorithm and finite-difference time-domain [J]. Microwave
     and Optical Technology Letters, 2007, 49(12): 3177-3180.
[39] Kerkhoff A J, Ling H. Design of a band-notched planar monopole antenna using
     genetic algorithm optimization [J]. IEEE Transactions on Antennas and
     Propagation, 2007, 55(3): 604-610.
[40] Fan X, Tian Y, Zhao Y. Optimal design of multiband microstrip antennas by self-
     renewing fitness estimation of particle swarm optimization algorithm [J].
     International Journal of Antennas and Propagation, 2019, 2019: 1-9.
[41] Mirjalili S, Mirjalili S M, Lewis A. Grey wolf optimizer [J]. Advances in
     Engineering Software, 2014, 69: 46-61.
[42] Mirjalili S, Lewis A. The whale optimization algorithm [J]. Advances in
     Engineering Software, 2016, 95: 51-67.
[43] Pan W-T. A new fruit fly optimization algorithm: Taking the financial distress
     model as an example [J]. Knowledge-Based Systems, 2012, 26: 69-74.
[44] Karaboga D, Basturk B. A powerful and efficient algorithm for numerical function
     optimization: artificial bee colony (ABC) algorithm [J]. Journal of Global
     Optimization, 2007, 39(3): 459-471.
[45] Goudos S K, Siakavara K, Sahalos J N. Novel spiral antenna design using artificial
     bee colony optimization for UHF RFID applications [J]. IEEE Antennas and
     Wireless Propagation Letters, 2014, 13: 528-531.
[46] Yuan P L, Guo C J, Ding J, et al. Synthesis of nonuniform sparse linear array
     antenna using whale optimization algorithm [C]. Proceedings of the 6th Asia-
     Pacific Conference on Antennas and Propagation (APCAP). Xian: IEEE, 2017: 1-
     3.
[47] Darvish A, Ebrahimzadeh A. Improved fruit-fly optimization algorithm and its
     applications in antenna arrays synthesis [J]. IEEE Transactions on Antennas and
     Propagation, 2018, 66(4): 1756-1766.

[48] Zhang X, Zhang X, Wang L. Antenna design by an adaptive variable differential
     artificial bee colony algorithm [J]. IEEE Transactions on Magnetics, 2018, 54(3):
     1-4.
[49] Li X, Luk K M. The grey wolf optimizer and its applications in electromagnetics
     [J]. IEEE Transactions on Antennas and Propagation, 2020, 68(3): 2186-2197.
[50] Wu Z, Pan S, Chen F, et al. A comprehensive survey on graph neural networks [J].
     IEEE Transactions on Neural Networks and Learning Systems, 2021, 32(1): 4-24.
[51] Dai H, Kozareva Z, Dai B, et al. Learning steady-states of iterative algorithms over
     graphs [C]. Proceedings of the 35th International Conference on Machine
     Learning (ICML). Stockholm: Journal Machine Learning Research, 2018: 1-9.
[52] Gallicchio C, Micheli A, Ieee. Graph echo state networks [C]. Proceedings of the
     2010 International Joint Conference on Neural Networks (IJCNN). Barcelona:
     IEEE, 2010: 1-8.
[53] Scarselli F, Gori M, Tsoi A C, et al. The graph neural network model [J]. IEEE
     Transactions on Neural Networks, 2009, 20(1): 61-80.
[54] Levie R, Monti F, Bresson X, et al. CayleyNets: Graph convolutional neural
     networks with complex rational spectral filters [J]. IEEE Transactions on Signal
     Processing, 2019, 67(1): 97-109.
[55] Chiang W-L, Liu X, Si S, et al. Cluster-GCN: An efficient algorithm for training
     deep and large graph convolutional networks [C]. Proceedings of the 25th ACM
     SIGKDD International Conference on Knowledge Discovery & Data Mining.
     Anchorage: Assos Computing Machinery, 2019: 257-266.
[56] Li Q, Han Z, Wu X-M, et al. Deeper insights into graph convolutional networks
     for semi-supervised learning [C]. Proceedings of the 32nd AAAI Conference on
     Artificial Intelligence. New Orleans: Assoc Advancement Artificial Intelligence,
     2018: 3538-3545.
[57] Monti F, Boscaini D, Masci J, et al. Geometric deep learning on graphs and
     manifolds using mixture model CNNs [C]. Proceedings of the 30th Conference on
     Computer Vision and Pattern Recognition (CVPR). Honolulu: IEEE, 2017: 5425-
     5434.
[58] Tu K, Cui P, Wang X, et al. Deep recursive network embedding with regular
     equivalence [C]. Proceedings of the 24th ACM SIGKDD Conference on
     Knowledge Discovery and Data Mining. London: Assoc Computing Machinery,
     2018: 2357-2366.
[59] Pan S, Hu R, Long G, et al. Adversarially regularized graph autoencoder for graph
     embedding [C]. Proceedings of the 27th International Joint Conference on
     Artificial Intelligence. Stockholm, 2018: 2609-2615.

[60] Simonovsky M, Komodakis N. GraphVAE: Towards generation of small graphs
     using variational autoencoders [C]. Proceedings of the Artificial Neural Networks
     and Machine Learning – ICANN 2018. Rhodes: Springer International Publishing,
     2018: 412-422.
[61] Bojchevski A, Shchur O, Zugner D, et al. NetGAN: Generating graphs via random
     walks [C]. Proceedings of the 35th International Conference on Machine Learning
     (ICML). Stockholm: Journal Machine Learning Research, 2018: 1-16.
[62] Dai J, Cao Y, Shen Q, et al. Traffic flow prediction based on multi-spatial-
     temporal graph convolutional network [J]. Application Research of Computers,
     2022, 39(3): 780-784.
[63] Jain A, Zamir A R, Savarese S, et al. Structural-RNN: Deep learning on spatio-
     temporal graphs [C]. Proceedings of the 2016 IEEE Conference on Computer
     Vision and Pattern Recognition (CVPR). Las Vegas: IEEE, 2016: 5308-5317.
[64] Li F, Li J, Zhu A, et al. Enhanced spatial and extended temporal graph
     convolutional network for skeleton-based action recognition [J]. Sensors, 2020,
     20(18).
[65] Xu K, Hu W, J L. How powerful are graph neural networks? [C]. Proceedings of
     the International Conference on Learning Representations (ICLR). Vancouver,
     2019: 1-17.
[66] Trabelsi C, Bilaniuk O, Zhang Y, et al. Deep Complex Networks [C]. Proceedings
     of the International Conference on Learning Representations (ICLR). 2018: 1-19.
[67] Feng C Z, Li W T, Cui C, et al. An efficient and universal static and dynamic
     convex optimization for array synthesis [J]. IEEE Antennas and Wireless
     Propagation Letters, 2022: 1-5.
[68] Li S, Chen H, Wang M, et al. Slime mould algorithm: A newmethod for stochastic
     optimization [J]. Future Generation Computer Systems, 2020, 111: 300-323.
[69] Tizhoosh H R. Opposition-based learning: A new scheme for machine intelligence
     [C]. Proceedings of the International Conference on Computational Intelligence
     for Modelling, Control and Automation and International Conference on
     Intelligent Agents, Web Technologies and Internet Commerce (CIMCA-IAWTIC).
     Vienna: IEEE, 2005: 695-701.
[70] Xin Y, Yong L, Guangming L. Evolutionary programming made faster [J]. IEEE
     Transactions on Evolutionary Computation, 1999, 3(2): 82-102.
[71] Kennedy J, Eberhart R. Particle swarm optimization [C]. Proceedings of the
     ICNN'95 - International Conference on Neural Networks. 1995: 1942-1948.
[72] Storn R, Price K. Differential Evolution - A Simple and Efficient Heuristic for
     global Optimization over Continuous Spaces [J]. Journal of Global Optimization,
     1997, 11(4): 341-359.


## 基于智能算法的天线模型优化设计_户肖剑：绪论

> 来源：文献/基于智能算法的天线模型优化设计_户肖剑.pdf
> 提取范围：第一章 绪论 + 参考文献

1.1 本课题研究背景及研究意义
   随着无线通信技术的迅速发展，对天线性能的要求日益提高。传统的天线设计方
法主要依赖经验公式和实验调试，设计周期长、成本高，且难以满足现代通信系统对多
频段、宽带宽、高增益和低功耗的需求。使用自适应免疫退火算法提升了可重构天线设
计的精度和自动化效率[1]。一种高效率的改进的粒子群优化算法通过三个天线模型验证
了有效性[2]。使用一种改进的杜鹃搜索算法优化单脉冲天线的差异模式，获得了较好的
天线性能[3]。通过人工智能(Artificial intelligence，AI)深度学习算法强大的决策能力以
及逼近任意非线性函数关系的能力等优势，建立了一种模拟电磁求解器的深层物理模
型。即将 AI 深度学习算法模型进行不断的学习训练，获得一种天线结构的物理参数到
性能参数之间的映射关系，用以辅助设计者进行天线的设计[4-5]。虽然 AI 深度学习算
法的预测不确定性的特性，需要很多天线尺寸与对应的性能参数的训练集样本来弥补，
导致天线的优化设计时间相应增长。但是通过结合多种优化理论框架，改进算法不足
之处，AI 深度学习还是可以作为天线的自动化设计的一种有效方法。随着 5G 和未来
6G 通信技术的快速发展，对于快速设计高性能天线的需求也更加紧迫。这使得研究者
们面对着很大的挑战。

1.2 国内外研究现状

   根据智能算法优化天线模型的实现方式，主要分为启发式种群行为算法，深度学

习算法以及自适应混合算法等。
1.2.1 启发式算法的天线设计的研究现状

   启发式智能算法包括遗传算法、粒子群优化、模拟退火以及蚁群优化等算法。启发

式智能算法不依赖于求解问题的具体数学特性，而是通过模拟自然界中的物理机制(如

进化和群体行为等)搜索最优解。天线设计是一个复杂的多目标优化问题，通常在阻抗

带宽、增益、辐射方向图以及尺寸等多个性能之间进行权衡。启发式智能算法能够在复

杂的非线性设计空间中快速找到近似最优解，具有很好的灵活性，适用于各种天线类

型的设计问题，包括单目标和多目标优化[6]，这能够帮助天线工程师有效地处理天线模

型优化复杂的问题。

  文献[7]采用自适应的粒子群优化算法对一种三维结构的半球形天线的谐振寄生条

带结构迭代优化。如图 1-1 所示，以寄生条带的长度和位置作为优化参数，天线模型的

阻抗带宽作为优化目标，最终实现了整体天线结构的小型化，并获得了 700 MHz 的阻

抗带宽。

                   图 1-1 自适应 PSO 优化的 3 维的半球形天线结构

   Figure 1-1 Adaptive PSO optimized hemispherical antenna structure in three dimensions

  文献[8]通过引入一种蝙蝠群体智能优化算法，解决了多层堆叠贴片天线的宽阻抗

带宽设计问题。如图 1-2 所示，在蝙蝠群体智能算法进行天线堆叠结构的优化中，将两

层以及三层贴片结构的的长度、宽度以及两个馈电端口的距离作为优化参数，阻抗带

宽定义为优化目标，实现了 34.9%的相对阻抗带宽。

                     (a) 两层的贴片天线                           (b) 三层的贴片天线

                            图 1-2 蝙蝠算法优化的贴片天线

                    Figure 1-2 Patch antenna optimized by bat algorithm

  文献[9]应用改进的二进制粒子群优化算法对理想的“±1”激励磁电偶极子天线阵

进行优化，“+1”激励表示磁电偶极子从默认方向馈电，“−1”激励表示磁电偶极子

从物理相反的方向馈电，如图 1-3 所示。根据优化的阵列组合，将磁电偶极子单元与

256 路并联功分器相结合，设计了一个 16×16 的毫米波天线阵，优化的工作带宽为 24.1–

35.4 GHz，相对带宽为 38%，驻波比小于 2.4，信噪比小于-17.2 dB，增益高达 26.3 dBi。

这个设计方法对于宽带毫米波无线应用具有重大意义。

                         图 1-3 毫米波天线结构

                Figure 1-3 Millimeter wave antenna structure

1.2.2 深度学习的天线设计的研究现状

   在常规微波器件的设计中，通常需要反复调整和优化大量参数（如几何尺寸、材料

特性、工作频率等）以达到预期的性能目标（如带宽、增益、回波损耗、辐射方向图

等）。这一过程往往依赖经验驱动的试错方法或基于电磁仿真的迭代优化，耗时费力且

计算成本高昂。AI 深度学习作为一种高效的代理模型，能够显著加速这一优化过程。

通过训练神经网络等深度学习模型，可以快速预测微波器件的性能，替代部分耗时的

电磁仿真，从而实现高效的参数优化 AI 深度学习不仅能够加速设计流程，还能探索传

统方法难以发现的设计空间，为微波器件的创新设计提供强大支持[10]。因此，AI 深度

学习作为一种代理模型，正在成为微波器件设计中不可或缺的工具，极大地提升了设

计效率和性能。

   文献[11]提出了一种用于超宽带天线设计的深度学习模型，该模型通过智能算法确

定深度网络的模型结构，具有较高的特征学习能力和非线性函数逼近能力，并应用在

MIMO 天线陷波结构的优化设计中，其 S 参数拟合良好，达到了期望的超宽带目标，

如图 1-4 所示。在相同训练样本的情况下，其所提出的优化模型与普遍使用的建模方法

相比，获得了精度较高的效果，优化的 MIMO 天线的均方根误差为 3.56%。该模型具

有更高的预测和泛化能力，也可用于更复杂的天线设计。

                           图 1-4 MIMO 天线结构

                     Figure 1-4 MIMO Antenna Structure

   文献[12]提出一种新颖的双通道神经网络用于第五代手机的宽带 MIMO 天线阵列

设计，该天线工作于 N77/N78/N79 的 Sub-6 GHz 新无线电频段，如图 1-5 所示。该方

法使用特征简化方法减小设计空间，生成有效标识特征的数据集。通过对多个设计目

标使用定义的特征图来代替深度学习模型的直接优化方法。优化后的-6 dB 阻抗带宽达

到了 45.5% (3.28–5.21 GHz)，隔离度小于-12.5 dB。

                         图 1-5 5G 手机 MIMO 天线

                    Figure 1-5 5G cellphone MIMO antenna

   文献[13]使用多层感知机神经网络模型对于微带天线辐射体上十字形槽的 4 个尺

寸以及介质基板与金属地板之间的空气间隙值进行了输出预测，输入层为天线的性能

参数，包括双谐振点(Dual resonance, DR)，双频增益(Dual frequency gain, DFG)，双频

指向(Dual frequency directivity, DFD)，双频天线效率(Dual frequency antenna efficiency,

DFA)以及双频率辐射效率(Dual frequency radiation efficiency, DFR)，输出层为槽的尺寸

和空气间隙，如图 1-6 所示，最终实现了 1.74–1.99 GHz 的阻抗带宽。

                             图 1-6 多层感知机神经网络模型

                   Figure 1-6 Multi-layer perceptron neural network model

    文献[14]提出一种用于降低频率可重构天线互耦和隔离器优化的人工智能驱动的

自动设计程序，其采用并行代理模型辅助差分进化算法加速天线的设计过程。优化的

频率可重构天线阵列拓扑结构，如图 1-7 所示，其可以在 2.5 GHz 的 ISM 频段和 3.4

GHz 的 WiMAX 频段之间灵活切换。通过使用该方法对隔离器的优化，高频段和低频

段的互耦分别降低了 8 dB 和 7 dB。

                          图 1-7 频率可重构天线阵列拓扑结构

              Figure 1-7 Topology of the frequency reconfigurable antenna array

    文献[15]通过一种优化方法用于模拟天线特性的深度神经网络的最优模型超参数，

并基于长短期记忆深度神经网络，如图 1-8 所示，用于天线的结构尺寸优化。目标是预

测扩展的频率响应，在训练深度神经网络时应用各种随机方法确定最优超参数。可以

从散射参数、增益和辐射方向图方面对于天线逆向设计结构。该方法解决了在确定超

参数时严重依赖设计者经验的问题。此外，预测天线未来的频率响应可以大大减少设

计者在计算或测量大频段时的工作量，即无需再计算或测量整个频段。以一个包含两

个阵元的天线阵列验证该方法的有效性。

             图 1-8 用于天线建模的基于 LSTM 的回归深度神经网络预测未来频带

Figure 1-8 LSTM-based regression deep neural network for forecasting future frequency band used for

                                        modeling antennas

1.2.3 自适应混合算法的天线设计的研究现状

   自适应混合算法是针对单个算法往往存在不足之处而产生的一种改进方式，可以

很好地改善单个算法的缺点，提升算法性能。遗传算法虽然应用范围很广，但是具有容

易过早收敛，陷入局部最优解，以及进化后期搜索效率低的弱点。如果将模拟退火算法

引入到遗传算法群体更新的阶段，既能保证群体的多样性，又能在后期逐步加快收敛

速度，克服了遗传算法的以上弱点，从而得到一种混合算法，即模拟退火遗传算法。提

出一种人工神经网络和脉冲神经网络自适应混合神经网络算法，进一步提升了模型的

处理速度和精确度。

   文献[16]将两种粒子群优化算法整合到基于群算子分解的多目标优化算法中，提出

了一种改进的混合优化算法，并作为天线设计的自动优化方案。该设计方案的有效性

通过一个单频带天线以及一个双频紧凑型高隔离多输入多输出(Multiple Input Multiple

Output, MIMO)天线验证，如图 1-9 所示。优化结果显示，单频带天线能够实现 1.24 GHz

的阻抗带宽以及大于 20 dB 的隔离度。双频的 MIMO 天线可以实现 3.3–3.62 GHz 以及

4.85–5.90 GHz 的阻抗带宽以及大于 20 dB 的隔离度。

       (a) 单频段天线                                     (b) 双频段天线

                   图 1-9 蝙蝠算法优化的贴片天线

           Figure 1-9 Patch antenna optimized by bat algorithm

  文献[17]提出了一种基于改进的遗传算法和粒子群优化算法的相控阵天线方向图

合成算法。所优化的阵元结构，如图 1-10 所示。通过图 1-11 的优化流程，对于阵元激

励幅值的优化设计，可以有效的实现天线阵列的波束扫描。对比于常规的遗传算法和

粒子群算法，该混合算法具有更好的优化性能，并具有较高的精度和稳定的鲁棒性。

                (a) 顶层                               (b) 底层

                        图 1-10 天线阵元结构

               Figure 1-10 Antenna array element structure

                           图 1-11 优化流程

                     Figure 1-11 Optimization flow

  文献[18]提出一种基于法布里-珀罗腔技术的高增益宽带谐振型的移动通信基站天

线。采用一种实值编码混合遗传算法优化一个由方形和环形贴片构成的覆盖层，如图

1-12 所示。在保证天线高增益宽频带的特性下，可以将覆盖层优化的很薄，这在实际

应用中具有很强的优势。

                        图 1-12 覆盖层结构

                    Figure 1-12 Overlay structure

  文献[19]提出了一种基于人工神经网络和模拟退火算法的混合方法用以设计宽带

贴片天线。人工神经网络的结构，如图 1-13 所示，其用以描述天线几何尺寸和 S 参数

之间的非线性关系，并利用 HFSS 全波仿真获得的数据集对人工神经网络进行训练，

                    图 1-14 展示了所优化的贴片天线结构。
再结合模拟退火算法用以展宽贴片天线的带宽。

                    图 1-13 人工神经网络结构

            Figure 1-13 Artificial neural network structure

                             图 1-14 天线结构

                       Figure 1-14 Antenna structure

  文献[20]提出将自定义变异粒子群算法和基于混沌映射的改进遗传算法相结合，具

有更快的收敛速度并且不会陷入局部最优解，如图 1-15 所示。通过将如图 1-16 所示的

一个偶极子天线的阻抗带宽优化到 3.3–3.8 GHz，验证了该混合算法的有效性。

                       图 1-15 多目标混合优化算法

          Figure 1-15 Multi-objective hybrid optimization algorithm

                            图 1-16 偶极子天线

                         Figure 1-16 Dipole antenna

  综上所述，启发式智能算法，深度学习算法以及自适应混合算法都能够高效的优

化天线结构和目标性能，这三种方法对于实现 5G 和 6G 时代天线高性能的快速优化设

计有着重要的研究意义。
1.3 本课题研究目标和主要研究内容
1.3.1 课题研究目标
  本课题的研究目标是：
  (1) 全双工天线的神经网络算法设计研究
  将神经网络算法应用于全双工天线的优化设计，前提是理解天线的工作机理。此
外，天线的结构形式多种多样，有些结构参数对于天线的性能影响很小，但有些参数对
于天线性能非常敏感。将天线性能敏感性参数作为优化变量，而天线的阻抗带宽，隔离
度以及增益性能作为目标函数。通过电磁仿真软件与神经网络算法运行软件进行联合
计算，最终得到期望的天线性能。
  (2) 小型化天线的 AI 深度学习设计研究
  通过构建 AI 深度学习模型，实现对天线结构的高效优化，在保证天线小型化尺寸
的同时，显著提升其隔离度性能。深入分析不同天线结构与性能之间的复杂关系，挖掘
隐藏在数据中的规律，为天线设计提供更精准的指导。此外，开发一套基于 AI 深度学
习的天线设计工具，降低人工设计成本，缩短研发周期，推动天线设计向智能化、高效
化方向发展，为 5G/6G 及未来通信技术提供高性能的小型化天线解决方案。
  (3) 天线阵列的启发式智能算法优化研究
  启发式智能算法多种多样，但是在算法的收敛速率、优化效率以及找寻最优解能
力方面有着差异。因此，首先需要比较多种经典启发式智能算法的优劣，找寻一种性能
较优的算法。从而，在该算法基础上进一步改进，使得对于天线阵列的优化设计更有效
率。最后，通过设定多种优化目标函数，并使用启发式智能算法对天线阵列的激励幅值
和激励相位进行优化调整，实现对于旁瓣电平的有效抑制和控制。

1.3.2 主要研究内容
  第一章介绍了智能算法优化天线设计的研究起源、重要性和同行研究情况。
  第二章提出了一种全双工天线的精确径向基函数神经网络的设计方法。基于本次
研究，我们搭建了一个三层的神经网络结构，对天线的阻抗带宽，隔离度以及增益进行
优化，其工作带宽可达 3.40–3.68 GHz。通过 I 型带和 E 型带的解耦结构的引入，实现

了全双工天线大于 35 dB 的隔离度。
   第三章提出了一种新的多输入天线性能和多输出天线尺寸变量的 AI 深度学习方
法。我们设计了一个 4 层的神经网络，其包含一个输入层，2 个隐藏层和一个输出层。
其中输入层的输入定义为双极化天线的 L 形短路结构，矩形槽，L 形金属贴片以及 T
形分支的结构参数，共包含 12 个尺寸。输出层为阻抗带宽，隔离度和增益三种性能。
最终实现了 3.47–3.58 GHz 内的大于 40 dB 的高隔离度。
   第四章提出了一种改进的多元宇宙算法对于平面天线阵列的辐射图进行优化，可
以实现低副瓣的方向图，并可以有效的控制方向图的旁瓣电平为任意值。此外，在主瓣
指向角度确定的情况下，可以在任意角度生成零点，同时也可以降低旁瓣电平。

1.4 课题来源
   本课题来源于广东省重点领域研发计划项目(2020B010176001)以及省部共建精密
电子制造技术与装备国家重点实验室项目并受以上项目资助。

1.5 本章小结

   本章介绍了天线的智能算法设计的研究背景，并阐述了提升天线高效率高性能设

计的紧迫性。随后，以国内外有关研究成果作为依据，进行了详细的分析和介绍。对主

流天线算法优化技术进行深入剖析，确立了针对新一代无线通信系统的高性能技术研

究目标和基本研究内容。

### 参考文献

[1]    X. Xu et al.. Intelligent Design of Reconfigurable Microstrip Antenna Based on
       Adaptive Immune Annealing Algorithm [J]. IEEE Transactions on Instrumentation and
       Measurement, 2022, 71: 1-14.
[2] K. Fu, X. Cai, B. Yuan, Y. Yang and X. Yao. An Efficient Surrogate Assisted Particle
       Swarm Optimization for Antenna Synthesis [J]. IEEE Transactions on Antennas and
       Propagation, 2022, 70(7): 4977-4984.
[3]    X. Li, S. Ma and G. Yang, Synthesis of Difference Patterns for Monopulse Antennas by
       an Improved Cuckoo Search Algorithm [J]. IEEE Antennas and Wireless Propagation
       Letters, 2017, 16: 141-144.
[4] K. Tan, Y. Lu, Z. Zhao, D. Wang and B. Chen. A Dual-Kernel Computational Framework
       for Artificial Intelligence-Based Indoor Communication Analysis [J]. IEEE Antennas
       and Wireless Propagation Letters, 2023, 22(12): 2984-2987.
[5]    R. Mwang'amba, P. Mei, M. O. Akinsolu, B. Liu and S. Zhang. Gain Bandwidth
       Enhancement and Sidelobe Level Stabilization of mmWave Lens Antennas Using AI-
       Driven Optimization [J]. IEEE Antennas and Wireless Propagation Letters, 2024, 23(11):
       3554-3558.
[6] J. S. Smith and M. E. Baginski. Thin-Wire Antenna Design Using a Novel Branching
       Scheme and Genetic Algorithm Optimization [J]. IEEE Transactions on Antennas and
       Propagation, 2019, 67(5): 2934-2941.
[7] M.-C. Tang, X. Chen, M. Li, and R. W. Ziolkowski. Particle Swarm Optimized, 3-D-
       Printed, Wideband, Compact Hemispherical Antenna [J]. IEEE Antennas Wireless
       Propagation Letters, 2018, 17(11): 2031-2035.
[8] Z. Shao, L. -F. Qiu, and Y. P. Zhang. Design of Wideband Differentially Fed Multilayer
       Stacked Patch Antennas Based on Bat Algorithm [J]. IEEE Antennas Wireless
       Propagation Letters, 2020, 19(7):1172-1176.
[9] X. Dai, X. Li and K. -M. Luk. A Planar Wideband Millimeter-Wave Antenna Array With
       Low Sidelobe Using ‘±1’ Excitations [J]. IEEE Transactions on Antennas and
       Propagation, 2021, 69(10): 6999-7004.
[10]   Y. Xiao, K. W. Leung, K. Lu and C. -S. Leung, Mode Recognition of Rectangular
       Dielectric Resonator Antenna Using Artificial Neural Network [J]. IEEE Transactions
       on Antennas and Propagation, 2022, 70(7): 5209-5216.
[11]    J. Nan, H. Xie, M. Gao, Y. Song, and W. Yang. Design of UWB Antenna Based on

       Improved Deep Belief Network and Extreme Learning Machine Surrogate Models [J].
       IEEE Access, 2021, 9:126541-126549.
[12] H. Ahmed, X. Zeng, H. Bello, Y. Wang, and N. Iqbal. Sub-6 GHz MIMO antenna design
       for 5G smartphones: A deep learning approach [J]. AEU-International Journal of
       Electronics and Communications, 2023, 168.
[13]   T. Khan, A. De, and M. Uddin. Prediction of Slot-Size and Inserted Air-Gap for
       Improving the Performance of Rectangular Microstrip Antennas Using Artificial Neural
       Networks [J]. IEEE Antennas Wireless Propagation Letters, 2013, 12: 1367-1371.
[14]   J. Zhang, M. O. Akinsolu, B. Liu, and G. A. E. Vandenbosch. Automatic AI-driven
       design of mutual coupling reducing topologies for frequency reconfigurable antenna
       arrays [J]. IEEE Transactions on Antennas and Propagation, 2021, 69(3):1831-1836.
[15] L. Kouhalvandi and L. Matekovits. Hyperparameter Optimization of Long Short-Term
       Memory-Based Forecasting DNN for Antenna Modeling Through Stochastic Methods
       [J]. IEEE Antennas Wireless Propagation Letters, 2022, 21(4): 725-729.
[16] Q. -Q. Li, Q. -X. Chu, and Y. -L. Chang. Design of Compact High-Isolation MIMO
       Antenna with Multiobjective Mixed Optimization Algorithm [J]. IEEE Antennas
       Wireless Propagation Letters, 2020, 19(8):1306-1310.
[17] W. T. Li, X. W. Shi, Y. Q. Hei, S. F. Liu, and J. Zhu. A Hybrid Optimization Algorithm
       and Its Application for Conformal Array Pattern Synthesis [J]. IEEE Transactions on
       Antennas and Propagation, 2010, 58(10): 3401-3406.
[18] D. Kim, J. Ju, and J. Choi. A Mobile Communication Base Station Antenna Using a
       Genetic Algorithm Based Fabry-Pérot Resonance Optimization [J]. IEEE Transactions
       on Antennas and Propagation, 2012, 60(2): 1053-1058.
[19] Y. He, J. Huang, W. Li, L. Zhang, S. -W. Wong, and Z. N. Chen. Hybrid Method of
       Artificial Neural Network and Simulated Annealing Algorithm for Optimizing
       Wideband Patch Antennas [J]. IEEE Transactions on Antennas and Propagation, 2024,
       72(1): 944-949.
[20] V. H. Hasbestan, Y. Farhang, K. Majidzadeh, and C. Ghobadi. Multi-Objective Hybrid
       Optimization Algorithm for Design a Printed MIMO Antenna With n78–5G NR
       Frequency Band Applications [J]. IEEE Access, 2023, 11: 68231-68242.
[21]   Y. -Z. Liang, F. -C. Chen, K. -R. Xiang, and W. -F. Zeng. Wideband Co-Polarized
       Stacked Patch Antenna for In-Band Full-Duplex Applications [J]. IEEE Transactions on
       Antennas and Propagation, 2023, 71(12): 9920-9925.

[22] J. C. Dash and D. Sarkar. A Colinearly Polarized Full-Duplex Antenna With Extremely
       High Tx–Rx Isolation [J]. IEEE Antennas and Wireless Propagation Letters, 2022,
       21(12): 2387-2391.
[23]   J. P. Jacobs. Accurate Modeling by Convolutional Neural-Network Regression of
       Resonant Frequencies of Dual-Band Pixelated Microstrip Antenna [J]. IEEE Antennas
       and Wireless Propagation Letters, 2021, 20(12): 2417-2421.
[24] J. Jin, Q. Su, Y. Xu, Z. He and Y. Lu, Efficient Radiation Pattern Prediction of Array
       Antennas Based on Complex-Valued Graph Neural Networks [J]. IEEE Antennas and
       Wireless Propagation Letters, 2022, 21(12) pp. 2467-2471.
[25] F. Feng, W. Na, J. Jin, J. Zhang, W. Zhang and Q. -J. Zhang. Artificial Neural Networks
       for Microwave Computer-Aided Design: The State of the Art [J]. IEEE Transactions on
       Microwave Theory and Techniques, 2022, 70(11): 4597-4619.
[26]   L. Yuan, L. Wang, X. -S. Yang, H. Huang and B. -Z. Wang. An Efficient Artificial
       Neural Network Model for Inverse Design of Metasurfaces [J]. IEEE Antennas and
       Wireless Propagation Letters, 2021, 20(6): 1013-1017.
[27]   A. Massa, D. Marcantonio, X. Chen, M. Li, and M. Salucci. DNNs as Applied to
       Electromagnetics, Antennas, and Propagation—A Review [J]. IEEE Antennas Wireless
       Propagation Letters, 2019, 18(11): 2225-2229.
[28]   S. Koziel, N. Çalık, P. Mahouti, and M. A. Belen. Low-Cost and Highly Accurate
       Behavioral Modeling of Antenna Structures by Means of Knowledge-Based Domain-
       Constrained Deep Learning Surrogates [J]. IEEE Transactions on Antennas and
       Propagation, 2023, 71(1): 105-118.
[29] Y. Su, Y. Yin, S. Li, H. Zhao and X. Yin. Bandwidth Improvement for Patch Antenna
       via Knowledge-Based Deep Reinforcement Learning [J]. IEEE Antennas and Wireless
       Propagation Letters, 2024, 23(12): 4094-4098.
[30] N. Sarker, P. Podder, M. R. H. Mondal, S. S. Shafin and J. Kamruzzaman. Applications
       of Machine Learning and Deep Learning in Antenna Design, Optimization, and
       Selection: A Review [J]. IEEE Access, 2023, 11: 103890-103915.
[31] J. H. Kim and S. W. Choi. A Deep Learning-Based Approach for Radiation Pattern
       Synthesis of an Array Antenna [J]. IEEE Access, 2020, 8: 226059-226063.
[32] Z. Zhou, Z. Wei, J. Ren, Y. Yin, G. F. Pedersen and M. Shen. Two-Order Deep Learning
       for Generalized Synthesis of Radiation Patterns for Antenna Arrays [J]. IEEE
       Transactions on Artificial Intelligence, 2023, 4(5): 1359-1368.

[33] J. Zhang, M. O. Akinsolu, B. Liu, and S. Zhang. Design of Zero Clearance SIW Endfire
       Antenna Array Using Machine Learning-Assisted Optimization [J]. IEEE Transactions
       on Antennas and Propagation, 2022, 70(5): 3858-3863.
[34] J. Tak, A. Kantemur, Y. Sharma, and H. Xin. A 3-D-Printed W-Band Slotted Waveguide
       Array Antenna Optimized Using Machine Learning [J]. IEEE Antennas Wireless
       Propagation Letters, 2018, 17(11): 2008-2012.
[35]   W. -Q. Deng, S. Xu, Z. Xu and S. -Y. Zhu. Automated Design of Millimeter-Wave
       Dielectric Transmitarray Antenna Using Tandem Network [J]. IEEE Antennas and
       Wireless Propagation Letters, 2024, 23(11): 3549-3553.
[36]   B. Liu et al.. An Efficient Method for Complex Antenna Design Based on a Self
       Adaptive Surrogate Model-Assisted Optimization Technique [J]. IEEE Transactions on
       Antennas and Propagation, 2021, 69(4): 2302-23151.
[37] C.-Y. Chan and P. M. Goggans. Multiobjective Design of Linear Antenna Arrays Using
       Bayesian Inference Framework [J]. IEEE Transactions on Antennas and Propagation,
       2014, 62(11): 5524-5530.
[38] Y. Liu et al.. An Efficient Method for Antenna Design Based on a Self-Adaptive
       Bayesian Neural Network-Assisted Global Optimization Technique [J]. IEEE
       Transactions on Antennas and Propagation, 2022, 70(12): 11375-11388.
[39] L. Cui, Y. Zhang, R. Zhang, and Q. H. Liu. “A Modified Efficient KNN Method for
       Antenna Optimization and Design [J]. IEEE Transactions on Antennas and Propagation,
       2020, 68(10): 6858-6866.
[40] L. -Y. Xiao, W. Shao, F. -L. Jin, B. -Z. Wang, and Q. H. Liu. Inverse Artificial Neural
       Network for Multiobjective Antenna Design [J]. IEEE Transactions on Antennas and
       Propagation, 2021, 69(10): 6651-6659.
[41] I. Goodfellow, Y. Bengio, and A. Courville. “6 Deep Feedforward Networks,” in Deep
       Learning [M]. Cambridge, MA, USA: MIT Press, 2016.
[42] D.-L. Wu, J. H. Chen, K. Y. Yang, W. J. Zhu, and L. H. Ye. A Compact Dual-Polarized
       Patch Antenna With L-Shaped      Short Pins [J]. IEEE Antennas Wireless Propagation
       Letters, 2023, 22(4): 689-693.
[43] L.-Y. Xiao, W. Shao, F.-L. Jin, and B.-Z. Wang. Multiparameter modeling with ANN
       for antenna design [J]. IEEE Transactions on Antennas and Propagation, 2018, 66(7):
       3718-3723.
[44] Q. Zeng et al.. Phase Modulation Technique for Harmonic Beamforming in Time-

      Modulated Arrays [J]. IEEE Transactions on Antennas and Propagation, 2022, 70(3):
      1976-1988.
[45] Y. Wang, Q. Yang, H. Wang, and Y. Zeng. Grating Lobe Suppression for Distributed
      Phased Array via Accumulated Array Pattern Synthesis [J]. IEEE Antennas Wireless
      Propagation Letters, 2023, 22(7): 1527-1531.
[46] N. Jin and Y. Rahmat-Samii. Advances in Particle Swarm Optimization for Antenna
      Designs: Real-Number, Binary, Single-Objective and Multiobjective Implementations
      [J]. IEEE Transactions on Antennas and Propagation, 2007, 55(3): 556-567.
[47] L. Cen, Z. L. Yu, W. Ser, and W. Cen. Linear Aperiodic Array Synthesis Using an
      Improved Genetic Algorithm [J]. IEEE Transactions on Antennas and Propagation, 2012,
      60(2): 895-902.
[48] B. V. Ha, M. Mussetta, P. Pirinoli, and R. E. Zich. Modified Compact Genetic Algorithm
      for Thinned Array Synthesis [J]. IEEE Antennas Wireless Propagation Letters, 2016, 15:
      1105-1108.
[49] X. Li and K. M. Luk. The Grey Wolf Optimizer and Its Applications in Electromagnetics
      [J]. IEEE Transactions on Antennas and Propagation, 2020, 68(3): 2186-2197.
[50] D. G. Kurup, M. Himdi, and A. Rydberg. Synthesis of uniform amplitude unequally
      spaced antenna arrays using the differential evolution algorithm [J]. IEEE Transactions
      on Antennas and Propagation, 2003, 51(9): 2210-2217.
[51] S. Fang, W. Li, Z. Xue, and W. Ren. Synthesis of Distributed Array Consisting of Two
      Subarrays via Hybrid Method of Differential Evolution Optimization and Convex
      Optimization [J]. IEEE Antennas Wireless Propagation Letters, 2021, 20(2): 125-129.
[52] Z. D. Zaharis, I. P. Gravas, P. I. Lazaridis, T. V. Yioultsis, C. S. Antonopoulos, and T. D.
      Xenos. An Effective Modification of Conventional Beamforming Methods Suitable for
      Realistic Linear Antenna Arrays [J]. IEEE Transactions on Antennas and Propagation,
      2020, 68(7): 5269-5279.
[53] K. Zhong, J. Hu, Y. Cong, G. Cui, and H. Hu. RMOCG: A Riemannian Manifold
      Optimization-Based Conjugate Gradient Method for Phase-Only Beamforming
      Synthesis [J]. IEEE Antennas Wireless Propagation Letters, 2022, 21(8): 1625-1629.
[54] Z. Zheng, Y. Fu, and W. -Q. Wang. Sparse Array Beamforming Design for Coherently
      Distributed Sources [J]. IEEE Transactions on Antennas and Propagation, 2021, 69(5):
      2628-2636.
[55] B. Jalal, X. Yang, Q. Liu, T. Long, and T. K. Sarkar. Fast and Robust Variable-Step-Size

      LMS Algorithm for Adaptive Beamforming [J]. IEEE Antennas Wireless Propagation
      Letters, 2020, 19(7): 1206-1210.
[56] I. Mallioras, Z. D. Zaharis, P. I. Lazaridis, and S. Pantelopoulos. A Novel Realistic
      Approach of Adaptive Beamforming Based on Deep Neural Networks [J]. IEEE
      Transactions on Antennas and Propagation, 2022, 70(10): 8833-8848.
[57] S. Mirjalili, S. Mohammad Mirjalili, and A. Hatamlou. Multi-Verse Optimizer: a nature-
      inspired algorithm for global optimization [J]. Neural Computing and Applications,
      2016, 27(2): 495-513.
[58] C. A. Balanis. “Arrays: Linear, Planar, and Circular,” in Antenna Theory: Analysis and
      Design [M]. NJ, USA: Wiley, 2016.
[59] I. P. Gravas, Z. D. Zaharis, T. V. Yioultsis, P. I. Lazaridis, and T. D. Xenos. Adaptive
      Beamforming With Sidelobe Suppression by Placing Extra Radiation Pattern Nulls [J].
      IEEE Transactions on Antennas and Propagation, 2019, 67(6): 3853-3862.

淝   Ｉ

Ｓ
＿

，


）
 
       Ｍ

        議
        
        


## 基于深度学习与代理模型的天线优化设计研究进展_汤昊臻：全文

> 来源：文献/基于深度学习与代理模型的天线优化设计研究进展_汤昊臻.pdf
> 提取范围：题名、摘要、正文、参考文献

基于深度学习与代理模型的天线优化设计研究进展

基于深度学习与代理模型的
天线优化设计研究进展
□文 / 汤昊臻 1，吴季航 1，张伟 1,2*

（1. 北京邮电大学 电子工程学院，北京 100876；2. 信息光子学与光通信国家重点实验室，北京 100876）

  摘要：随着现代无线通信的快速发展,通信系统正在对天线设计提出宽带、高增益、小型化等
  一系列要求。高性能的要求往往需要复杂的天线结构,这将带来一系列挑战,如参数高维化、
  计算成本高昂和研发周期延长等。为应对不断增长的高性能天线需求,本文聚焦于人工神经网
  络与机器学习在天线设计方面的应用现状与发展趋势,系统总结了深度神经网络、基于知识的
  神经网络、逆向网络和自适应进化网络等核心算法原理,探讨了结合神经网络或深度学习进行
  天线设计的可行性、优势和所面临的问题。之后,本文展示了基于神经网络空间映射技术的超
  表面圆极化天线优化实例,该实例结果表明,采用空间映射策略仅需 4 次迭代即可收敛,成功
  实现了 1.98 GHz ～ 2.2 GHz 频段内回波损耗优于 -20 dB、轴比小于 3 dB 的优化目标。相较于
  耗时 34 小时的传统遗传算法与耗时 6.8 小时的传统神经网络,该方法的计算效率分别提升了
  约 10 倍、2 倍。基于深度学习与代理模型的天线优化技术正推动着天线设计从传统仿真试错
  向模型快速预测转变,为天线设计提出了新的解决方案和技术路径。

  关键词：天线设计；人工神经网络；机器学习；代理模型；空间映射；拓扑优化

  中图分类号：TP18；TN820        文献标志码：A 文章编号：2096-5036(2026)01-0049-17

  DOI：10.16453/j.2096-5036.202606

  0 引言                                        从传统天线的几个几何结构变量到多层叠、多
                                              耦合的新型天线，或者周期排列的复杂超表面
         随着无线通信的快速发展，一些复杂结构                   单元，甚至拓扑优化中的高维数像素点状态，
  的天线（如超宽带天线、可重构天线，异形天                        这些结构一方面确实丰富了天线的创新，但是
  线等）的设计周期和设计难度也随之增加                    。
                                    [1-3]
                                              同时也带来了计算成本大、设计周期长的问题。
  天线设计已不满足单一的阻抗匹配，需同时兼                        当前，天线设计的性能验证主要是基于传统的
  顾增益、轴比、旁瓣电平和交叉极化等多项指                        有限元法、矩量法或时域有限差分法等电磁计
  标       。这种多目标大大增加了设计的自由度。
      [4-8]
                                              算方法 [9-11]。尽管这些方法能够提供准确的电

     磁场解，但仿真耗时确实也很长，尤其是对于                           结合先验知识的空间映射技术，将优化分步
     复杂的天线结构，或者对于精密的网格剖分。                           骤进行，先进行粗略模型搭建，然后以少量高
     当结合遗传算法或粒子群优化等全局优化算法                           精度数据校正粗略模型，减少高精度数据的
     进行设计空间搜索时，往往需要调用长时间的                           依赖 [25,26]；或者通过结合主动学习与迁移学习
     全波仿真进行试错，这进一步增加了设计的成                           的策略，实现样本集的复用，一个样本支持多
     本         。
         [12-15]
                                                    类天线设计，能够有效缓解电磁数据获取成本
            为缓解天线设计中的计算成本问题，基于                      高的难题。
     深度学习与代理模型的天线优化技术提出了新
     的解决方案和技术路径 [16-18]。神经网络与机器                     1 神经网络与算法模型基础
     学习的拟合能力为建立天线结构参数与电磁响
     应之间的非线性映射提供了新的数学工具。区                           1.1 前馈神经网络
     别于传统插值方法，训练完备的神经网络能够                             前馈神经网络是指神经元之间的连接不形
     学习高维参数空间中的复杂电磁规律，作为高                           成循环的人工神经网络，通常被用于解决非动
     精度的代理模型，替代昂贵的全波仿真求解器。                          态建模问题。其中，多层感知器是较常用的前
     一个构建完成的代理模型，其预测速度大大缩                           馈神经网络结构，被广泛应用于微波建模，包
     短，能够以极低的计算成本支持大规模优化迭                           括无源元件（如天线、滤波器等）结构参数优
     代。其中的逆向设计方法可根据目标 参数或                           化和有源器件（如二极管、三极管等）输出特
     远场性能指标直接预测出最优几何结构，从而                           性预测 [27,28]，这里重点关注其在天线领域的应
     实现目标到结构的反向解决方案 [19-24]。同时，                     用。主要是利用多层感知器构建从输入空间，
     对于模型样本获取难、高精度仿真代价高等问                           如天线的几何尺寸、结构参数，到输出空间，
     题，业界正提出越来越多的解决方案。比如，                           如 参数、增益、轴比的非线性映射。

                                              隐藏层
                         输入层                                   输出层

                                                                     Data Flow

                                         图 1 典型多层感知器的神经网络概念图

                                 基于深度学习与代理模型的天线优化设计研究进展

    一个典型的全连接多层感知器由输入层、                      得益于其强大的表征能力，前馈神经网络
若干隐藏层和输出层构成，各层之间通过权重                     适用于处理高维输入数据，这在微波器件建模
连接，一个最简单的三层感知器如图 1 所示。                   中的效果是很可观的 [30]。
对于一个包含 层的网络，第 层神经元的输出
向量    可表示为前一层输出的线性变换经非                   1.2 基于知识的神经网络
线性激活函数处理后的结果：                               尽管多层感知器对非线性问题可以做到良
                                   (1)   好的拟合预测，但它同样存在着训练集依赖问
其中，    、       分别为第 层的权重矩阵与偏置            题，数据集的优劣将直接决定预测效果的好坏，
向量矩阵，          为 Sigmoid、Tanh 或 ReLU 等   这与想要减少高精度训练集的初衷不合。为了
非线性激活函数。                                 有效利用已有的微波物理知识，Feng 等提出
    在天线设计中，前馈神经网络的训练可以                   基于知识的神经网络（KBNN），将先验知识与
看作是对权重 的优化，利用仿真软件得出天                     数据驱动模型集合，提升模型的数据利用效率，
线的性能作为训练集，通过反向传播调整权重                     减少数据依赖 [31]。
，最小化预测响应               与全波仿真真值    之间        基于知识的神经网络一般包含“粗糙模
的损失函数              :                     型”与“精细模型”两部分。粗糙模型         通
                                         常基于经验公式、等效电路模型或低精度快速
                                   (2)
                                         仿真算法，能够提供天线响应的物理趋势，为
    但是，对于一些具有急剧变化的响应曲                    天线优化指明方向，但其精度有限，无法直接
线，多层感知器往往表现出训练难、准度低                      作为最终模型预测输出响应；精细模型         则
的问题。针对这一点，常选用更针对局部优                      是一个神经网络，用于学习粗糙模型与高精度
化的径向基函数网络。设计中一般采用具有                      全波仿真真值之间的偏差或映射函数。一种典
局部特征的函数，比如，可以使用高斯函数                      型的加法型基于知识的神经网络输出如图 2 (a)
                         作为激活函数，输        所示，其数学表达式为：
出                      为这些响应的线性组                                  (3)
合。相比于采用全局逼近方式的多层感知机，                        基于这种将物理先验知识与神经网络融
建模效果更好，训练过程也更为简便。这主要                     合，校正粗细模型差异的思想，主要实施方法
得益于网络结构对输入输出映射中的局部非                      可分为四类 [32-35]。
线性行为拟合，有利于缓解响应曲线急剧变                         1）差 分 方 法：数 学 表 达 为
化的问题。此外，一些其他的网络结构同样能                                      ，类同加法型基于知识
达到这种效果，比如小波神经网络主要就是                      的神经网络，模型的核心思想在于校正粗糙模
针对高度非线性或存在突变的响应曲线拟合。                     型的误差。首先，利用粗模型提取主要物理非
同样，如果对准确度要求宽松一些，极限学习                     线性，然后基于神经网络开始拟合残差函数，
机作为单隐藏层前馈神经网络的典型，可以                      基于先验知识保证了残差函数具有小动态范围
在有限的训练数据集下，快速完成电磁参数                      和平滑曲面特性，这种做法可以让神经网络易
的粗略建模   [29]
               。                         于收敛，可以减少训练样本的需求。

                                                先验知识                 空间映射
                  通用结构
                                                 输入                  神经网络

        神经网络                          神经网络                  神经网络
                                                              输出映射
                            全波电磁                   全波电磁              隐式     全波电磁
     先验知识    神经网络                                           先验知识
                             仿真    先验知识             仿真               映射      仿真
                                                              输入映射
        神经网络                                                神经网络

        (a) 通用概念模型                   (b) 先验知识输入模型           (c) 空间映射神经网络模型

                                    图 2 基于知识的神经网络模型图

       2）多 知 识 嵌 入 方 法：将 多 个 粗 糙 模 型                                           (6)
                            的预测同时嵌入网络，           使得在该映射空间下，粗糙模型的响应能够复
     神经网络充当融合器，学习其自适应加权融合，                       现精细模型的行为：
     从而获得比任何单一先验模型更广的有效覆盖                                                      (7)

     空间：                                            同样地，映射模型训练完后，设计迭代将
                                          (4)    在粗模型中完成，之后以少量高精度数据完成
       这里的粗模型能够探索更为广阔的设计空                        粗模型到细模型的误差校正，由于粗模型仿真
     间，粗细模型集成学习，可以有效加快模型的                        获取难度低，优化迭代速度快，这有利于大幅
     训练过程，同时保证了模型的鲁棒性。                           提升设计与优化的效率空间，如此一来，复杂
       3）先验知识输入方法：主要思想就是利用                       的电磁优化问题就转化为寻找空间映射函数的
     粗模型输出与真实响应之间的相关性为训练提                        问题 [36]。
     供指引，粗模型的输出                    将作为神经网           基于先验知识的神经网络就是通过在网络
     络的附加输入，这可以对输入空间作出扩展。                        训练过程中或者在网络训练之前引入相关的经
                                          (5)    验公式或者粗模型的物理约束，从而减少高精
       利用粗细模型的相关性简化神经网络的映                        度样本的依赖，能够在数据需求、模型精度和
     射复杂度，将一部分的复杂度移交给更为简单                        鲁棒性上取得平衡，这对于需要高成本的电磁
     的粗模型完成，通过先验知识的相关程度与精                        仿真模型是一个行之有效的解决途径 [37-39]。
     细模型的数据量在仿真精度和训练难易程度上
     做出平衡。                                       1.3 逆向神经网络结构
       4）空间映射神经网络：空间映射技术是微波                         一个完整的天线设计要能够根据预期性能
     工程领域的经典优化算法。该技术被引入作为神                       指标设计所对应的天线结构，与正向代理模型
     经网络的先验模型，在基于知识的神经网络框架                       不同，逆向神经网络致力于根据目标找到合适
     下，神经网络被训练用于寻找一个从精细参数空                       的天线输入参数。但是，在天线设计中，往往
     间 到粗糙参数空间 的非线性映射 ：                          需要多目标一起满足，这就对优化方法作出了

                            基于深度学习与代理模型的天线优化设计研究进展

要求，因为一对多的映射往往会使网络难以收                网络的训练与遗传算法、粒子群优化等优化算
敛。为应对多目标设计需求，逆向神经网络常                法结合 [41]，这可以减少数据集的浪费，将大规
采用多分支并行结构。一个需要同时满足 参                模训练样本获取改为了先小样本训练，再结合
数、增益和远场性能的设计任务，其映射关系                优化算法判断样本范围需求，之后再生成样本
可描述为：                               进行训练，这常常用在神经网络缺乏某些先验
                              (8)   数据或设计空间未知的情况下。具体而言，就
其中，   代表针对第 类物理指标进行特征提              是借助闭环演化思想，利用两个合作协同的网
取的子网络，           为特征融合与回归网络。        络，在这里，一个网络用于生成设计参数，这
在这里，网络输入层被解耦为多个独立的并行                可能是一个训练好的逆向网络；一个用于快速
分支进行处理。                             评估性能，一般引入传统优化算法，通过梯度
                                    下降更新权重，或者利用进化策略动态调整网
                        输入层
                                    络的权重参数。这将使得网络能够同步进行学
                                    习、判断和生成，即系统利用当前预测结果指
                        隐藏层         导新的全波仿真采样，并将新样本实时加入训
                                    练集用以更新网络。这种思想是以最小的仿真
  前馈
                                    代价实现设计收敛，但是有可能会出现局部最
 神经网络                   输出层

                     先验知识
                                    优的情况。

  逆向                    输入层
 神经网络                               1.5 高斯过程与降维模型
                                      前述神经网络常用于天线设计领域的参数
                        隐藏层         优化问题，即拓扑结构已知，优化参数。不过，
                                    为了探索天线设计的极限问题，现在有高维数
                                    拓扑优化的天线设计，这就需要统计学习的方
                        输出层
                                    法与降维技术。其中，高斯过程回归常被提及，

 图 3 使用正向神经网络获取先验知识的正逆向神            其得益于非参数化的思想，可以应用于小样本
          经网络结合模型                   条件下的代理建模。高斯过程不再是传统确定
  此外，为解决逆向训练数据获取难题，结                性预测的思想，而是选择提供预测值的概率分
合高精度的正向网络作为先验知识产生大量可                布。对于输入 ，其输出     服从高斯过程：
靠的虚拟样本，成为在低成本下充分训练逆向                                      (9)
网络的有效策略      。图 3 即为一种正逆向神经
          [40]
                                    其中，   为均值函数，    为协方差核函数。
网络结合的模型。                              当前，像素化天线越来越多被提及，而且
                                    一些复杂超表面结构或者一些高自由度的天线
1.4 自适应与进化神经网络                      拓扑，它们的变量维度往往不再是几个特定的
  自适应与进化神经网络吸收了原来传统优                几何结构参数，而是通过数字比特的思想转化
化设计中的思想，引入动态演化机制，将神经                为了高维像素阵列。传统方法在这里完全不再

     适用，结构不再是手动可调的规则几何，直接                       Enc、解码器 Dec 两部分完成。其中，编码器
     构建代理模型不可行                 。为了解决这种复杂几
                            [42]
                                                负责将高维像素化或网格化几何 压缩为低维
     何与计算模型之间的矛盾，这就需要采用降维                       潜在向量 ：
     技术将高维离散的拓扑结构映射到低维连续的                                            (11)

     像素流。此时，一些具有统计规律（也就是结                       解码器则负责从潜在向量重构几何结构：
     构相似）的几何群首先被捕捉，利用线性特征                                            (12)
     提取这种结构作为整体结构的主成分，之后这                         在此框架下，神经网络代理模型构建在低
     些主成分，可以通过正交变换（如旋转、平移、                                   （即
                                                维潜在空间 与电磁响应 之间        。
                                                                      ）
     重叠等）将复杂的高维几何拓扑映射为包含主                       优化算法在光滑、连续且维数较低的潜在空间
     要特征的低维几何主成分。                               中寻找最优解，最后通过解码器将其还原为
                                                物理可实现的复杂拓扑结构。这不仅降低了计
                                         (10)
                                                算复杂度，还能够通过潜在空间的正则化约束
     其中， 是从几何样本中提取的基函数， 是低                      保证生成结构的平滑性与可加工性。
     维设计系数， 是误差补偿量。通过这种方式，                        基于知识的神经网络通过在学习过程中引
     优化过程仅需针对少量的系数 进行，从而大                       入物理约束，成功在模型精度、训练成本和泛
     幅压缩搜索空间。                                   化能力之间取得了平衡；而结合几何降维技术，
       而针对更复杂的非线性拓扑特征，引入了                       进一步拓展了其在处理高维、复杂微波器件拓
     非线性流形学习思想，具体借助深度自动编码                       扑优化问题上的能力，是现代电磁辅助设计的
     器的强大降维能力，将几何结构降维由编码器                       重要发展方向。

                                                                     输出

      输入           跨通道层                多重小卷积层             全连接层

                                      (a) 卷积神经网络拓扑优化架构
           金属             非金属

                                        (b) 优化后的模型示例

                 图 4 文献 [43] 提出的用于滤波器拓扑优化的卷积神经网络拓扑优化架构与模型示例

                       基于深度学习与代理模型的天线优化设计研究进展

2 机器学习驱动的天线优化关键技术            种做法打破了传统思想直接将整个阵列几何作
                             为输入，巧妙地利用了有源单元方向图，在考
  本章探讨了具体的代理模型应用于天线设         虑互耦的情况下简化了训练难度，这主要是阵
计中的效果，借助神经网络和机器学习的非线         元互耦具有局部衰减特性，远处的阵元对中心
性拟合能力，用以缓解参数复杂性、仿真高成         阵元的影响足以忽略，构建了以子阵列特征为
本、多目标高性能的需求所带来的一系列挑战。        核心的输入机制。文中还有一大创新，即引入
通过列举实际案例，分析解决思路及方法效果，        了异构迁移学习策略，实现一个数据集多类天
概述了其中的算法原理和网络架构设计，并指         线设计复用的思想，大大缓解了不同规模阵列
出了其应用的天线模型。                  需重复生成数据集，占用昂贵的仿真资源等问
                             题。对于不同单元数量的大规模阵列性能，文
2.1 高保真代理模型与几何特征提取           章利用在小规模阵列数据集上训练好的模型，
  在处理大型非均匀阵列或相控阵设计           直接进行大规模的阵列预测，并且对于不等间
时，常利用深度神经网络替代全波电磁仿真          距排列的阵列，同样具有良好的远场性能预测。
的计算，解决传统数值计算方法耗时过长的          实验表明，在 128 单元的圆极化交叉偶极子阵
问题      。在这里，神经网络架构通常包含输入
     [44]
                             列综合中，该方法将全波仿真所需的 6 小时计
层、隐藏层，以及输出层。其中，输入层为阵列        算时间压缩至 0.235 秒，同时能够保持预测精
       如天线相位馈电或位置坐标，
的几何分布参数，                     度，达到了旁瓣电平误差 <0.5 dB，轴比误差
这将影响单元间距，隐藏层通过训练，调整权         <0.1 dB。这种将整体结构拆分为局部特征，利
重系数 ，使网络拟合输入参数与电磁响应之         用局部特征预测整体，充分结合了有源单元方
间的非线性映射关系，输出层一般为 参数、有        向图的思想，成功将计算复杂度由      降低
源单元方向图或其他远场性能指标。在具体实         至    。这一方法不仅实现了对非均匀阵列
施中，可以结合传统进化算法，如粒子群优化、        低旁瓣、低交叉极化和宽角扫描特性的快速预
遗传算法，进行适应度评估并生成下一次迭代         测，同时也为大规模阵列预测提出了新的解决
所需的样本，从而提升阵列综合的效率。           思路。文献 [45] 提出的网络架构与模型优化示
  例如，针对大型非均匀阵列设计中的互耦         例如图 5 所示。
问题，文献 [45] 提出了一种结合机器学习辅助
代理模型与有源单元方向图技术的解决方案。         2.2 基于深度神经网络的多目标逆向设计
网络输入为目标阵元与其前后邻域单元的几何             天线设计已不满足单一的阻抗匹配，需同
向量，借助有源单元方向图思想以解构整体阵         时兼顾增益、轴比、旁瓣电平和交叉极化等多
列，大大减少了训练的复杂度。之后利用两个         项指标。这种多目标大大增加了设计的自由度，
并行的深度神经网络分别建立子阵列几何参          直接构建从响应空间到几何参数空间的逆向映
数，这里就是前文的邻域几何向量与中心阵元         射网络容易收敛困难。为此，提出了多分支并
矢量有源单元方向图（Vec. AEP）的非线性映     行架构与知识辅助策略，以解耦多目标优化中
射。为了进一步简化训练，提取有源单元方向         不同物理量（如 参数、增益、方向图）在输入
图的特征向量，即正交的     与   作为输出。这   要求上的差异性，这同样是一种整体到局部的

                                 (a) 机器学习辅助的代理模型架构

                  (b) 圆极化交叉偶极子天线阵列与机器学习辅助优化的阵列右旋圆极化结果

                           图 5 机器学习辅助的天线阵列优化架构与优化结果示例 [45]

     简化思想，因为多目标综合复杂度总是大于单                   经网络架构。文献主要设计了三个独立的特征
     目标综合的。具体而言，采用了独立的并行分支                  提取子网络，分别对这三类目标进行独立训练，
     分别提取特征，不再仅仅关注单一层的传统设                   缓解了多目标下 参数、增益，以及辐射方向
     计，各分支的目标综合将再引入深层的神经网                   图的复杂性提高，训练难收敛的问题。三个分
     络进行训练，最终各目标在深层网络进行融合，                  支提取的异构特征将采用深层神经网络进行融
     有效缓解了多目标间的特征干扰、难以收敛的                   合，最终直接预测给出输出天线的几何结构参
     问题。同样地，为了减少训练成本，采用先验知                  数。实验结果表明，文中成功实现了给定目标
     识辅助，缓解全波电磁仿真耗时导致的高精度                   下，快速预测给出几何尺寸，这有效解决了多
     训练集难以获取问题。具体实施为训练一个高                   目标的特征冲突，且预测的准确率优于传统的
     精度的正向神经网络作为快速替代模型，并利                   单分支网络。
     用该正向神经网络扩展数据集，即再生成海量                      此外，文献 [47] 在 3D 打印介质透镜天线
     的虚拟数据。此操作扩充了训练集，有利于平滑                  的设计中，验证了知识型神经网络的高效性。
     训练过程，提高逆向网络的泛化能力与精度。                   针对透镜天线几何复杂、单次全波仿真极为耗
       以文献 [46] 为例，针对超宽带天线多目标               时等挑战，文章采用了先验知识嵌入策略，利
     难优化的问题，Xiao 等提出了多分支逆向神                 用 CST 仿真软件结合拉丁超立方采样获取少量

                           基于深度学习与代理模型的天线优化设计研究进展

                                  力耦合的复杂微波器件时，这种映射关系已无
                                  法采用某一函数代替，因为简单函数无法跨拟
                                  合多物理场之间的高维非线性偏差。为了解决
                                  这个问题，文献 [48] 中提出了多保真度传输函
                                  数网络，通过引入分层递进的映射，将复杂问
                                  题分布处理，缓解多物理场带来的复杂度增加。
                                     该方法不再试图一步到位地学习从低精度
                                  模型到多物理场模型的映射，而是构建了双层

        (a) 多模共振天线的几何模型           空间映射框架。第一层映射用于建立“粗网格
                                  电磁模型”与“细网格单物理场电磁模型”之间
                                  的关联。第二层映射用于建立“细网格单物理
                                                   （包含
                                  场电磁模型”与“高保真多物理场模型”
                                  电 - 热 - 力耦合，计算最慢）之间的关联。在此
                                  框架中，神经网络被参数化为非线性传输函数，
                                  分别学习各层级模型之间的输入空间映射和输
   (b) 多目标多分支神经网络优化后的结果对比         出校正参数。数学上，这种级联映射可表示为：
图 6 文献 [46] 中用于验证多目标多分支神经网络优化算法                                    (13)
        的多模共振天线模型与预测结果对比
                                     这种架构有效地将复杂的多物理场计算负
                                  担分解，利用中间层级（单物理场 em）作为“桥
                                  梁”，降低了神经网络对昂贵多物理场训练数
                                  据的依赖。

 图 7 3D 打印透镜天线模型与仿真实测对比结果 [47]

高保真样本，训练基础前馈神经网络，随后利
用该网络作为生成器构建了包含数万组“几何 -
性能”映射的增强数据集。这种做法显著降低
了对仿真数据的依赖，样本效率提升，大幅降                (a) 压电贴片天线模型   (b) 多物理场仿真与神经网络预测对比

                                  图 8 文献 [48] 中的验证双层多保真度空间映射算法的
低了高性能定制化介质天线的设计门槛。                      压电贴片天线模型与优化结果对比

2.3 基于物理知识的空间映射优化                    例如，文献 [48] 提出了一种基于人工神经
  传统的空间映射技术通过建立“粗糙模               网络的双层多保真度空间映射算法，利用神经
型”与“精细模型”之间的线性或简单非线性关             网络作为映射算子，训练神经网络学习从“粗
联来加速优化。但是，在处理涉及电磁 - 热 - 应         网 格 em”到“ 细 网 格 em”，以 及 从“ 细 网 格

     em”到“多物理场”的非线性偏差。这种算法                          量。为了在预测精度与计算效率之间取得平衡，
     可以在低成本的粗模型空间中进行快速搜索，                           架构中通常会集成卷积神经网络构成的分类
     扩展参数优化的空间，其中，关键节点选择调                           器，快速判别生成的拓扑结构是否满足连通性，
     用少量多物理场仿真进行校正。实验验证表明，                          是否物理可实现，从而剔除大量无效解；同时，
     该方法在保证设计精度的同时，显著减少了昂                           集成高斯过程回归或深度神经网络的回归器，
     贵的多物理场仿真次数。其天线结构与结果如                           对拓扑结构进行精细的性能预测，筛除不符合
     图 8 所示。                                        预期的无效结构，同时提供预测的不确定性估
                                                    计以指导采样。
     2.4 基于机器学习的天线几何拓扑演化与                              该类方法的具体实施通常嵌入在进化算
     多物理场协同优化                                       法的迭代循环中，需要边学习边预测边筛选。
       现在，像素化天线越来越被提及，其变量                           机器学习模型替代全波电磁仿真或流体动力学
     的维度往往不再是几个几何结构参数，而是利                           仿真对离散化为像素矩阵的设计个体进行适应
     用数字比特的思想转化为了高维数的像素阵                            度预评估，因为像素化的结构在传统仿真中将
     列。进行这种非规则的天线拓扑优化，其思路                           极大耗费时间。关键的实施策略在于模型的序
     就是在输入层采用像素化网格或二进制矩阵以                           列化与模型的实时更新。文献 [49] 的思路就是
     表征天线表面的金属分布，数字化天线结构。                           采用分步优化策略，先利用流体仿真数据训练
     之后，利用卷积神经网络的平移不变性以及局                           代理模型优化接地结构的拓扑以降低风阻，固
     部特征提取能力，捕捉天线几何拓扑中的边缘                           定该最优拓扑后再针对辐射单元进行电磁性能
     耦合及电流路径，旨在提取出结构的主成分或                           的拓扑演化；为了指导优化方向、减少成本高
     者典型结构，之后将整体映射为低维的特征向                           昂的精细模型仿真数据，同时采用主动学习策

         (a) 优化前后的 5G 基站天线模型对比                            (b) 优化前后的抗风阻效果对比

                                     图 9 机器学习辅助拓扑设计的天线模型与效果对比 [50]

                              基于深度学习与代理模型的天线优化设计研究进展

略，仅对代理模型预测表现优异或不确定性较                    载过大问题，研究者提出了一种序列多物理
高的拓扑结构调用真实物理仿真进行验证，边                    场机器学习辅助优化方法。金属接地结构转
筛选边预测边仿真验证，之后将新生成的仿真                    为像素化拓扑，利用深度神经网络训练风阻
数据实时回馈至训练集，通过不断修正代理模                    预测模型，优化出一种具有低风阻特性的创
型的决策边界，以最小的仿真代价逼近全局最                    新拓扑结构；随后，在此基础上，对偶极子辐
优拓扑，这在保证精度的同时，能大大缓解数                    射臂的几何形状进行了电磁优化。最终设计
据依赖性。                                   的 4×4 双极化阵列在保持优异电磁性能的同
     具体而言，文献 [49] 为了解决像素化天              时，通过拓扑优化实现了 73% 的风荷载降低，
线设计的效率问题，提出采用机器学习辅助                     同时考虑了多物理场与天线性能，这为传统
天线几何设计。文中针对一款用于互耦降低                     天线优化提供了新的方向。
的 E 形贴片天线及多频天线，采用了卷积神经
网络作为分类器筛选无效几何，高斯过程回                     3 基于神经网络空间映射技术的天线
归作为回归器预测 S 参数，其中，网络的输入                  优化具体实践
为高分辨率的二进制像素矩阵，以 0/1 二进制
编码代替金属 / 非金属的拓扑。相比于传统的                  3.1 算法原理与网络代理模型架构
遗传算法，该方法通过代理模型能够过滤大                       针对全波电磁仿真在复杂天线优化中计算
部分无效的全波仿真计算，实验表明，其不仅                    成本过高的问题，在之前的研究工作 [51] 中提出
成功设计出了满足多频段指标的复杂像素结                     了一种结合输入空间映射与神经网络的代理模
构，且收敛速度比传统方法快 3 倍以上，有效                  型优化策略。该模型采用超表面与圆极化天线
提升了在超高维设计空间中的寻优能力。文                     的粗网格剖分求解器作为粗略模型，具有更快
献 [50] 则进一步将机器学习拓扑优化扩展至                 收敛、更快迭代速度，但准确度欠缺的响应结
低风载基站天线的多物理场设计中，如图 9 所                  果，采用细网格剖分求解器作为精细模型，具
示，针对 Sub-6 GHz 频段的大规模阵列风荷               有更好的收敛精度、更长的仿真时间和高保真

     粗网格数据        整体模型评估      细网格数据

 用于粗模型                           用于输入
   训练                             映射层
                                   训练       由 ANN1 构建的粗网格剖分求解器的
       粗网格剖分求解器
                  训练
                                                    等效替代模型
         的等效替代

                                          由 ANN2 构建的几何参数
直接      输入几何参数                                输入映射模型            直接传递
传递       的空间映射    训练

        (a) 空间映射神经网络的概念模型                 (b) 应用于圆极化天线设计的算法模型

                       图 10 空间映射神经网络的概念化模型与具体应用模型

     性的响应结果。核心思想就是构建一个计算廉                         这一步需要建立的是粗细模型的校准项，
     价的“粗糙模型”以探究更为广阔的设计空间，                      给定的精细模型几何参量可以通过空间映射函
     以及为精细模型提供优化方向；之后高保真的                       数得到校准后的粗模型输入，进而得到校准后
     “精细模型”用于保证预测的精度，由于粗模                       的天线性能响应。同样地，该部分神经网络模
     型带来的先验知识，这里只需使用少量数据以                       型采用常用的多层感知器神经网络模型，包含
     完成粗细模型之间的误差校正。如图 10 所示，                    输入层、隐藏层和输出层。隐藏层采用非线性
     空间映射神经网络由一个训练好的神经网络映                       的 sigmoid 函数作为激活函数。由于粗细模型
     射模块与一个使用粗网格剖分结果训练的粗糙                       之间的映射关系通常较为简单，且非线性程度
     模型级联结合而成。                                  比较低，因此可以采用较少数量的隐藏神经元
       在数学表述上，设                    为精细模型的设计     构建该映射。
     变量空间，            为其对应的电磁响应（如                 在该架构中，粗网格剖分全波求解器的仿
     参数、轴比）；同理，设                   与   分别为粗     真结果作为训练集 1，用于构建反映圆极化天
     糙模型的变量与响应。粗模型计算成本低廉，                       线基础响应及其大致优化方向的粗模型代理模
     但结果常会产生“错位现象”，导致结果出现                       型；而精细网格剖分求解器的仿真结果则作为
     一定程度上的失真。为减少超表面圆极化天                        训练集 2，用于学习因网格剖分差异所引起的
     线精细模型与粗略模型之间因网格划分差异                        频率偏移与幅度误差，通过少量样本即可保证
     导致的响应错位现象，这种差异可以表示为                        精细模型的精度。即，粗网格数据能够提供宽
                       ，文中引入神经网络作为              泛广阔的设计空间探索，并为高精度准确响应
     非线性映射的代理模型                    ，建立从精细空间     的模型提供全局优化的指导，而细网格数据则
     到粗糙空间的映射关系：                                确保最终解的保真度和准确性。
                                         (14)
     其中， 为神经网络的权重参数。通过训练神                       3.2 超表面加载圆极化天线设计实例
     经网络，使得映射后的整体模型能够准确反映                         为验证上述算法的工程有效性，选取了超
     精细模型的响应关系。                                 表面加载圆极化交叉偶极子天线作为优化对
                                         (15)   象。如图 11 所示，基于传统交叉偶极子天线进

                             图 11 优化所使用的超表面圆极化天线仿真模型与实物模型

                            基于深度学习与代理模型的天线优化设计研究进展

行了蛇形蜿蜒线加载、末端弯折、不等长切割                网格剖分的等效替代模型采用三层前馈神经网
等操作。不等长蛇形蜿蜒线，有效延长了电流                络结构，并设置 sigmoid 激活函数，输入输出
路径，既能够降低谐振频率，又能够缩小天线                分别为由 HFSS 得到的粗模型响应以及给定的
辐射体尺寸；双分支不等长切割的偶极子拓扑                几何参量。同样地，输入映射函数也采用三层
与末端加载的非对称箭头尾端可以达到双频展                感知机模型，激活函数为 sigmoid 函数。文中
宽的目的；背面反射器可实现较宽的轴比波束                精细模型采用高精度的自适应网格剖分，以确
宽度，并为天线辐射提供了良好的前向增益。                保结果作为精度的可靠性；粗模型低剖分网格
整体结构由顶层的印刷交叉偶极子辐射单元与                密度以快速获取响应结果，其单次仿真耗时仅
底层的超表面反射板组成。圆极化辐射是通过                为精细模型的几分之一，虽然精度有差异，但
交叉偶极子的空心四分之一波长（             ）延迟     是其保留了关键的响应趋势与物理规律。
环以形成关键的 90°相位差，而偶极子拓扑形
成相等的正交幅度。底部的超表面阵列用电磁                3.3 优化实施流程与仿真加工验证
带隙结构替代传统金属反射板，利用其对反射                   实验设计阶段采用了实验设计法进行正交
相位的操控，可以在目标频段内得到随频率变                取点，生成 81 组粗网格样本用于对粗模型的代
化的相位，以补偿只能满足单一频率（中心频                理模型进行训练，49 组精细网格样本用于对输
率    ）下的反射板高度     ，从而优化相位敏感         入神经网络进行训练，兼顾广阔的设计空间探
的圆极化性能；同时，亚波长的基本结构也有                索与局部误差校准。此外，测试集采用另外的
利于尺寸的缩减，通过精心设计的电磁带隙基                随机的 20 组样本仿真数据构建，用于泛化性能
本结构，在维持或提升天线辐射性能的同时，                评估。粗网格剖分求解器的替代模型以及输入
将反射板面积压缩至半波长（对应于中心频率                映射的代理模型性能的评估，通过测试集与代
 ）以下。不仅作为反射地平面，还通过其独                理模型预测之间的均方根误差进行量化。
特的相位响应改善天线的轴比带宽与增益特                    模型精度的量化评估采用均方根误差作
性，实现整体小型化的目的。                       为核心指标。在第四次即最终优化迭代中，
     基于实际的设计需求，超表面圆极化天线             粗网格替代模型的训练误差、测试误差分别
的优化指标设定在 L/S 波段，具体要求在 1.98          为 1.46%、1.95%；完 整 基 于 空 间 映 射 神 经
GHz 至 2.2 GHz 的频带范围内，同时满足回波         网络代理模型的训练误差、测试误差在最终
损耗    小于 -20 dB 以及轴比 AR 小于 3 dB。由   迭 代 中 达 到 1.5%、2.06%。经 过 四 个 优 化
于结合超表面的天线设计复杂度高，参数扫描                周期，也就是四次完整的迭代过程后，预测
或直接优化方法难以快速收敛，仿真时间长，                结 果 出 现 收 敛，最 终 优 化 后 的 设 计 参 数 为
优化成本高，采用空间映射神经网络作为代理                                               。
模型进行天线设计优化。对于粗粒度模型，设                HFSS仿真结果、基于空间映射神经网络的预测
定收敛精度为 0.04，在 HFSS 中仅需 2 ～ 4 次      结果与实物测量的结果如图 12 所示，同时该图
网格划分迭代即可。相比之下，设定精细模型                展示了加工的实物图及暗室测量图。如图 12 所
的收敛精度为 0.01，这就需要更多网格迭代次             示，仿真结果与预测结果吻合良好，且都满足了
数与计算时间，也能够获得更精确的结果。粗                预先设定的优化目标，说明了方法的有效性。

                    图 12 基于空间映射神经网络的天线 HFSS 仿真、模型预测、加工实测对比图

        最终，超表面加载的圆极化交叉偶极子                     从计算总耗时看，传统的遗传算法进行了约 100
     天 线 在 1.98 ～ 2.2 GHz 目 标 频 段 实 现 了 S11   次迭代，并行计算总时间为 34 小时，其结果仍
     小于 -20 dB（实测小于 -18.4 dB）、轴比小于            出现不完全收敛的情况。传统的神经网络优化
     3 dB 的性能指标，圆满达成了预期的设计目                   耗时大大降低，约为 6.8 小时，但是需要 9 次迭
     标。实际上，S11 在 1.78 ～ 3.03 GHz 范围内          代才能完成最终收敛。而基于空间映射的优化
     （相对带宽 51.9%）始终保持小于 -10 dB，3                                ，而
                                              策略，尽管用了更多的数据（粗模型数据）
     dB 轴 比 带 宽 覆 盖 1.96-2.23 GHz（ 相 对 带 宽    且一次迭代的耗时更多，但是其仅需 4 次迭代即
     12.9%）。                                  可收敛，时长再次降低为 3.4 小时。粗模型探索
        此外，为进一步阐明模型的优势，评估计                    广阔的设计空间并为精细模型提供设计指导，
     算效率，表1将提出的空间映射神经网络方法与                    精细模型少量但精确，用于校正响应的差异，每
     遗传算法（GA）及传统人工神经网络优化进行                    次迭代的计算负担有很大一部分转移到了廉价
     了对比。通过采用空间映射技术，代理模型能够                    的粗模型上。这大大降低了神经网络对精细模
     在保持相当精度的同时，以更少的迭代次数及                     型的依赖，大大减少了优化所需时间，为天线优
     CPU耗时实现了更优的收敛速度。总体而言，                    化设计提出了切实可行的解决途径。

                                基于深度学习与代理模型的天线优化设计研究进展

                   表 1 使用不同优化方法优化超表面圆极化天线的性能比较

            优化方法         遗传算法           传统神经网络优化               基于空间映射神经网络优化

            迭代次数           100                  9                           4

 每次迭代所需的粗网格样本数              —                   —                          81

 每次迭代所需的细网格样本数              1                   49                         49

  每次迭代粗网格计算时间               —                   —                      7 min 30 s

  每次迭代细网格计算时间            20.4 min            36.7 min                   36.7 min

 HFSS 仿真评估所需的总时间       20.4 min×100       36.7 min×10              (36.7 + 7.5)min×5

代理模型训练和优化所需的时间              —               4.4 min×9                  5.2 min×4

  并行计算所需要的总时间             34 h*                6.8 h                      3.4 h

注：* 表示设计目标尚未满足。

4 面临的挑战与未来展望                          减少数据依赖、平衡精度和训练量的方向，目
                                      标是开发一种通用的预训练模型，使其能作为
     基于神经网络与机器学习代理模型进行                一个有利的粗模型或先验知识来对各种具体天
的天线设计仍面临着许多挑战。未来的一大重                  线做出设计指导 [55]。
要方向是发展物理信息神经网络。物理神经网
络不再是简单地将模型的输入输出响应给到网                  5 结语
络，而是嵌入电磁场的麦克斯韦方程，这将使
得预测不再是无边界的，可能违反实际的设计，                      随着无线通信技术发展，对天线的需求
网络在给出预测的同时能判断这些结果是否满                  将越来越高，天线将更加复杂，将面临计算成
足基本物理定律，有利于在少样本甚至无典                   本高、设计周期长等新问题。本文系统性地总
型数据的情况下，确保预测结果满足物理一                   结了深度学习与代理模型在天线优化领域的
致性      。这样不仅能够大幅降低网络的数据依
     [52]
                                      相关研究、所具有的优势，以及所面临的问题。
赖性，还能够提高模型的鲁棒性及快速预测能                  文章从基础的神经网络架构到应用于天线设
力，为电磁场的计算提供一种可以替代传统数                  计中的实际案例，分析了具体应用中如何在
值算法的解决思路。另外，目前越来越多的天                  预测精度和数据需求之间做出平衡，以及如
线设计不满足于对传统拓扑进行优化，而是利                  何借助策略来简化训练，加快优化进程。本文
用生成对抗网络及扩散模型构建新型拓扑，这                  用实例说明了数据驱动结合物理先验知识是
将有利于逼近天线设计的极限性能 [53]，目前也              解决高维、非线性、高复杂度天线设计的有力
已经成为天线优化的一大热门方向。这将带来                  途径，能够在保证精度的前提下将设计周期
非常高的设计自由度，一些传统设计难以达到                  大大缩短。
的结构已经在超表面、滤波器、天线等领域得
以验证 [54]。此外，针对神经网络的数据问题，              参考文献
高精度电磁仿真数据获取成本高昂，这衍生了                  [1] Liu Fulai, Zhou Wu, Qin Dongbao, et al. CAWE-ACNN

     algorithm for coprime sensor array adaptive beamforming[J].         11: 103890-103915.
     Sensors, 2024, 24(17): 5454.                                        [17] Khan M M, Hossain S, Mozumdar P, et al. A review on
     [2] Xue Cong, Zhu Hairui, Zhang Shurui, et al. Broadband            machine learning and deep learning for various antenna
     beamforming weight g eneration network based on                     design applications[J]. Heliyon, 2022, 8(4): e09317.
     convolutional neural network[J]. IEEE Geoscience and                [18] Katkevičius A, Plonis D, Damaševičius R, et al. Trends of
     Remote Sensing Letters, 2024, 21: 4501105.                          microwave devices design based on artificial neural networks:
     [3] Marzetta T L. Noncooperative cellular wireless with unlimited   a review[J]. Electronics, 2022, 11(15): 2360.
     numbers of base station antennas[J]. IEEE Transactions on           [19] Gadhafi R, Copiaco A, Himeur Y, et al. Exploring the
     Wireless Communications, 2010, 9(11): 3590-3600.                    potential of deep-learning and machine-learning in dual-
     [4] Sievenpiper D F, Dawson D C, Jacob M M, et al.                  band antenna design[J]. IEEE Open Journal of the Computer
     Experimental validation of performance limits and design            Society, 2024, 5: 566-577.
     guidelines for small antennas[J]. IEEE Transactions on              [20] Gosal G, Almajali E, McNamara D, et al. Transmitarray
     Antennas and Propagation, 2012, 60(1): 8-19.                        antenna design using forward and inverse neural network
     [5] Li Qianqian, Zhang Haifeng. A high-gain circularly              modeling[J]. IEEE Antennas and Wireless Propagation
     polarized antenna array based on a chiral metastructure[J].         Letters, 2016, 15: 1483-1486.
     IEEE Transactions on Antennas and Propagation, 2023, 71(4):         [21] Waqar N, Wong K K, Tong K F, et al. Deep learning
     3033-3041.                                                          enabled slow fluid antenna multiple access[J]. IEEE
     [6] Li Weiran, Guo Huankun, Wang Xiaoyi, et al. A 2-bit             Communications Letters, 2023, 27(3): 861-865.
     reconfigurable metasurface with real-time control for               [22] Lin Yi Lu D, Maman L, Earls J, et al. Optimization of
     deflection, diffusion, and polarization[J]. IEEE Transactions       antenna array configurations using deep learning[J]. IEEE Open
     on Antennas and Propagation, 2024, 72(2): 1521-1531.                Journal of Antennas and Propagation, 2025, 6(5): 1367-1374.
     [7] Wang Jiafu, Li Yue, Jiang Zhi hao, et al. Metantenna: when      [23] Yao Yifan, Park J, Jin Xianglan. Deep learning detection
     metasurface meets antenna again[J]. IEEE Transactions on            on multi-antenna quantize-forward relay channel based on
     Antennas and Propagation, 2020, 68(3): 1332-1347.                   maximum likelihood[J]. IEEE Transactions on Cognitive
     [8] Venkateshwaran S, Rao K K, Kumbha S R. Design and               Communications and Networking, 2026, 12: 1978-1991.
     implementation of circular polarized patch antenna using            [24] Shao Xiaodan, Hu Limei, Sun Yulong, et al. Hybrid near-
     EBG structure[C]//2022 IEEE Microwaves, Antennas, and               far field 6D movable antenna design exploiting directional
     Propagation Conference, December 12-16, 2022, Bangalore,            sparsity and deep learning[PP/OL]. V1. arXiv (2025-06-18)
     India. IEEE, 2022: 1792-1797.                                       [2026-01-20]. https://doi.org/10.48550/arXiv.2506.15808.
     [9] Garg A, Sharma A, Zheng Weiguang, et al. A review on            [25] Koziel S, Ogurtsov S. Antenna design by simulation-
     artificial intelligence-enabled mechanical analysis of 3D           driven optimization[M]. Cham: Springer International
     printed and FEM-modelled auxetic metamaterials[J]. Virtual          Publishing, 2014.
     and Physical Prototyping, 2025, 20: e2445712.                       [26] Bandler J W, Cheng Q S, Dakroury S A, et al. Space
     [10] Alhaj Hasan A, Nguyen T M, Kuksenko S P, et al. Wire-grid      mapping: the state of the art[J]. IEEE Transactions on
     and sparse MoM antennas: past evolution, present implementation,    Microwave Theory and Techniques, 2004, 52(1): 337-361.
     and future possibilities[J]. Symmetry, 2023, 15(2): 378.            [27] Zhang Q J, Gupta K C. Neural networks for RF and
     [11] David D S K, Jeong Y, Wu Yin chao, et al. An analytical        microwave design[M]. Boston: Artech House, 2000.
     antenna modeling of electromagnetic wave propagation                [28] Zhang Wei, Feng Feng, Jin Jing, et al. Parallel
     in inhomogeneous media using FDTD: a comprehensive                  multiphysics optimization for microwave devices exploiting
     study[J]. Sensors, 2023, 23(8): 3896.                               neural network surrogate[J]. IEEE Microwave and Wireless
     [12] Rahmat-Samii Y, Gies D, Robinson J. Particle swarm             Components Letters, 2021, 31(4): 341-344.
     optimization (PSO): a novel paradigm for antenna designs[J].        [29] Zhang Wei, Feng Feng, Liu Wenyuan, et al. Advanced
     URSI Radio Science Bulletin, 2003, 2003(306): 14-22.                parallel space-mapping-based multiphysics optimization
     [13] El Misilmani H M, Naous T, Al Khatib S K. A review             for high-power microwave filters[J]. IEEE Transactions on
     on the design and optimization of antennas using machine            Microwave Theory and Techniques, 2021, 69(5): 2470-2484.
     learning algorithms and techniques[J]. International Journal        [30] Feng Feng, Na Weicong, Jin Jing, et al. Artificial neural
     of RF and Microwave Computer-Aided Engineering, 2020,               networks for microwave computer-aided design: the state
     30(10): e22356.                                                     of the art[J]. IEEE Transactions on Microwave Theory and
     [14] Cha Haotian, Fallahi H, Dai Yuchen, et al. Multiphysics        Techniques, 2022, 70(11): 4597-4619.
     microfluidics for cell manipulation and separation: a               [31] Feng Feng, Zhang Jianan, Zhang Wei, et al. Coarse- and
     review[J]. Lab on a Chip, 2022, 22(3): 423-444.                     fine-mesh space mapping for EM optimization incorporating
     [15] Zhu Yi, Schenk M, Filipov E T. A review on origami             mesh deformation[J]. IEEE Microwave and Wireless
     simulations: from kinematics, to mechanics, toward                  Components Letters, 2019, 29(8): 510-512.
     multiphysics[J]. Applied Mechanics Reviews, 2022, 74(3): 030801.    [32] Watson P M, Gupta K C. EM-ANN models for
     [16] Sarker N, Podder P, Mondal M R H, et al. Applications          microstrip vias and interconnects in dataset circuits[J]. IEEE
     of machine learning and deep learning in antenna design,            Transactions on Microwave Theory and Techniques, 1996,
     optimization, and selection: a review[J]. IEEE Access, 2023,        44(12): 2495-2503.

                                                        基于深度学习与代理模型的天线优化设计研究进展

[33] Watson P M, Gupta K C, Mahajan R L. Applications               [45] Yang Xin, Zhao Yanwen, Wan Mi, et al. Circularly
of knowledge-based artificial neural network modeling to            polarized antenna array synthesis based on machine-learning-
microwave components[J]. International Journal of RF and            assisted surrogate modeling[J]. IEEE Transactions on
Microwave Computer-Aided Engineering, 1999, 9(3): 254-260.          Antennas and Propagation, 2024, 72(2): 1469-1482.
[34] Wang Fang, Zhang Qi Jun. Knowledge-based neural                [46] Xiao Liye, Shao Wei, Jin Fulong, et al. Inverse artificial
models for microwave design[J]. IEEE Transactions on                neural network for multiobjective antenna design[J]. IEEE
Microwave Theory and Techniques, 1997, 45(12): 2333-2343.           Transactions on Antennas and Propagation, 2021, 69(10):
[35] Na Weicong, Feng Feng, Zhang Chao, et al. A unified            6651-6659.
automated parametric modeling algorithm using knowledge-            [47] Liu Yanfang, Peng Lin, Shao Wei. An efficient knowledge-
based neural network and l1 optimization[J]. IEEE                   based artificial neural network for the design of circularly
Transactions on Microwave Theory and Techniques, 2017,              polarized 3-D-printed lens antenna[J]. IEEE Transactions on
65(3): 729-745.                                                     Antennas and Propagation, 2022, 70(7): 5007-5014.
[36] Jin Jing, Zhang Chao, Feng Feng, et al. Deep neural            [48] Yang Xiao, Wang Zimeng, Hu Haitian, et al. Multifidelity
network technique for high-dimensional microwave modeling           space-mapping-based approach for accelerated multiphysics
and applications to parameter extraction of microwave               optimization of microwave devices[J]. IEEE Transactions on
filters[J]. IEEE Transactions on Microwave Theory and               Microwave Theory and Techniques, 2025, 73(12): 9902-9919.
Techniques, 2019, 67(10): 4140-4155.                                [49] Wu Qi, Chen Weiqi, Yu Chen, et al. Machine-learning-
[37] Liu Zhijun, Hu Xin, Liu Ting, et al. Attention-based           assisted optimization for antenna geometry design[J]. IEEE
deep neural network behavioral model for wideband                   Transactions on Antennas and Propagation, 2024, 72(3): 2083-
wireless power amplifiers[J]. IEEE Microwave and Wireless           2095.
Components Letters, 2020, 30(1): 82-85.                             [50] Han Biying, Wu Qi, Yu Chen, et al. Low-wind-load
[38] Zhang Sun, Hu Xin, Liu Zhijun, et al. Deep neural              broadband dual-polarized antenna and array designs
network behavioral modeling based on transfer learning for          using sequential multiphysics machine-learning-assisted
broadband wireless power amplifier[J]. IEEE Microwave and           optimization[J]. IEEE Transactions on Antennas and
Wireless Components Letters, 2021, 31(7): 917-920.                  Propagation, 2025, 73(1): 135-148.
[39] Simsek M, Zhang Q J, Kabir H, et al. The recent                [51] Zhai Tong, Tang Haozhen, Liu Zuodong, et al. Advanced
developments in knowledge based neural modeling[J].                 space-mapping-based approach for accelerated optimization
Procedia Computer Science, 2010, 1(1): 1321-1330.                   of metasurface-combined circularly polarized antenna[J].
[40] Ojaroudi M, Bila S, Torrès F. A new approach of multi-         IEEE Antennas and Wireless Propagation Letters, 2025,
parameter UWB antenna modeling based on knowledge-based             24(11): 3991-3995.
artificial neural network[C]//12th European Conference              [52] Raissi M, Perdikaris P, Karniadakis G E. Physics-informed
on Antennas and Propagation (EuCAP 2018), London,                   neural networks: a deep learning framework for solving
UK. Institution of Engineering and Technology, 2018: 1-5.           forward and inverse problems involving nonlinear partial
DOI:10.1049/cp.2018.0497.                                           differential equations[J]. Journal of Computational Physics,
[41] Jin Yaochu. Surrogate-assisted evolutionary computation:       2019, 378: 686-707.
Recent advances and future challenges[J]. Swarm and                 [53] Hu Yanyan, Jin Yuchen, Wu Xuqing, et al. A theory-
Evolutionary Computation, 2011, 1(2): 61-70.                        guided deep neural network for time domain electromagnetic
[42] Liu Bo, Aliakbarian H, Ma Zhongkun, et al. An efficient        simulation and inversion using a differentiable programming
method for antenna design optimization based on evolutionary        platform[J]. IEEE Transactions on Antennas and Propagation,
computation and machine learning techniques[J]. IEEE                2022, 70(1): 767-772.
Transactions on Antennas and Propagation, 2014, 62(1): 7-18.        [ 5 4 ] A n S e n s o n g , Z h e n g B o we n , Ta n g H o n g , e t a l .
[43] Yang Yiqi, Zhang Wei, Jin Jing, et al. A cross-channel         Multifunctional metasurface design with a generative
parametric CNN framework for microwave topology                     adversarial network[J]. Advanced Optical Materials, 2021,
modeling[J]. IEEE Transactions on Microwave Theory and              9(5): 2001433.
Techniques, 2025(99): 1-17.                                         [55] Meng Yuquan, Dong Zhiqiao, Lu K C, et al. Meta-
[44] Stanković Z Ž, Olćan D I, Dončov N S, et al. Consensus deep    learning-based domain generalization for cost-effective tool
neural networks for antenna design and optimization[J]. IEEE        condition monitoring in ultrasonic metal welding[J]. IEEE
Transactions on Antennas and Propagation, 2022, 70(7): 5015-5023.   Transactions on Industrial Informatics, 2025, 21(1): 653-662.

